#include "demo_1f/tasks/task_manager.hpp"
#include "demo_1f/tasks/demo_recovery.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace demo_1f {
namespace {
TimePoint after(double seconds) {
  return Clock::now() + std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double>(seconds));
}
} // namespace

bool TaskManager::demo_busy() const {
  return demo_goal_ || pick_pending_ || pick_handle_ || demo_faulted_;
}

CupMotion * TaskManager::demo_cup_module(const std::string & action) {
  for (auto & module : cup_motions_) {
    if (module.action_name == action) return &module.motion;
  }
  return nullptr;
}

void TaskManager::setup_demo_server() {
  demo_max_retries_ = node_.declare_parameter<int>("demo.max_retries", 2);
  demo_retry_delay_ = node_.declare_parameter<double>("demo.retry_delay", 1.0);
  demo_pick_timeout_ = node_.declare_parameter<double>("demo.pick_timeout", 180.0);
  if (demo_max_retries_ < 0 || demo_retry_delay_ < 0 || demo_pick_timeout_ <= 0) {
    throw std::invalid_argument("Invalid demo retry/timeout parameters");
  }
  reset_demo_service_ = node_.create_service<std_srvs::srv::Trigger>("~/reset_demo",
      [this](std_srvs::srv::Trigger::Request::ConstSharedPtr,
          std_srvs::srv::Trigger::Response::SharedPtr response) {
        response->success = !shutdown_requested_ && !demo_busy() && !goal_start_pending_ &&
            !active_goal_handle_ && !cup_goal_ && !motion_.running() && !cup_motion_blocked();
        if (!response->success) {
          response->message = "Cannot reset while busy or stop/scene is unconfirmed";
          return;
        }
        demo_resumable_ = false;
        demo_completed_stage_.clear();
        demo_stage_.clear();
        context_.set_task("IDLE", "", "", "Demo record reset by operator; no motion commanded");
        response->message = "Demo checkpoint reset; no motion commanded";
      });
  pick_client_ = rclcpp_action::create_client<PickAction>(&node_, "/dynamixel_pick/execute");
  demo_server_ = rclcpp_action::create_server<DemoAction>(&node_, "/run_demo",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const DemoAction::Goal> request) {
        std::string reason;
        const bool resume = request->command == "resume";
        const bool initial_only = request->command == "initial" || (resume && demo_initial_only_);
        const bool valid = resume || request->command == "initial" ||
            request->command == "start" || request->command == "pick";
        if (!valid || shutdown_requested_ || demo_busy() || goal_start_pending_ ||
            active_goal_handle_ || cup_goal_ || motion_.running() || cup_motion_blocked() ||
            (resume && !demo_resumable_) ||
            !context_.ready(reason) ||
            (!resume && request->command != "pick" &&
                (!initial_motion_ || !initial_motion_->is_ready_to_start(reason))) ||
            (!initial_only && (!resume || demo_stage_ == "INITIALIZE_ROBOT" || demo_stage_ == "PICK_DYNAMIXEL") &&
                !pick_client_->action_server_is_ready()) ||
            (!initial_only && request->command != "pick" &&
                (!demo_cup_module("/approach_cup") || !demo_cup_module("/refine_cup") ||
                    !demo_cup_module("/withdraw_cup")))) {
          RCLCPP_WARN(node_.get_logger(),
              "Rejected demo: %s", reason.empty() ?
              "busy, backend unavailable, invalid command, or no safe resume checkpoint" : reason.c_str());
          return rclcpp_action::GoalResponse::REJECT;
        }
        goal_start_pending_ = true;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<DemoHandle> handle) {
        if (handle != demo_goal_) return rclcpp_action::CancelResponse::REJECT;
        demo_cancel_requested_ = true;
        cancel_demo_motion();
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](const std::shared_ptr<DemoHandle> handle) {
        goal_start_pending_ = false;
        demo_goal_ = handle;
        demo_cancel_requested_ = shutdown_requested_;
        demo_retries_ = 0;
        demo_retry_wait_ = false;
        demo_stage_active_ = false;
        if (handle->get_goal()->command != "resume") {
          demo_stage_ = handle->get_goal()->command == "pick" ? "PICK_DYNAMIXEL" : "INITIALIZE_ROBOT";
          demo_completed_stage_.clear();
          demo_cup_id_ = handle->get_goal()->cup_id.empty() ? "cup" : handle->get_goal()->cup_id;
          demo_pick_only_ = handle->get_goal()->command == "pick";
          demo_initial_only_ = handle->get_goal()->command == "initial";
        } else {
          demo_pick_only_ = false;
        }
        demo_resumable_ = false;
        demo_detail_ = "Starting " + demo_stage_;
        context_.set_task("RUNNING", "right_to_left");
      });
}

void TaskManager::cancel_demo_motion() {
  if (pick_handle_ && !pick_cancel_sent_) {
    pick_cancel_sent_ = true;
    pick_client_->async_cancel_goal(pick_handle_);
  }
  if (demo_stage_ == "HANDOVER") motion_.cancel();
  if ((demo_stage_ == "INITIALIZE_ROBOT" || demo_stage_ == "RETURN_INITIAL") && initial_motion_)
    initial_motion_->cancel();
  if (active_cup_motion_ && demo_goal_) active_cup_motion_->cancel();
}

void TaskManager::start_demo_stage() {
  demo_stage_started_ = Clock::now();
  demo_stage_active_ = true;
  std::string reason;
  if (demo_stage_ == "INITIALIZE_ROBOT" || demo_stage_ == "RETURN_INITIAL") {
    if (!initial_motion_ || !initial_motion_->start(reason)) {
      finish_demo(false, "NOT_READY", reason.empty() ? "Initial pose module missing" : reason, true);
    }
    return;
  }
  if (demo_stage_ == "PICK_DYNAMIXEL") {
    pick_result_.reset();
    pick_handle_.reset();
    pick_pending_ = true;
    pick_cancel_sent_ = false;
    pick_phase_ = "PICK_STARTING";
    PickAction::Goal goal;
    goal.execute = true;
    goal.move_to_observe = false; // Full demo initializes first; pick-only uses the operator's current pose.
    rclcpp_action::Client<PickAction>::SendGoalOptions options;
    options.goal_response_callback = [this](PickHandle::SharedPtr handle) {
      pick_pending_ = false;
      pick_handle_ = handle;
      if (!handle) {
        pick_result_ = std::make_shared<PickAction::Result>();
        pick_result_->stop_confirmed = true;
        pick_result_->error_code = "PICK_REJECTED";
        pick_result_->message = "Pick backend rejected request";
      } else if (!demo_goal_ || demo_cancel_requested_ || demo_faulted_) {
        pick_cancel_sent_ = true;
        pick_client_->async_cancel_goal(handle);
      }
    };
    options.feedback_callback = [this](PickHandle::SharedPtr,
        const std::shared_ptr<const PickAction::Feedback> feedback) {
      pick_phase_ = feedback->phase;
      demo_detail_ = feedback->detail;
    };
    options.result_callback = [this](const PickHandle::WrappedResult & result) {
      pick_handle_.reset();
      pick_pending_ = false;
      pick_result_ = result.result;
      if (!pick_result_) {
        pick_result_ = std::make_shared<PickAction::Result>();
        pick_result_->error_code = "STOP_UNCONFIRMED";
        pick_result_->message = "Pick backend returned no result";
      } else if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
        pick_result_->success = false;
      }
    };
    try {
      pick_client_->async_send_goal(goal, options);
    } catch (const std::exception & error) {
      pick_pending_ = false;
      demo_faulted_ = true; // Dispatch outcome is unknown; do not send another motion.
      finish_demo(false, "STOP_UNCONFIRMED", error.what());
    }
    return;
  }
  if (demo_stage_ == "HANDOVER") {
    if (!motion_.start("right_to_left", reason)) {
      finish_demo(false, "HANDOVER_NOT_READY", reason);
    }
    return;
  }
  active_cup_motion_ = demo_cup_module(
      demo_stage_ == "APPROACH_CUP" ? "/approach_cup" :
      demo_stage_ == "REFINE_CUP" ? "/refine_cup" : "/withdraw_cup");
  const auto arm = demo_stage_ == "WITHDRAW_CUP" ? "both" : "left";
  if (!active_cup_motion_ || !active_cup_motion_->start(arm, demo_cup_id_, reason, true)) {
    finish_demo(false, "NOT_READY", reason.empty() ? "Cup module missing" : reason, true);
  }
}

void TaskManager::finish_demo(bool success, const std::string & code,
    const std::string & detail, bool resumable) {
  if (!demo_goal_) return;
  demo_resumable_ = resumable && !demo_faulted_ && !demo_cancel_requested_;
  auto result = std::make_shared<DemoAction::Result>();
  result->success = success;
  result->error_code = code;
  result->message = detail;
  result->last_completed_stage = demo_completed_stage_;
  result->holding_state = "UNVERIFIED";
  result->can_resume = demo_resumable_;
  context_.set_task(success ? "SUCCEEDED" : demo_resumable_ ? "WAITING_FOR_OPERATOR" : "FAILED",
      "right_to_left", code, detail);
  if (demo_goal_->is_canceling()) demo_goal_->canceled(result);
  else if (success) demo_goal_->succeed(result);
  else demo_goal_->abort(result);
  RCLCPP_INFO(node_.get_logger(), "Demo %s: %s", success ? "SUCCEEDED" : "STOPPED", detail.c_str());
  demo_goal_.reset();
  active_cup_motion_ = nullptr;
  demo_stage_active_ = false;
  demo_retry_wait_ = false;
  publish_status();
}

void TaskManager::fail_demo_stage(const std::string & code,
    const std::string & detail, bool retryable) {
  if (code == "STOP_UNCONFIRMED" || code == "SCENE_UNCONFIRMED") demo_faulted_ = true;
  retryable = retryable && !demo_faulted_ && !demo_cancel_requested_ && !cup_motion_blocked();
  if (retryable && demo_retries_ < demo_max_retries_) {
    ++demo_retries_;
    demo_stage_active_ = false;
    demo_retry_wait_ = true;
    demo_deadline_ = after(demo_retry_delay_);
    demo_detail_ = "Retry " + std::to_string(demo_retries_) + ": " + detail;
    return;
  }
  finish_demo(false, code, detail, retryable);
}

void TaskManager::tick_demo() {
  if (!demo_goal_) return;
  if (demo_cancel_requested_ && !demo_stage_active_) {
    finish_demo(false, "CANCELLED", "Cancelled between stages; grippers unchanged");
    return;
  }
  if (demo_retry_wait_) {
    if (Clock::now() >= demo_deadline_) demo_retry_wait_ = false;
  } else if (!demo_stage_active_) {
    start_demo_stage();
  } else {
    bool done = false;
    bool success = false;
    bool retryable = false;
    std::string code, detail;
    if (demo_stage_ == "PICK_DYNAMIXEL") {
      if (demo_cancel_requested_) cancel_demo_motion();
      if (!pick_result_ && elapsed(demo_stage_started_) > demo_pick_timeout_) {
        demo_cancel_requested_ = true;
        cancel_demo_motion();
        if (elapsed(demo_stage_started_) > demo_pick_timeout_ + 10.0) {
          demo_faulted_ = true;
          finish_demo(false, "STOP_UNCONFIRMED", "Pick backend timed out without a final stop result");
          return;
        }
      }
      if (pick_result_) {
        done = true;
        success = pick_result_->success && pick_result_->stop_confirmed;
        code = pick_result_->error_code;
        detail = pick_result_->message;
        retryable = pick_result_->retryable && !pick_result_->motion_started;
        if (!pick_result_->stop_confirmed) code = "STOP_UNCONFIRMED";
      }
    } else {
      const MotionResult * outcome = nullptr;
      if ((demo_stage_ == "INITIALIZE_ROBOT" || demo_stage_ == "RETURN_INITIAL") && initial_motion_) {
        initial_motion_->tick();
        if (!initial_motion_->running()) outcome = &initial_motion_->result();
      } else if (demo_stage_ == "HANDOVER") {
        motion_.tick();
        if (!motion_.running()) outcome = &motion_.result();
      } else if (active_cup_motion_) {
        active_cup_motion_->tick();
        if (elapsed(last_publish_) >= .2) publish_cup_target();
        if (!active_cup_motion_->running()) outcome = &active_cup_motion_->result();
      }
      if (!outcome && active_cup_motion_) demo_detail_ = active_cup_motion_->result().detail;
      if (outcome) {
        done = true;
        success = outcome->sequence_completed;
        code = outcome->code;
        detail = outcome->detail;
        retryable = demo_can_retry(demo_stage_, code);
      }
    }
    if (done) {
      if (demo_cancel_requested_) {
        if (code == "STOP_UNCONFIRMED" || code == "SCENE_UNCONFIRMED") demo_faulted_ = true;
        finish_demo(false, demo_faulted_ ? "STOP_UNCONFIRMED" : "CANCELLED",
            detail.empty() ? "Stopped; grippers unchanged" : detail);
        return;
      }
      if (!success) {
        fail_demo_stage(code.empty() ? "EXECUTION_FAILED" : code, detail, retryable);
      } else {
        demo_completed_stage_ = demo_stage_;
        demo_stage_active_ = false;
        demo_retries_ = 0;
        active_cup_motion_ = nullptr;
        if (demo_stage_ == "INITIALIZE_ROBOT") {
          if (demo_initial_only_) {
            finish_demo(true, "", "Initial pose reached; no pick started");
            return;
          }
          demo_stage_ = "PICK_DYNAMIXEL";
        } else if (demo_stage_ == "PICK_DYNAMIXEL") {
          demo_stage_ = "HANDOVER";
          if (demo_pick_only_) {
            finish_demo(true, "", "Pick complete; resume continues with handover", true);
            return;
          }
        } else if (demo_stage_ == "HANDOVER") demo_stage_ = "APPROACH_CUP";
        else if (demo_stage_ == "APPROACH_CUP") demo_stage_ = "REFINE_CUP";
        else if (demo_stage_ == "REFINE_CUP") demo_stage_ = "WITHDRAW_CUP";
        else if (demo_stage_ == "WITHDRAW_CUP") demo_stage_ = "RETURN_INITIAL";
        else {
          finish_demo(true, "", "Pick, handover, cup release and return to initial pose completed; "
                               "transfer and landing are not sensor-verified");
          return;
        }
        context_.set_phase(demo_stage_, demo_completed_stage_);
        demo_detail_ = "Starting " + demo_stage_;
      }
    }
  }
  if (demo_goal_ && elapsed(last_publish_) >= .2) {
    auto feedback = std::make_shared<DemoAction::Feedback>();
    feedback->stage = demo_stage_;
    feedback->phase = demo_retry_wait_ ? "RECOVERY_WAIT" : demo_stage_ == "PICK_DYNAMIXEL" ?
        pick_phase_ : context_.snapshot().phase;
    feedback->detail = demo_detail_;
    feedback->retry_count = demo_retries_;
    feedback->holding_state = "UNVERIFIED";
    demo_goal_->publish_feedback(feedback);
    context_.set_task(demo_retry_wait_ ? "RECOVERING" : "RUNNING", "right_to_left", "", demo_detail_);
  }
}
} // namespace demo_1f
