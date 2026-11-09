#include "demo_1f/tasks/task_manager.hpp"
#include <algorithm>
#include <utility>

namespace demo_1f {
TaskManager::TaskManager(rclcpp::Node & node,
    RobotContext & context,
    HandoverMotion & motion,
    std::vector<CupMotionRegistration> cup_motions, InitialPoseMotion * initial_motion)
    : node_(node), context_(context), motion_(motion), initial_motion_(initial_motion),
      cup_motions_(std::move(cup_motions)) {
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  const auto action_name =
      node.declare_parameter<std::string>("handover_action", "/handover", descriptor);
  status_publisher_ =
      node.create_publisher<demo_1f::msg::DemoStatus>("~/status", rclcpp::QoS(1).transient_local());
  capture_service_ = node.create_service<std_srvs::srv::Trigger>("~/capture_targets",
      [this](std_srvs::srv::Trigger::Request::ConstSharedPtr,
          std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::string reason;
        const auto targets = context_.capture_targets(reason);
        response->success = targets.has_value();
        response->message = targets ? *targets : reason;
      });

  server_ = make_handover_server(action_name, false);
  prepare_server_ = make_handover_server("/prepare_handover", true);
  if (!cup_motions_.empty()) {
    setup_cup_servers();
  }
  setup_demo_server();
  timer_ = node.create_wall_timer(std::chrono::milliseconds(20), [this]() {
    tick();
  });
  publish_status();
}

rclcpp_action::Server<TaskManager::Action>::SharedPtr TaskManager::make_handover_server(
    const std::string & action_name, bool prepare_only) {
  return rclcpp_action::create_server<Action>(
      &node_,
      action_name,
      [this, prepare_only](
          const rclcpp_action::GoalUUID &, std::shared_ptr<const Action::Goal> request) {
        std::string reason;
        if (demo_busy() || shutdown_requested_ || goal_start_pending_ || active_goal_handle_ != nullptr ||
            cup_goal_ != nullptr || cup_motion_blocked() ||
            !motion_.is_ready_to_start(request->direction, reason, prepare_only)) {
          if (reason.empty()) {
            reason = "Task manager is busy or shutting down";
          }
          RCLCPP_WARN(node_.get_logger(), "Rejected handover: %s", reason.c_str());
          if (!active_goal_handle_ && !cup_goal_ && !goal_start_pending_ && !motion_.running()) {
            context_.set_task("REJECTED", request->direction, "NOT_READY", reason);
          }
          return rclcpp_action::GoalResponse::REJECT;
        }
        goal_start_pending_ = true;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<GoalHandle> handle) {
        if (handle != active_goal_handle_ || !motion_.running()) {
          return rclcpp_action::CancelResponse::REJECT;
        }
        motion_.cancel();
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this, prepare_only](const std::shared_ptr<GoalHandle> handle) {
        goal_start_pending_ = false;
        std::string reason;
        if (shutdown_requested_ ||
            !motion_.start(handle->get_goal()->direction, reason, prepare_only)) {
          auto result = std::make_shared<Action::Result>();
          result->error_code = "NOT_READY";
          result->message = reason.empty() ? "Shutting down" : reason;
          handle->abort(result);
          return;
        }
        demo_resumable_ = false; // Manual motions invalidate a saved automatic checkpoint.
        active_goal_handle_ = handle;
        context_.set_task("RUNNING", handle->get_goal()->direction);
        RCLCPP_INFO(node_.get_logger(), "Starting %s", handle->get_goal()->direction.c_str());
      });
}

void TaskManager::request_shutdown() {
  if (shutdown_requested_) {
    return;
  }
  shutdown_requested_ = true;
  demo_cancel_requested_ = true;
  cancel_demo_motion();
  motion_.cancel();
  for (const auto & module : cup_motions_) {
    module.motion.cancel();
  }
}

bool TaskManager::cup_motion_blocked() const {
  return std::any_of(cup_motions_.begin(), cup_motions_.end(), [](const auto & module) {
    return module.motion.blocks_other_motion();
  });
}

bool TaskManager::can_shutdown() const {
  if (demo_faulted_ && !demo_goal_) return true;
  return !demo_goal_ && !pick_pending_ && !pick_handle_ && !goal_start_pending_ && !active_goal_handle_ && !motion_.running() &&
         (!initial_motion_ || !initial_motion_->running()) && !cup_goal_ &&
         std::none_of(cup_motions_.begin(), cup_motions_.end(), [](const auto & module) {
           return module.motion.running();
         });
}

void TaskManager::tick() {
  if (demo_goal_) {
    tick_demo();
  } else if (cup_goal_) {
    tick_cup_action();

  } else {
    motion_.tick();
  }
  if (active_goal_handle_) {
    const auto & outcome = motion_.result();
    context_.set_task(
        outcome.state, active_goal_handle_->get_goal()->direction, outcome.code, outcome.detail);
    if (!motion_.running()) {
      auto result = std::make_shared<Action::Result>();
      result->sequence_completed = outcome.sequence_completed;
      result->handover_confirmed = outcome.handover_confirmed;
      result->error_code = outcome.code;
      result->message = outcome.detail;
      result->last_completed_step = outcome.last_completed_step;
      if (active_goal_handle_->is_canceling()) {
        active_goal_handle_->canceled(result);
      } else if (outcome.sequence_completed) {
        active_goal_handle_->succeed(result);
      } else {
        active_goal_handle_->abort(result);
      }
      RCLCPP_INFO(
          node_.get_logger(), "Handover %s: %s", outcome.state.c_str(), outcome.detail.c_str());
      active_goal_handle_.reset();
      publish_status();
    }
  }
  if (elapsed(last_publish_) >= 0.2) {
    publish_status();
    if (active_goal_handle_) {
      const auto state = context_.snapshot();
      auto feedback = std::make_shared<Action::Feedback>();
      feedback->direction = state.direction;
      feedback->phase = state.phase;
      feedback->execution_state = to_string(state.execution.state);
      feedback->holding_state = "UNVERIFIED";
      active_goal_handle_->publish_feedback(feedback);
    }
    last_publish_ = Clock::now();
  }
}

void TaskManager::publish_status() {
  const auto state = context_.snapshot();
  demo_1f::msg::DemoStatus status;
  status.task_state = state.task_state;
  status.direction = state.direction;
  status.phase = state.phase;
  status.last_completed_step = state.last_completed_step;
  status.execution_state = to_string(state.execution.state);
  status.holding_state = "UNVERIFIED";
  status.error_code = state.error_code;
  status.detail = state.detail;
  std::string reason;
  status.robot_state_fresh = context_.ready(reason);
  if (!status.robot_state_fresh && status.detail.empty()) {
    status.detail = reason;
  }
  status_publisher_->publish(status);
}
} // namespace demo_1f
