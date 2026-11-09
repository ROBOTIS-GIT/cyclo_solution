#include "demo_1f/tasks/task_manager.hpp"

namespace demo_1f {
void TaskManager::setup_cup_servers() {
  approach_target_publisher_ = node_.create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/approach_target", rclcpp::QoS(1).transient_local());
  cup_estimate_publisher_ = node_.create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/cup_estimate", rclcpp::QoS(1).transient_local());
  for (const auto & module : cup_motions_) {
    cup_servers_.push_back(make_cup_server(module.action_name, module.motion));
  }
}

rclcpp_action::Server<TaskManager::ApproachAction>::SharedPtr TaskManager::make_cup_server(
    const std::string & name, CupMotion & module) {
  return rclcpp_action::create_server<ApproachAction>(
      &node_,
      name,
      [this, &module, name](
          const rclcpp_action::GoalUUID &, std::shared_ptr<const ApproachAction::Goal> request) {
        std::string reason;
        if (demo_busy() || shutdown_requested_ || goal_start_pending_ || active_goal_handle_ || cup_goal_ ||
            motion_.running() || cup_motion_blocked() ||
            !module.is_ready_to_start(request->arm, request->cup_id, reason, request->execute)) {
          RCLCPP_WARN(node_.get_logger(),
              "Rejected cup approach: %s",
              reason.empty() ? "Task manager busy/shutting down" : reason.c_str());
          return rclcpp_action::GoalResponse::REJECT;
        }
        goal_start_pending_ = true;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this, &module](const std::shared_ptr<ApproachGoalHandle> handle) {
        if (handle != cup_goal_ || !module.running()) {
          return rclcpp_action::CancelResponse::REJECT;
        }
        module.cancel();
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this, &module, name](const std::shared_ptr<ApproachGoalHandle> handle) {
        goal_start_pending_ = false;
        const auto request = handle->get_goal();
        std::string reason;
        if (shutdown_requested_ ||
            !module.start(request->arm, request->cup_id, reason, request->execute)) {
          auto result = std::make_shared<ApproachAction::Result>();
          result->error_code = "NOT_READY";
          result->message = reason.empty() ? "Shutting down" : reason;
          handle->abort(result);
          return;
        }
        demo_resumable_ = false;
        active_cup_motion_ = &module;
        cup_goal_ = handle;
        context_.set_task("RUNNING", request->arm);
        RCLCPP_INFO(node_.get_logger(),
            "Cup %s %s: %s",
            name.c_str(),
            request->execute ? "execute" : "preview",
            request->arm.c_str());
      });
}

void TaskManager::publish_cup_target() {
  if (!active_cup_motion_->approach_target().header.frame_id.empty()) {
    approach_target_publisher_->publish(active_cup_motion_->approach_target());
  }
  if (!active_cup_motion_->estimated_cup_pose().header.frame_id.empty()) {
    cup_estimate_publisher_->publish(active_cup_motion_->estimated_cup_pose());
  }
}

void TaskManager::tick_cup_action() {
  active_cup_motion_->tick();
  const auto & outcome = active_cup_motion_->result();
  context_.set_task(outcome.state, cup_goal_->get_goal()->arm, outcome.code, outcome.detail);
  if (active_cup_motion_->running()) {
    if (elapsed(last_publish_) >= .2) {
      auto feedback = std::make_shared<ApproachAction::Feedback>();
      feedback->phase = context_.snapshot().phase;
      feedback->detail = outcome.detail;
      feedback->target_pose = active_cup_motion_->approach_target();
      cup_goal_->publish_feedback(feedback);
      publish_cup_target();
    }
    return;
  }
  auto result = std::make_shared<ApproachAction::Result>();
  result->success = outcome.sequence_completed;
  result->reached_observation_pose = active_cup_motion_->at_observation_pose();
  result->target_pose = active_cup_motion_->approach_target();
  result->estimated_cup_pose = active_cup_motion_->estimated_cup_pose();
  result->error_code = outcome.code;
  result->message = outcome.detail;
  if (cup_goal_->is_canceling()) {
    cup_goal_->canceled(result);
  } else if (outcome.sequence_completed) {
    cup_goal_->succeed(result);
  } else {
    cup_goal_->abort(result);
  }
  publish_cup_target();
  RCLCPP_INFO(
      node_.get_logger(), "ApproachCup %s: %s", outcome.state.c_str(), outcome.detail.c_str());
  cup_goal_.reset();
  active_cup_motion_ = nullptr;
  publish_status();
}
} // namespace demo_1f
