#pragma once
#include "demo_1f/runtime/execution/trajectory_executor.hpp"
#include <type_traits>

// Shared ROS action lifecycle; kept private to the execution implementation.
namespace demo_1f {
template <class Action>
void TrajectoryExecutor::dispatch(const typename rclcpp_action::Client<Action>::SharedPtr & client,
    const typename Action::Goal & goal,
    const std::shared_ptr<Operation> & op,
    std::function<std::string(const typename Action::Result &)> error,
    std::function<void(const typename Action::Result &)> on_success) {
  if (!client->action_server_is_ready()) {
    throw std::runtime_error("Action server unavailable");
  }
  op->action_terminal = false;
  op->action_success = false;
  op->cancel_action = {};
  typename rclcpp_action::Client<Action>::SendGoalOptions options;
  options.goal_response_callback =
      [op, client](typename rclcpp_action::ClientGoalHandle<Action>::SharedPtr handle) {
        if (!handle) {
          op->action_terminal = true;
          if (!op->cancel_requested) {
            op->result = {ExecutionState::Failed, "GOAL_REJECTED", "Remote goal rejected"};
          }
          return;
        }
        op->cancel_action = [client, handle]() {
          client->async_cancel_goal(handle);
        };
        // Cancellation may arrive before the server accepts the goal. Never orphan it.
        if (op->cancel_requested) {
          try {
            op->cancel_action();
          } catch (const std::exception & e) {
            op->cancel_reason += "; " + std::string(e.what());
          }
        }
      };
  options.result_callback =
      [op, error, on_success, planning_only = op->planning_only](
          const typename rclcpp_action::ClientGoalHandle<Action>::WrappedResult & response) {
        op->action_terminal = true;
        op->cancel_action = {};
        const auto detail = response.result ? error(*response.result) : "Missing action result";
        op->action_success =
            response.code == rclcpp_action::ResultCode::SUCCEEDED && detail.empty();
        if (!op->cancel_requested && op->action_success && on_success &&
            op->result.state == ExecutionState::Running) {
          try {
            on_success(*response.result);
          } catch (const std::exception & error) {
            op->result = {ExecutionState::Failed, "PATH_PREPARATION_FAILED", error.what()};
          }
          return;
        }
        if (!op->cancel_requested && !op->action_success) {
          op->result = {ExecutionState::Failed,
              "EXECUTION_FAILED",
              detail.empty() ? "Action did not succeed" : detail};
          if constexpr (std::is_same_v<Action, Move>) {
            // Generic OMPL failures are retryable only for an explicit plan-only request.
            // Combined planning/execution errors must never be mistaken for pre-motion failure.
            if (response.result &&
                (response.result->error_code.val == moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT ||
                    (planning_only &&
                        (response.result->error_code.val == moveit_msgs::msg::MoveItErrorCodes::FAILURE ||
                            response.result->error_code.val ==
                                moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED))) &&
                response.result->planned_trajectory.joint_trajectory.points.empty() &&
                response.result->planned_trajectory.multi_dof_joint_trajectory.points.empty() &&
                response.result->executed_trajectory.joint_trajectory.points.empty() &&
                response.result->executed_trajectory.multi_dof_joint_trajectory.points.empty()) {
              op->result.code =
                  response.result->error_code.val == moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT
                      ? "PLANNING_TIMEOUT" : "PLANNING_FAILED";
            }
          }
        }
      };
  client->async_send_goal(goal, options);
}
} // namespace demo_1f
