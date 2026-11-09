#include "action_dispatch.hpp"
#include "demo_1f/runtime/execution/trajectory_executor.hpp"

namespace demo_1f {
void TrajectoryExecutor::send_gripper(const std::shared_ptr<Operation> & op) {
  const auto & arm = config_.arms.at(op->step.arm);
  trajectory_msgs::msg::JointTrajectory trajectory;
  trajectory.joint_names = {arm.gripper_joint};
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = op->step.target;
  point.velocities = {0.0};
  point.time_from_start = rclcpp::Duration::from_seconds(op->step.duration);
  trajectory.points.push_back(point);
  if (config_.smooth_motion) {
    trajectory = op->path.trajectory.joint_trajectory;
    if (trajectory.points.empty()) {
      throw std::runtime_error("Gripper profile has not been validated");
    }
    op->step.duration = rclcpp::Duration(trajectory.points.back().time_from_start).seconds();
    if (op->step.duration + config_.settle_time + config_.gripper_stop_window >=
        op->step.timeout - elapsed(op->started)) {
      throw std::runtime_error(
          "Gripper timeout is too short for the limited trajectory and stop verification");
    }
  }
  op->gripper_sent = Clock::now();
  if (config_.gripper_transport == "topic") {
    if (publishers_.at(op->step.arm)->get_subscription_count() == 0) {
      throw std::runtime_error("No gripper trajectory subscriber");
    }
    op->topic_command = true;
    publishers_.at(op->step.arm)->publish(trajectory);
  } else {
    Joint::Goal goal;
    goal.trajectory = trajectory;
    control_msgs::msg::JointTolerance tolerance;
    tolerance.name = arm.gripper_joint;
    tolerance.position = goal_tolerance(arm.gripper_joint, op->step);
    if (op->step.gripper_wait_only) {
      // JointTolerance -1 erases controller goal checks (0 inherits defaults).
      // Keep path checks, action acceptance/errors and cancellation intact.
      tolerance.position = tolerance.velocity = tolerance.acceleration = -1.0;
    }
    goal.goal_tolerance = {tolerance};
    goal.goal_time_tolerance = rclcpp::Duration::from_seconds(config_.settle_time);
    dispatch<Joint>(joint_clients_.at(op->step.arm), goal, op, [](const Joint::Result & result) {
      return result.error_code == Joint::Result::SUCCESSFUL
                 ? std::string{}
                 : "Controller error " + std::to_string(result.error_code) + ": " +
                       result.error_string;
    });
  }
}

void TrajectoryExecutor::publish_hold(const std::shared_ptr<Operation> & op) {
  const auto & arm = config_.arms.at(op->step.arm);
  const auto state = context_.snapshot();
  const auto it = state.joints.find(arm.gripper_joint);
  if (it == state.joints.end() || !context_.joint_fresh(it->second)) {
    return;
  }
  trajectory_msgs::msg::JointTrajectory hold;
  hold.joint_names = {arm.gripper_joint};
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = {it->second.position};
  point.velocities = {0.0};
  point.time_from_start = rclcpp::Duration::from_seconds(0.2);
  hold.points.push_back(point);
  publishers_.at(op->step.arm)->publish(hold);
  op->hold_sent = true;
}
} // namespace demo_1f
