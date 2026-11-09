#include "demo_1f/runtime/execution/trajectory_executor.hpp"
#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <algorithm>
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
double TrajectoryExecutor::goal_tolerance(const std::string & joint, const Step & step) const {
  for (const auto & [side, arm] : config_.arms) {
    (void)side;
    if (joint == arm.gripper_joint) {
      return std::max(step.tolerance, config_.gripper_goal_tolerance);
    }
  }
  return step.tolerance;
}

bool TrajectoryExecutor::settled(const std::shared_ptr<Operation> & op, bool check_target) {
  const auto & arm = config_.arms.at(
      op->step.type == "dual_joint" || op->step.type == "joint_group" ? "left" : op->step.arm);
  const auto state = context_.snapshot();
  auto joints =
      op->step.type == "gripper" ? std::vector<std::string>{arm.gripper_joint} : arm.joints;
  if (!op->joint_targets.empty()) {
    joints.clear();
    for (const auto & [name, target] : op->joint_targets) {
      (void)target;
      joints.push_back(name);
    }
  }
  const auto after = op->cancel_requested ? op->cancelling : op->started;
  bool valid = true;
  std::string reason;
  for (size_t i = 0; i < joints.size(); ++i) {
    const auto it = state.joints.find(joints[i]);
    const bool commanded_gripper =
        (op->step.type == "gripper" || op->step.type == "joint_gripper" ||
            op->step.type == "dual_joint_gripper") && joints[i] == arm.gripper_joint;
    if (check_target && !op->cancel_requested && !commanded_gripper &&
        !config_.require_passive_gripper_stationary && context_.is_gripper_joint(joints[i])) {
      // A held current-limited finger is not an arm goal or a grasp sensor.
      // Cancellation still uses the normal measured-stop checks below.
      if (it != state.joints.end() && context_.joint_fresh(it->second)) continue;
      reason = "Missing fresh passive gripper feedback: " + joints[i];
      valid = false;
      break;
    }
    if (it == state.joints.end() ||
        !context_.joint_stationary(
            joints[i], it->second, reason, after, check_target && op->step.type == "gripper")) {
      if (it == state.joints.end()) {
        reason = "Missing joint feedback: " + joints[i];
      }
      valid = false;
      break;
    }
    if (check_target && op->step.type != "linear" && op->step.type != "pose") {
      const double target =
          op->joint_targets.empty() ? op->step.target[i] : op->joint_targets.at(joints[i]);
      const double error = std::abs(it->second.position - target);
      const double tolerance = goal_tolerance(joints[i], op->step);
      if (error > tolerance) {
        reason = "Joint target not reached: " + joints[i] +
                 " (measured=" + std::to_string(it->second.position) +
                 ", target=" + std::to_string(target) + ", error=" + std::to_string(error) +
                 ", tolerance=" + std::to_string(tolerance) + ")";
        valid = false;
        break;
      }
    }
  }
  if (valid && check_target && (op->step.type == "linear" || op->step.type == "pose")) {
    const auto current = context_.pose_of(arm.tcp);
    if (!current) {
      reason = "Waiting for fresh TCP transform: " + arm.tcp;
      valid = false;
    } else {
      const auto & a = current->pose;
      const auto & b = op->pose_target.pose;
      const double distance =
          std::hypot(std::hypot(a.position.x - b.position.x, a.position.y - b.position.y),
              a.position.z - b.position.z);
      const double dot = a.orientation.x * b.orientation.x + a.orientation.y * b.orientation.y +
                         a.orientation.z * b.orientation.z + a.orientation.w * b.orientation.w;
      const double angle = 2.0 * std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
      bool orientation_ok = angle <= config_.pose_orientation_tolerance;
      if (op->step.orientation_slack > 0) {
        tf2::Quaternion actual, desired;
        tf2::fromMsg(a.orientation, actual);
        tf2::fromMsg(b.orientation, desired);
        auto error = (actual * desired.inverse()).normalized();
        if (error.w() < 0) {
          error = tf2::Quaternion(-error.x(), -error.y(), -error.z(), -error.w());
        }
        const auto vector = error.getAxis() * error.getAngle();
        orientation_ok = std::abs(vector.x()) <= op->step.orientation_slack &&
                         std::abs(vector.y()) <= op->step.orientation_slack &&
                         std::abs(vector.z()) <= op->step.orientation_slack;
      }
      valid =
          distance <= config_.pose_position_tolerance && (op->step.position_only || orientation_ok);
      if (valid && op->step.tcp_roll_limit > 0) {
        const auto roll = tcp_roll(a.orientation);
        if (!roll || std::abs(*roll) > op->step.tcp_roll_limit) {
          valid = false;
          reason = "TCP roll outside configured limit or forward axis is vertical";
        }
      }
      if (valid && op->step.tcp_pitch_limit > 0) {
        const auto pitch = tcp_pitch(a.orientation);
        if (!pitch || std::abs(*pitch) > op->step.tcp_pitch_limit ||
            (op->step.tcp_pitch_downward_only && *pitch < op->step.tcp_pitch_min)) {
          valid = false;
          reason = "TCP pitch outside configured limit";
        }
      }
      if (valid && op->step.tcp_yaw_limit > 0) {
        const auto actual_yaw = tcp_yaw(a.orientation), target_yaw = tcp_yaw(b.orientation);
        if (!actual_yaw || !target_yaw ||
            std::abs(std::remainder(*actual_yaw - *target_yaw, 2 * M_PI)) >
                op->step.tcp_yaw_limit) {
          valid = false;
          reason = "TCP yaw outside target heading tolerance";
        }
      }
      if (valid && op->step.tcp_yaw_inward_only) {
        const auto yaw = tcp_yaw(a.orientation);
        if (!yaw || !tcp_yaw_is_inward(*yaw, op->step.arm)) {
          valid = false;
          reason = "TCP yaw outside forward/inward goal range";
        }
      }
      if (!valid) {
        if (reason.empty()) {
          reason = "TCP target not reached: position error=" + std::to_string(distance) + " m";
        }
        if (!op->step.position_only) {
          reason += ", orientation error=" + std::to_string(angle) + " rad";
        }
      }
    }
  }
  if (!valid) {
    op->completion_wait_reason = reason;
    if (check_target) {
      RCLCPP_WARN_THROTTLE(node_.get_logger(),
          *node_.get_clock(),
          2000,
          "Waiting for %s completion: %s",
          op->step.name.c_str(),
          reason.c_str());
    }
    op->stable_since.reset();
    return false;
  }
  if (!op->stable_since) {
    op->stable_since = Clock::now();
  }
  const double settle_time =
      check_target && op->step.type == "gripper" && config_.gripper_settle_time > 0
          ? config_.gripper_settle_time
          : config_.settle_time;
  const bool complete = elapsed(*op->stable_since) >= settle_time;
  op->completion_wait_reason = complete ? "" : "Waiting for settle_time";
  return complete;
}
} // namespace demo_1f
