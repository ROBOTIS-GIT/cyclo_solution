#include "demo_1f/runtime/state/robot_context.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
std::optional<geometry_msgs::msg::PoseStamped> RobotContext::transform_pose(
    const geometry_msgs::msg::PoseStamped & input) const {
  if (input.header.frame_id.empty()) {
    return std::nullopt;
  }
  if (input.header.frame_id == config_.planning_frame) {
    return input;
  }
  try {
    const auto transform = buffer_.lookupTransform(
        config_.planning_frame, input.header.frame_id, rclcpp::Time(input.header.stamp));
    const auto stamp = rclcpp::Time(transform.header.stamp).nanoseconds();
    // A zero transform timestamp represents an entirely static transform chain.
    if (stamp != 0) {
      const double age = (node_.now().nanoseconds() - stamp) * 1e-9;
      if (age > config_.tf_max_age || age < -0.1) {
        RCLCPP_WARN_THROTTLE(node_.get_logger(),
            *node_.get_clock(),
            2000,
            "TF timestamp rejected: %s -> %s age=%.3f s limit=%.3f s",
            input.header.frame_id.c_str(),
            config_.planning_frame.c_str(),
            age,
            config_.tf_max_age);
        return std::nullopt;
      }
    }
    geometry_msgs::msg::PoseStamped output;
    tf2::doTransform(input, output, transform);
    return output;
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "TF lookup failed: %s -> %s: %s",
        input.header.frame_id.c_str(),
        config_.planning_frame.c_str(),
        error.what());
    return std::nullopt;
  }
}

std::optional<geometry_msgs::msg::PoseStamped> RobotContext::pose_of(
    const std::string & frame) const {
  geometry_msgs::msg::PoseStamped origin;
  origin.header.frame_id = frame;
  origin.pose.orientation.w = 1.0;
  return transform_pose(origin);
}
} // namespace demo_1f
