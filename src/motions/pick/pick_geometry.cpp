#include "demo_1f/motions/pick/pick_geometry.hpp"
#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
std::string PickCupConfig::validate(double height) const {
  for (double x : {grasp_height, stand_off, speed}) {
    if (!std::isfinite(x) || x <= 0) {
      return "Pick approach settings must be positive and finite";
    }
  }
  if (grasp_height < .05 || grasp_height > height - .04 || stand_off < .10 || stand_off > .20 ||
      speed > .2 || !std::isfinite(tcp_forward_offset) || std::abs(tcp_forward_offset) > .03) {
    return "Invalid cup pregrasp geometry or speed";
  }
  return {};
}

PickCupConfig load_pick_config(rclcpp::Node & node) {
  PickCupConfig c;
  rcl_interfaces::msg::ParameterDescriptor d;
  d.read_only = true;
  auto number = [&](const char * name, double & value) {
    value = node.declare_parameter(std::string("pick.") + name, value, d);
  };
  number("grasp_height", c.grasp_height);
  number("stand_off", c.stand_off);
  number("speed", c.speed);
  number("tcp_forward_offset", c.tcp_forward_offset);
  return c;
}

geometry_msgs::msg::Pose cup_grasp_pose(const geometry_msgs::msg::Point & rim,
    double height,
    const PickCupConfig & config,
    double yaw) {
  geometry_msgs::msg::Pose p;
  p.position = rim;
  p.position.x -= config.tcp_forward_offset * std::cos(yaw);
  p.position.y -= config.tcp_forward_offset * std::sin(yaw);
  p.position.z += config.grasp_height - height;
  // SG2: forward is local -Z, jaw separation is local Y; local X points up.
  tf2::Quaternion q;
  q.setRPY(0, -M_PI / 2, yaw);
  p.orientation = tf2::toMsg(q);
  return p;
}

std::optional<double> cup_approach_yaw(
    const geometry_msgs::msg::Pose & initial_tcp, const geometry_msgs::msg::Point & cup) {
  const double dx = cup.x - initial_tcp.position.x;
  const double dy = cup.y - initial_tcp.position.y;
  if (!std::isfinite(dx) || !std::isfinite(dy)) {
    return {};
  }
  // Near the cup axis, small perception errors can flip the heading.
  if (std::hypot(dx, dy) < .05) {
    return tcp_yaw(initial_tcp.orientation);
  }
  return std::atan2(dy, dx);
}

} // namespace demo_1f
