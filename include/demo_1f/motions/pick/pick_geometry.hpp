#pragma once
#include <geometry_msgs/msg/pose.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>

namespace demo_1f {
struct PickCupConfig {
  double grasp_height{.065}, stand_off{.12}, speed{.12};
  double tcp_forward_offset{-.03};
  std::string validate(double cup_height) const;
};

PickCupConfig load_pick_config(rclcpp::Node & node);
geometry_msgs::msg::Pose cup_grasp_pose(const geometry_msgs::msg::Point & rim,
    double cup_height,
    const PickCupConfig & config,
    double yaw);

std::optional<double> cup_approach_yaw(
    const geometry_msgs::msg::Pose & initial_tcp, const geometry_msgs::msg::Point & cup);

} // namespace demo_1f
