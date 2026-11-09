#pragma once

#include "demo_1f/core/types.hpp"
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

namespace demo_1f {
// Call before rclcpp::init: use MoveIt's supported node-less logger fallback.
// The deployed Jazzy logger otherwise keeps a ROS node alive until process exit.
void initialize_moveit_logging();

struct SmoothPath {
  moveit_msgs::msg::RobotTrajectory trajectory;
  std::vector<size_t> point_steps;
  std::vector<double> segment_end_times;
};

MotionLimits effective_limits(const Config & config,
    const std::string & name,
    const Step & step,
    const moveit::core::RobotModelConstPtr & model = {});

// Ruckig runs locally, one target state at a time. No online waypoint service is used.
SmoothPath smooth_segments(const std::vector<moveit_msgs::msg::RobotTrajectory> & segments,
    const std::vector<Step> & steps,
    const Config & config,
    const moveit::core::RobotModelConstPtr & model = {});

// Validate the controller's quintic interpolation, not just the supplied knots.
void validate_smooth_path(const SmoothPath & path,
    const std::vector<Step> & steps,
    const Config & config,
    planning_scene::PlanningScene & scene);
} // namespace demo_1f
