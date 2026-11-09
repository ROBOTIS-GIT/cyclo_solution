#pragma once
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

namespace demo_1f {
struct ApproachConfig {
  double cup_height{.165}, inner_radius{.045}, wall_thickness{.005};
  double max_age{30.0}, observation_timeout{2}, max_position_error{.15}; // Saved head BODY lifetime.
  double travel_scale{.2}, tcp_roll_limit_deg{30}, tcp_pitch_limit_deg{40};
  double tcp_pitch_min_deg{0};
  bool tcp_pitch_downward_only{true};
  bool tcp_yaw_inward_only{true};
  std::vector<double> offset_xy{0, 0};
  double offset_z{0}; // TCP target offset in planning frame; does not change cup geometry.
  double refine_pitch_step_deg{5.0}, refine_max_correction{.12};
  double refine_view_wait{1.0}; // Fresh images without stability progress before changing view.
  double refine_speed{.06}, refine_clearance{.05}, refine_wait{3.0};
  double refine_search_timeout{15.0}; // Includes observation and pitch planning/execution.
  double refine_correction_speed{.20}, refine_correction_acceleration{.12};
  double refine_lower_distance{.03};
  double refine_gripper_open{0.0};
  double refine_pitch_ceiling_deg{75.0};
  std::string validate(const std::string & arm) const;
};

ApproachConfig load_approach_config(rclcpp::Node & node);
} // namespace demo_1f
