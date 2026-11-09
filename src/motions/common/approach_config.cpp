#include "demo_1f/motions/common/cup_scene.hpp"
#include <algorithm>
#include <cmath>

namespace demo_1f {
std::string ApproachConfig::validate(const std::string & arm) const {
  if (arm != "left" && arm != "right") {
    return "Arm must be left or right";
  }
  if (!std::isfinite(tcp_roll_limit_deg) || tcp_roll_limit_deg < 0 ||
      (tcp_roll_limit_deg > 0 && (tcp_roll_limit_deg < 5 || tcp_roll_limit_deg > 90))) {
    return "TCP roll limit must be zero (disabled) or 5..90 degrees";
  }
  if (!std::isfinite(tcp_pitch_limit_deg) || tcp_pitch_limit_deg < 0 ||
      (tcp_pitch_limit_deg > 0 && (tcp_pitch_limit_deg < 5 || tcp_pitch_limit_deg > 89))) {
    return "TCP pitch limit must be zero (disabled) or 5..89 degrees";
  }
  if (tcp_pitch_downward_only && tcp_pitch_limit_deg == 0) {
    return "Downward-only pitch requires a nonzero pitch limit";
  }
  if (!std::isfinite(tcp_pitch_min_deg) || tcp_pitch_min_deg < 0 ||
      (tcp_pitch_min_deg > 0 &&
          (!tcp_pitch_downward_only || tcp_pitch_min_deg + 2 >= tcp_pitch_limit_deg))) {
    return "Minimum downward pitch needs downward-only mode and more than 2 degrees below maximum";
  }
  for (double value : {cup_height,
           inner_radius,
           wall_thickness,
           max_age,
           observation_timeout,
           max_position_error,
           travel_scale}) {
    if (!std::isfinite(value) || value <= 0) {
      return "Approach dimensions, timing and limits must be positive and finite";
    }
  }
  if (cup_height > .5 || inner_radius > .3 || max_position_error > .2 || travel_scale > 1 ||
      offset_xy.size() != 2 ||
      !std::all_of(offset_xy.begin(),
          offset_xy.end(),
          [](double x) {
            return std::isfinite(x);
          }) ||
      std::hypot(offset_xy[0], offset_xy[1]) > .3 || !std::isfinite(offset_z) ||
      std::abs(offset_z) > .3) {
    return "Invalid approach dimensions, offset or limits";
  }
  for (double value : {refine_pitch_step_deg,
           refine_max_correction,
           refine_speed,
           refine_clearance,
           refine_wait,
           refine_view_wait,
           refine_search_timeout,
           refine_pitch_ceiling_deg,
           refine_correction_speed,
           refine_correction_acceleration,
           refine_lower_distance}) {
    if (!std::isfinite(value) || value <= 0) {
      return "Refinement limits must be positive and finite";
    }
  }
  if (refine_view_wait >= refine_wait) {
    return "View retry delay must be shorter than wrist observation timeout";
  }
  if (refine_pitch_step_deg > 5 || refine_pitch_ceiling_deg > 80 ||
      refine_max_correction > .15 || refine_speed > .1 || refine_clearance < .03 ||
      refine_clearance > .15) {
    return "Invalid refinement tilt, correction, speed or clearance limits";
  }
  if (refine_correction_speed > .2 || refine_correction_acceleration > .15 ||
      refine_lower_distance > .03 || refine_clearance - refine_lower_distance < .015 ||
      !std::isfinite(refine_gripper_open) || refine_gripper_open < 0 || refine_gripper_open > .8) {
    return "Invalid refinement release or speed settings";
  }
  return {};
}

ApproachConfig load_approach_config(rclcpp::Node & node) {
  ApproachConfig c;
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  auto number = [&](const std::string & key, double & value) {
    value = node.declare_parameter("approach." + key, value, descriptor);
  };
  number("cup_height", c.cup_height);
  number("offset_z", c.offset_z);
  number("inner_radius", c.inner_radius);
  number("wall_thickness", c.wall_thickness);
  number("max_age", c.max_age);
  number("observation_timeout", c.observation_timeout);
  number("max_position_error", c.max_position_error);
  number("travel_scale", c.travel_scale);
  number("tcp_roll_limit_deg", c.tcp_roll_limit_deg);
  number("tcp_pitch_limit_deg", c.tcp_pitch_limit_deg);
  number("tcp_pitch_min_deg", c.tcp_pitch_min_deg);
  c.tcp_pitch_downward_only = node.declare_parameter(
      "approach.tcp_pitch_downward_only", c.tcp_pitch_downward_only, descriptor);
  c.tcp_yaw_inward_only =
      node.declare_parameter("approach.tcp_yaw_inward_only", c.tcp_yaw_inward_only, descriptor);
  number("refine_pitch_step_deg", c.refine_pitch_step_deg);
  number("refine_pitch_ceiling_deg", c.refine_pitch_ceiling_deg);
  number("refine_max_correction", c.refine_max_correction);
  number("refine_speed", c.refine_speed);
  number("refine_correction_speed", c.refine_correction_speed);
  number("refine_correction_acceleration", c.refine_correction_acceleration);
  number("refine_lower_distance", c.refine_lower_distance);
  number("refine_gripper_open", c.refine_gripper_open);

  number("refine_clearance", c.refine_clearance);
  number("refine_wait", c.refine_wait);
  number("refine_view_wait", c.refine_view_wait);
  number("refine_search_timeout", c.refine_search_timeout);
  c.offset_xy = node.declare_parameter("approach.offset_xy", c.offset_xy, descriptor);
  return c;
}

std::optional<geometry_msgs::msg::Pose> approach_tcp_target(const geometry_msgs::msg::Pose & body,
    const ApproachConfig & config,
    const std::string & arm,
    double uncertainty,
    std::string & reason) {
  reason = config.validate(arm);
  if (!reason.empty()) {
    return {};
  }
  for (double value : {body.position.x, body.position.y, body.position.z, uncertainty}) {
    if (!std::isfinite(value)) {
      reason = "Non-finite TCP target input";
      return {};
    }
  }
  if (uncertainty < 0 || uncertainty > config.max_position_error) {
    reason = "Excessive cup uncertainty";
    return {};
  }
  auto target = body;
  target.position.x += config.offset_xy[0];
  target.position.y += config.offset_xy[1];
  target.position.z += config.cup_height + config.offset_z;
  target.orientation = geometry_msgs::msg::Quaternion();
  target.orientation.w = 1; // Position goal placeholder; roll/pitch have separate goal constraints.
  reason.clear();
  return target;
}
} // namespace demo_1f
