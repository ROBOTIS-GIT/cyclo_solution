#include "demo_1f/moveit_plugins/snap_start_state_bounds.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace demo_1f {
moveit::core::MoveItErrorCode SnapStartStateBounds::adapt(
    const planning_scene::PlanningSceneConstPtr & scene,
    planning_interface::MotionPlanRequest & request) const {
  constexpr double roundoff_tolerance = 1e-8; // radians, or metres for prismatic joints
  moveit::core::MoveItErrorCode result;
  result.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  result.source = getDescription();
  const auto * group = scene->getRobotModel()->getJointModelGroup(request.group_name);
  if (!group) {
    result.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    result.message = "Unknown planning group: " + request.group_name;
    return result;
  }
  auto state = scene->getCurrentState();
  if (!moveit::core::robotStateMsgToRobotState(
          scene->getTransforms(), request.start_state, state)) {
    result.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID;
    result.message = "Cannot read planning start state";
    return result;
  }

  std::map<std::string, double> corrections;
  for (const auto * joint : group->getJointModels()) {
    for (const auto & name : joint->getVariableNames()) {
      const double position = state.getVariablePosition(name);
      const auto & bounds = scene->getRobotModel()->getVariableBounds(name);
      if (!std::isfinite(position)) {
        result.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID;
        result.message = "Non-finite start position: " + name;
        return result;
      }
      if (!bounds.position_bounded_) {
        continue;
      }
      const double bounded = std::clamp(position, bounds.min_position_, bounds.max_position_);
      const double error = std::abs(position - bounded);
      if (error > roundoff_tolerance) {
        result.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID;
        result.message = "Start position exceeds roundoff tolerance: " + name;
        RCLCPP_ERROR(rclcpp::get_logger("demo_1f.start_state_bounds"),
            "%s: %.12g is outside [%.12g, %.12g] by %.12g (maximum %.12g)",
            name.c_str(),
            position,
            bounds.min_position_,
            bounds.max_position_,
            error,
            roundoff_tolerance);
        return result;
      }
      if (error > 0) {
        corrections[name] = bounded;
      }
    }
  }
  // Apply only after every checked joint passes. Preserve velocities, attached
  // bodies and all unrelated state; subsequent MoveIt adapters still validate it.
  if (!corrections.empty()) {
    state.setVariablePositions(corrections);
    moveit::core::robotStateToRobotStateMsg(state, request.start_state);
    result.message = "Snapped numerical start-state roundoff to model bounds";
    RCLCPP_INFO(rclcpp::get_logger("demo_1f.start_state_bounds"),
        "Snapped %zu start-state values to model bounds (maximum correction %.1e)",
        corrections.size(),
        roundoff_tolerance);
  }
  return result;
}
} // namespace demo_1f

PLUGINLIB_EXPORT_CLASS(demo_1f::SnapStartStateBounds, planning_interface::PlanningRequestAdapter)
