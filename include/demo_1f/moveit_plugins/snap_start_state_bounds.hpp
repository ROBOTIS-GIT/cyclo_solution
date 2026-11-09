#pragma once

#include <moveit/planning_interface/planning_request_adapter.hpp>

namespace demo_1f {
// Correct floating-point roundoff in a planning start state, never robot limits.
class SnapStartStateBounds : public planning_interface::PlanningRequestAdapter {
public:
  std::string getDescription() const override {
    return "SnapStartStateBounds";
  }

  moveit::core::MoveItErrorCode adapt(const planning_scene::PlanningSceneConstPtr & scene,
      planning_interface::MotionPlanRequest & request) const override;
};
} // namespace demo_1f
