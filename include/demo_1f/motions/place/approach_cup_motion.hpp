#pragma once
#include "demo_1f/motions/common/cup_motion.hpp"
#include "demo_1f/motions/common/head_cup_observer.hpp"

namespace demo_1f {
// Head-camera BODY approach. Never opens the gripper.
class ApproachCupMotion final : public CupMotion {
public:
  using CupMotion::CupMotion;
  void tick() override;

private:
  const char * initial_phase() const override {
    return "FIND_CUP";
  }

  void reset() override {}

  bool begin_approach();
  bool dispatch(const geometry_msgs::msg::Pose & pose, double speed);
  HeadCupObserver observer_{context_, config_};
};
} // namespace demo_1f
