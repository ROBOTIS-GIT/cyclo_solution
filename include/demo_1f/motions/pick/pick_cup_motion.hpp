#pragma once
#include "demo_1f/motions/common/cup_motion.hpp"
#include "demo_1f/motions/common/head_cup_observer.hpp"
#include "demo_1f/motions/pick/pick_geometry.hpp"

namespace demo_1f {
class PickCupMotion final : public CupMotion {
public:
  PickCupMotion(const Config & robot,
      const ApproachConfig & config,
      const PickCupConfig & pick,
      RobotContext & context,
      ExecutorPort & executor,
      ApproachScenePort & scene);

  void tick() override;

private:
  const char * initial_phase() const override {
    return "PICK_APPROACH";
  }

  void reset() override;
  bool move(const std::string & phase, const geometry_msgs::msg::Pose & pose);
  bool open_gripper();
  bool apply_scene();
  HeadCupObserver observer_;
  const PickCupConfig & pick_;
  geometry_msgs::msg::Pose initial_reference_;
  void finish_approach();
  geometry_msgs::msg::Pose pregrasp_, body_;
};
} // namespace demo_1f
