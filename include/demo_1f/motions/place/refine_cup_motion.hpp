#pragma once
#include "demo_1f/motions/common/cup_motion.hpp"

namespace demo_1f {
// Wrist RIM observation -> lowered center -> release. Preview never moves.
class RefineCupMotion final : public CupMotion {
public:
  using CupMotion::CupMotion;
  void tick() override;

private:
  const char * initial_phase() const override {
    return "FIND_WRIST_RIM";
  }

  bool supports_arm(const std::string & arm) const override {
    return arm == "left";
  }

  void reset() override;
  void tilt_to_observe();
  void find_wrist_target();
  std::optional<TimePoint> view_started_;
  std::optional<TimePoint> gripper_settle_started_;
  bool dispatch_refinement(const geometry_msgs::msg::Pose & pose);
  int absent_frames_{0}, planning_retries_{0}, stable_progress_{0};
  double tilt_progress_deg_{0};
  TimePoint search_started_{};
  std::map<std::string, double> correction_start_joints_;
  int64_t last_wrist_stamp_{0};
  std::optional<geometry_msgs::msg::PoseStamped> refinement_start_, tilt_origin_;
};
} // namespace demo_1f
