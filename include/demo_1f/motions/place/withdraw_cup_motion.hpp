#pragma once
#include "demo_1f/motions/common/cup_motion.hpp"

namespace demo_1f {
// Independent withdrawal; "both" runs left then right. No gripper/perception commands.
class WithdrawCupMotion final : public CupMotion {
public:
  using CupMotion::CupMotion;
  bool is_ready_to_start(const std::string & arm, const std::string & cup_id,
      std::string & reason, bool execute = true) const override;
  bool start(const std::string & arm, const std::string & cup_id,
      std::string & reason, bool execute = true) override;
  void tick() override;

private:
  const char * initial_phase() const override { return "LIFT_FROM_CUP"; }
  void reset() override { both_hands_ = false; }
  bool both_hands_{false};
};
} // namespace demo_1f
