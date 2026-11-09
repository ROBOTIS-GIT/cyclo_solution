#include "demo_1f/motions/place/withdraw_cup_motion.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
bool WithdrawCupMotion::is_ready_to_start(const std::string & arm, const std::string & cup_id,
    std::string & reason, bool execute) const {
  if (arm != "both") return CupMotion::is_ready_to_start(arm, cup_id, reason, execute);
  if (!execute) {
    reason = "Preview each hand separately using arm=left or arm=right";
    return false;
  }
  return CupMotion::is_ready_to_start("left", cup_id, reason, execute) &&
         CupMotion::is_ready_to_start("right", cup_id, reason, execute);
}

bool WithdrawCupMotion::start(const std::string & arm, const std::string & cup_id,
    std::string & reason, bool execute) {
  if (!is_ready_to_start(arm, cup_id, reason, execute)) return false;
  if (!CupMotion::start(arm == "both" ? "left" : arm, cup_id, reason, execute)) return false;
  both_hands_ = arm == "both";
  return true;
}

void WithdrawCupMotion::tick() {
  if (!running()) return;
  if (moving_) {
    executor_.tick();
    std::string reason;
    if (!context_.ready(reason)) executor_.cancel("ROBOT_STATE_INVALID", reason);
    const auto outcome = executor_.result();
    if (outcome.state == ExecutionState::Running || outcome.state == ExecutionState::Cancelling)
      return;
    moving_ = false;
    if (outcome.state == ExecutionState::Failed || outcome.state == ExecutionState::StopUnconfirmed) {
      fail(outcome.code, arm_ + "/" + phase_ + ": " + outcome.detail);
      return;
    }
    if (outcome.state == ExecutionState::Cancelled || result_.state == "CANCELLING") {
      result_.state = "CANCELLED";
      result_.code = "CANCELLED";
      return;
    }
    if (outcome.state != ExecutionState::Succeeded) {
      fail("EXECUTOR_STATE", "Unexpected withdrawal executor state");
      return;
    }
    result_.last_completed_step = arm_ + "/" + phase_;
    if (phase_ == "RETREAT_FROM_CUP") {
      if (both_hands_ && arm_ == "left") {
        arm_ = "right";
        phase("LIFT_FROM_CUP");
        return; // Yield to cancellation before starting the other hand.
      }
      result_.state = "SUCCEEDED";
      result_.sequence_completed = true;
      result_.detail = both_hands_ ?
          "Both hands lifted 10cm and withdrawn 8cm; grippers unchanged" :
          "Hand lifted 10cm and withdrawn 8cm; gripper unchanged";
      phase("WITHDRAW_COMPLETE");
      return;
    }
    phase("RETREAT_FROM_CUP");
  }
  const auto tcp = context_.pose_of(robot_.arms.at(arm_).tcp);
  if (!tcp) {
    fail("TCP_TF_MISSING", "Cannot compute cup withdrawal from measured TCP");
    return;
  }
  auto target = tcp->pose;
  if (phase_ == "LIFT_FROM_CUP") target.position.z += .10;
  tf2::Quaternion rotation;
  tf2::fromMsg(target.orientation, rotation);
  // SG2 forward is -TCP Z; retreat along +TCP Z after clearing the rim.
  const auto backward = tf2::quatRotate(rotation, tf2::Vector3(0, 0, .08));
  if (backward.z() < -1e-6) {
    fail("RETREAT_DIRECTION", "Cup retreat would lower the TCP; inspect hand orientation");
    return;
  }
  if (phase_ == "RETREAT_FROM_CUP" || !execute_) {
    target.position.x += backward.x();
    target.position.y += backward.y();
    target.position.z += backward.z();
  }
  approach_target_ = *tcp;
  approach_target_.pose = target;
  if (!execute_) {
    result_.state = "SUCCEEDED";
    result_.sequence_completed = true;
    result_.detail = "Withdrawal geometry preview only; path not verified";
    phase("WITHDRAW_PREVIEW_READY");
    return;
  }
  Step step;
  step.name = phase_;
  step.type = "linear";
  step.arm = arm_;
  step.frame = robot_.planning_frame;
  const auto & p = target.position;
  const auto & q = target.orientation;
  step.target = {p.x, p.y, p.z, q.x, q.y, q.z, q.w};
  step.timeout = 30;
  step.velocity_scale = config_.refine_correction_speed;
  step.acceleration_scale = step.velocity_scale * .25;
  result_.detail = arm_ + ": " + phase_;
  execute_step(step);
}
} // namespace demo_1f
