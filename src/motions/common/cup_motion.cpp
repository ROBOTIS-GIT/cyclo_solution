#include "demo_1f/motions/common/cup_motion.hpp"
#include <algorithm>
#include <cmath>

namespace demo_1f {
CupMotion::CupMotion(const Config & robot,
    const ApproachConfig & config,
    RobotContext & context,
    ExecutorPort & executor,
    ApproachScenePort & scene)
    : robot_(robot), config_(config), context_(context), executor_(executor), scene_(scene) {}

bool CupMotion::running() const {
  return result_.state == "RUNNING" || result_.state == "CANCELLING";
}

bool CupMotion::is_ready_to_start(
    const std::string & arm, const std::string & id, std::string & reason, bool execute) const {
  reason = config_.validate(arm);
  if (!supports_arm(arm)) {
    reason = "This cup motion does not support the requested arm";
  }
  if (!reason.empty()) {
    return false;
  }
  if (blocks_other_motion() || (execute && !scene_.ready())) {
    reason = "Approach/scene busy, unavailable, or stop unconfirmed";
    return false;
  }
  if (id.empty() || !robot_.arms.count(arm)) {
    reason = "Missing cup ID or unknown arm";
    return false;
  }
  if (!execute) {
    return true;
  }
  if (!context_.ready(reason)) {
    return false;
  }
  Step move;
  move.type = "pose";
  move.arm = arm;
  MotionSequence sequence;
  sequence.steps = {move};
  return executor_.ready(sequence, reason);
}

bool CupMotion::start(
    const std::string & arm, const std::string & id, std::string & reason, bool execute) {
  if (!is_ready_to_start(arm, id, reason, execute)) {
    return false;
  }
  reset();
  execute_ = execute;
  arm_ = arm;
  cup_id_ = id;
  moving_ = scene_pending_ = at_observation_pose_ = false;
  approach_target_ = geometry_msgs::msg::PoseStamped();
  estimated_cup_pose_ = geometry_msgs::msg::PoseStamped();
  result_ = {};
  result_.state = "RUNNING";
  phase(initial_phase());
  return true;
}

bool CupMotion::execute_step(const Step & step) {
  std::string reason;
  if (!context_.ready(reason) || !executor_.ready(MotionSequence{"", "", "", {step}, ""}, reason)) {
    fail("ROBOT_NOT_READY", reason);
    return false;
  }
  if (!executor_.start(step)) {
    const auto outcome = executor_.result();
    fail(outcome.code.empty() ? "COMMAND_REJECTED" : outcome.code, outcome.detail);
    return false;
  }
  moving_ = true;
  return true;
}

void CupMotion::phase(const std::string & name) {
  phase_ = name;
  phase_started_ = Clock::now();
  if (result_.state == "RUNNING") {
    result_.detail.clear();
  }
  context_.set_phase(name, result_.last_completed_step);
}

void CupMotion::fail(const std::string & code, const std::string & detail) {
  result_.state = (code == "STOP_UNCONFIRMED" || code == "SCENE_UNCONFIRMED") ? "FAULT" : "FAILED";
  result_.code = code;
  result_.detail = detail;
}

void CupMotion::cancel() {
  if (!running()) {
    return;
  }
  result_.state = "CANCELLING";
  if (moving_) {
    executor_.cancel();
  } else if (!scene_pending_) {
    result_.state = "CANCELLED";
    result_.code = "CANCELLED";
  }
}

} // namespace demo_1f
