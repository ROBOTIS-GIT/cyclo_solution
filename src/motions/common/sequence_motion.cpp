#include "demo_1f/motions/common/sequence_motion.hpp"
#include "demo_1f/runtime/config.hpp"

namespace demo_1f {
bool SequenceMotion::running() const {
  return result_.state == "RUNNING" || result_.state == "CANCELLING";
}

bool SequenceMotion::is_ready_to_start(const MotionSequence & sequence, std::string & reason) const {
  if (running()) { reason = "A motion sequence is already running"; return false; }
  reason = validate_motion_sequence(sequence, config_);
  if (!reason.empty() || !context_.ready(reason)) return false;
  return executor_.ready(sequence, reason);
}

bool SequenceMotion::start(const MotionSequence & sequence, std::string & reason,
    const std::string & detail) {
  if (!is_ready_to_start(sequence, reason)) return false;
  selected_sequence_ = sequence;
  active_sequence_ = &selected_sequence_;
  result_ = {};
  result_.state = "RUNNING";
  result_.detail = detail;
  step_index_ = 0;
  step_active_ = false;
  context_.set_phase("CHECK_READY", {});
  return true;
}

void SequenceMotion::fail(const std::string & code, const std::string & detail) {
  result_.state = code == "STOP_UNCONFIRMED" ? "FAULT" : "FAILED";
  result_.code = code;
  result_.detail = detail;
}

void SequenceMotion::cancel() {
  if (!running()) {
    return;
  }
  result_.state = "CANCELLING";
  if (step_active_) {
    executor_.cancel();
  } else {
    result_.state = "CANCELLED";
    result_.code = "CANCELLED";
    result_.detail = "Cancelled between steps";
  }
}

void SequenceMotion::tick() {
  executor_.tick();
  if (!running()) {
    return;
  }
  if (step_active_) {
    auto execution = executor_.result();
    if (!execution.phase.empty()) {
      context_.set_phase(execution.phase, result_.last_completed_step);
    }
    std::string reason;
    if (result_.state == "RUNNING" && !context_.ready(reason)) {
      if (execution.state == ExecutionState::Running) {
        executor_.cancel("ROBOT_STATE_INVALID", reason);
        execution = executor_.result();
      } else if (execution.state == ExecutionState::Succeeded) {
        fail("ROBOT_STATE_INVALID", reason);
        return;
      }
    }
    if (execution.state == ExecutionState::Running ||
        execution.state == ExecutionState::Cancelling) {
      return;
    }
    if (execution.state == ExecutionState::StopUnconfirmed ||
        execution.state == ExecutionState::Failed) {
      fail(execution.code, execution.detail);
      return;
    }
    if (execution.state == ExecutionState::Cancelled || result_.state == "CANCELLING") {
      result_.state = "CANCELLED";
      result_.code = "CANCELLED";
      result_.detail = "Execution stopped; no automatic release or retreat was issued";
      return;
    }
    if (execution.state != ExecutionState::Succeeded) {
      fail("EXECUTOR_STATE", "Unexpected executor state");
      return;
    }
    step_index_ += active_step_count_;
    result_.last_completed_step = active_sequence_->steps[step_index_ - 1].name;
    step_active_ = false;
    context_.set_phase("BETWEEN_STEPS", result_.last_completed_step);
    // Yield to ROS callbacks before a potentially irreversible next step (e.g. release).
    return;
  }
  if (step_index_ == active_sequence_->steps.size()) {
    result_.state = "SUCCEEDED";
    result_.sequence_completed = true;
    result_.handover_confirmed = false;
    context_.set_phase("COMPLETE", result_.last_completed_step);
    return;
  }
  std::string reason;
  if (!context_.ready(reason)) {
    fail("ROBOT_STATE_INVALID", reason);
    return;
  }
  const auto & step = active_sequence_->steps[step_index_];
  std::vector<Step> steps{step};
  while (steps.back().blend_with_next) {
    steps.push_back(active_sequence_->steps[step_index_ + steps.size()]);
  }
  active_step_count_ = steps.size();
  context_.set_phase(step.name, result_.last_completed_step);
  const bool started = steps.size() == 1 ? executor_.start(step) : executor_.start_sequence(steps);
  if (!started) {
    const auto execution = executor_.result();
    fail(execution.code.empty() ? "COMMAND_REJECTED" : execution.code, execution.detail);
    return;
  }
  step_active_ = true;
}
} // namespace demo_1f
