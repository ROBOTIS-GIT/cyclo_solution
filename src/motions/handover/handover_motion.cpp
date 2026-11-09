#include "demo_1f/motions/handover/handover_motion.hpp"
#include "demo_1f/runtime/config.hpp"

namespace demo_1f {
namespace {
MotionSequence select_sequence(const MotionSequence & full, bool prepare_only) {
  auto selected = full;
  if (prepare_only && !selected.steps.empty()) {
    selected.steps.resize(1);
    selected.steps.front().blend_with_next = false;
  }
  return selected;
}
} // namespace

bool HandoverMotion::is_ready_to_start(const std::string & direction,
    std::string & reason, bool prepare_only) const {
  const auto it = config_.motion_sequences.find(direction);
  if ((direction != "left_to_right" && direction != "right_to_left") || it == config_.motion_sequences.end()) {
    reason = "Direction must be left_to_right or right_to_left";
    return false;
  }
  const auto selected = select_sequence(it->second, prepare_only);
  if (prepare_only && (selected.steps.empty() || selected.steps.front().name != "face_each_other" ||
                          selected.steps.front().type != "dual_joint")) {
    reason = "Preparation requires a face_each_other dual_joint step; no gripper commands allowed";
    return false;
  }
  return SequenceMotion::is_ready_to_start(selected, reason);
}

bool HandoverMotion::start(const std::string & direction, std::string & reason, bool prepare_only) {
  if (!is_ready_to_start(direction, reason, prepare_only)) return false;
  return SequenceMotion::start(select_sequence(config_.motion_sequences.at(direction), prepare_only),
      reason, prepare_only ? "Preparation pose only; grippers unchanged"
                           : "Scripted sequence; physical object transfer is not verified");
}
} // namespace demo_1f
