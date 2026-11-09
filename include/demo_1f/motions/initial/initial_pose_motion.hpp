#pragma once
#include "demo_1f/motions/common/sequence_motion.hpp"

namespace demo_1f {
class InitialPoseMotion : public SequenceMotion {
public:
  using SequenceMotion::SequenceMotion;
  bool is_ready_to_start(std::string & reason) const {
    const auto it = config_.motion_sequences.find("initial_pose");
    if (it == config_.motion_sequences.end()) {
      reason = "Initial pose sequence is not configured";
      return false;
    }
    return SequenceMotion::is_ready_to_start(it->second, reason);
  }
  bool start(std::string & reason) {
    if (!is_ready_to_start(reason)) return false;
    return SequenceMotion::start(config_.motion_sequences.at("initial_pose"), reason,
        "Moving arms, head and lift to the DYNAMIXEL observation pose");
  }
};
} // namespace demo_1f
