#pragma once
#include "demo_1f/motions/common/sequence_motion.hpp"

namespace demo_1f {
class HandoverMotion : public SequenceMotion {
public:
  HandoverMotion(const Config & config, RobotContext & context, ExecutorPort & executor)
      : SequenceMotion(config, context, executor) {}
  bool is_ready_to_start(const std::string & direction, std::string & reason,
      bool prepare_only = false) const;
  bool start(const std::string & direction, std::string & reason, bool prepare_only = false);
};
} // namespace demo_1f
