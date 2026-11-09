#pragma once
#include "demo_1f/core/executor_port.hpp"
#include "demo_1f/core/motion_result.hpp"
#include "demo_1f/runtime/state/robot_context.hpp"

namespace demo_1f {
// Shared non-blocking sequence lifecycle for initialization and handover.
class SequenceMotion {
public:
  SequenceMotion(const Config & config, RobotContext & context, ExecutorPort & executor)
      : config_(config), context_(context), executor_(executor) {}
  bool is_ready_to_start(const MotionSequence & sequence, std::string & reason) const;
  bool start(const MotionSequence & sequence, std::string & reason, const std::string & detail);
  void tick();
  void cancel();
  bool running() const;
  const MotionResult & result() const { return result_; }

protected:
  const Config & config_;

private:
  void fail(const std::string & code, const std::string & detail);
  RobotContext & context_;
  ExecutorPort & executor_;
  MotionSequence selected_sequence_;
  const MotionSequence * active_sequence_{nullptr};
  size_t step_index_{0};
  bool step_active_{false};
  size_t active_step_count_{1};
  MotionResult result_;
};
} // namespace demo_1f
