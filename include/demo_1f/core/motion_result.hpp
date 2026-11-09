#pragma once
#include <string>

namespace demo_1f {
struct MotionResult {
  std::string state{"IDLE"};
  std::string code;
  std::string detail;
  std::string last_completed_step;
  bool sequence_completed{false};
  bool handover_confirmed{false};
};

} // namespace demo_1f
