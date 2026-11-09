#pragma once
#include <string>

namespace demo_1f {
// Retry only before irreversible stages. Execution failures need operator inspection.
inline bool demo_can_retry(const std::string & stage, const std::string & code) {
  if (stage == "INITIALIZE_ROBOT") return code == "SCENE_UNAVAILABLE";
  if (stage == "APPROACH_CUP") {
    return code == "CUP_NOT_FOUND" || code == "CUP_CHANGED" || code == "SCENE_UNAVAILABLE";
  }
  if (stage == "REFINE_CUP") {
    return code == "WRIST_NOT_READY" || code == "WRIST_INPUT_MISSING" ||
           code == "RIM_NOT_CONFIRMED";
  }
  return false;
}
} // namespace demo_1f
