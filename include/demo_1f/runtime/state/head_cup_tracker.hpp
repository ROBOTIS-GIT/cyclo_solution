#pragma once
#include "demo_1f/core/types.hpp"
#include "demo_1f/msg/cup_observation.hpp"
#include <cstdint>
#include <deque>

namespace demo_1f {
struct CupState {
  msg::CupObservation observation;
  TimePoint received; // Original input receipt, not TF completion or cache access time.
  int stable_frames{0};
};

struct HeadCupTrackConfig {
  double inner_radius{.045}, wall_thickness{.005}, max_position_error{.15};
};

// Receives BODY samples already transformed into the stationary planning frame.
// Failed observations clear only the candidate window, never the last confirmed location.
class HeadCupTracker {
public:
  explicit HeadCupTracker(HeadCupTrackConfig config = {}) : config_(config) {}
  void observe(const CupState & sample);
  std::optional<CupState> last_known(const std::string & cup_id) const;

private:
  struct Track {
    int64_t last_stamp{0};
    std::deque<CupState> samples;
    std::optional<CupState> confirmed;
  };
  HeadCupTrackConfig config_;
  std::map<std::string, Track> tracks_;
};
} // namespace demo_1f
