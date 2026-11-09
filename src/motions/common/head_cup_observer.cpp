#include "demo_1f/motions/common/head_cup_observer.hpp"
#include <cmath>

namespace demo_1f {
bool HeadCupObserver::valid_sample(
    CupState & state, const std::string & cup_id, std::string & reason) const {
  const auto observed = context_.tracked_head_cup(cup_id, config_.max_age, reason);
  if (!observed) return false;
  state = *observed; // Validate and consume one consistent observation snapshot.
  const auto & observation = state.observation;
  if (!observation.valid) {
    reason = "Cup detection INVALID: " + (observation.detail.empty()
                                                 ? std::string("perception supplied no reason")
                                                 : observation.detail);
    return false;
  }
  if (observation.observation_type != msg::CupObservation::BODY) {
    reason = "Expected BODY cup observation, received type=" +
             std::to_string(observation.observation_type);
    return false;
  }
  if (observation.position_error > config_.max_position_error) {
    reason = "Cup position uncertainty=" + std::to_string(observation.position_error) +
             " m exceeds limit=" + std::to_string(config_.max_position_error) + " m";
    return false;
  }
  if (observation.position_error < config_.inner_radius + config_.wall_thickness) {
    reason = "Cup uncertainty does not cover configured body radius: error=" +
             std::to_string(observation.position_error) +
             " m, required=" + std::to_string(config_.inner_radius + config_.wall_thickness) + " m";
    return false;
  }
  if (observation.confidence < .15) {
    reason = "Cup valid depth coverage=" + std::to_string(observation.confidence) + " below 0.15";
    return false;
  }
  if (std::abs(observation.inner_radius - config_.inner_radius) > .003) {
    reason = "Cup radius mismatch: perception=" + std::to_string(observation.inner_radius) +
             " m, configured=" + std::to_string(config_.inner_radius) + " m (tolerance 0.003 m)";
    return false;
  }
  reason.clear();
  return true;
}

bool HeadCupObserver::poll(const std::string & cup_id,
    geometry_msgs::msg::PoseStamped & estimate,
    double & position_error,
    std::string & reason) const {
  CupState head;
  if (!valid_sample(head, cup_id, reason)) return false;
  estimate.header = head.observation.header;
  estimate.pose.position = head.observation.body_position;
  estimate.pose.orientation = geometry_msgs::msg::Quaternion{};
  estimate.pose.orientation.w = 1;
  position_error = head.observation.position_error;
  return true;
}
} // namespace demo_1f
