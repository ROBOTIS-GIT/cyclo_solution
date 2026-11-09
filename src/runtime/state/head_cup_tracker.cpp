#include "demo_1f/runtime/state/head_cup_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <rclcpp/time.hpp>

namespace demo_1f {
namespace {
double distance(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b) {
  return std::hypot(std::hypot(a.x - b.x, a.y - b.y), a.z - b.z);
}
} // namespace

void HeadCupTracker::observe(const CupState & sample) {
  const auto & input = sample.observation;
  if (input.source != "head" || input.cup_id.empty()) return;
  const int64_t stamp = rclcpp::Time(input.header.stamp).nanoseconds();
  auto & track = tracks_[input.cup_id];
  if (stamp <= track.last_stamp) return;
  track.last_stamp = stamp;
  const auto & p = input.body_position;
  const bool eligible = input.valid && input.observation_type == msg::CupObservation::BODY &&
      !input.header.frame_id.empty() && std::isfinite(p.x) && std::isfinite(p.y) &&
      std::isfinite(p.z) && std::isfinite(input.confidence) && input.confidence >= .15 &&
      std::isfinite(input.position_error) &&
      input.position_error >= config_.inner_radius + config_.wall_thickness &&
      input.position_error <= config_.max_position_error && std::isfinite(input.inner_radius) &&
      std::abs(input.inner_radius - config_.inner_radius) <= .003;
  if (!eligible) {
    track.samples.clear();
    return;
  }
  auto & samples = track.samples;
  if (!samples.empty()) {
    const auto & previous = samples.back();
    const double gap = (stamp - rclcpp::Time(previous.observation.header.stamp).nanoseconds()) * 1e-9;
    const double receipt_gap = std::chrono::duration<double>(sample.received - previous.received).count();
    if (gap >= 1.0 || receipt_gap < 0 || receipt_gap >= 1.0 ||
        input.header.frame_id != previous.observation.header.frame_id ||
        distance(p, samples.front().observation.body_position) >= .02 ||
        distance(p, previous.observation.body_position) >= .02) {
      samples.clear();
    }
  }
  samples.push_back(sample);
  if (samples.size() > 3) samples.pop_front();
  if (samples.size() != 3) return;

  CupState estimate = sample;
  auto & output = estimate.observation;
  output.body_position = geometry_msgs::msg::Point{};
  for (const auto & point : samples) {
    output.body_position.x += point.observation.body_position.x / 3;
    output.body_position.y += point.observation.body_position.y / 3;
    output.body_position.z += point.observation.body_position.z / 3;
  }
  output.position_error = 0;
  for (const auto & point : samples) {
    output.position_error = std::max(output.position_error,
        point.observation.position_error + distance(point.observation.body_position, output.body_position));
    output.confidence = std::min(output.confidence, point.observation.confidence);
  }
  if (output.position_error > config_.max_position_error) return;
  estimate.stable_frames = 3;
  track.confirmed = estimate; // Keep the true image timestamp and original receipt time.
}

std::optional<CupState> HeadCupTracker::last_known(const std::string & cup_id) const {
  const auto found = tracks_.find(cup_id);
  return found == tracks_.end() ? std::optional<CupState>{} : found->second.confirmed;
}
} // namespace demo_1f
