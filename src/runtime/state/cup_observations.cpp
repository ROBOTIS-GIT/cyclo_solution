#include "demo_1f/runtime/state/robot_context.hpp"
#include <algorithm>
#include <cmath>

namespace demo_1f {
void RobotContext::queue_cup(msg::CupObservation::ConstSharedPtr input) {
  const auto stamp = rclcpp::Time(input->header.stamp);
  const double age = (node_.now() - stamp).seconds();
  if (input->cup_id.empty() ||
      (input->source != "head" && input->source != "wrist_left" &&
          input->source != "wrist_right") ||
      input->header.frame_id.empty() || stamp.nanoseconds() <= 0 || age < -.1 ||
      age > config_.state_max_age) {
    observe_cup(*input); // Preserve the validation diagnostics, without queuing bad timestamps.
    return;
  }
  if (!input->valid) {
    // Loss of detection invalidates the cache immediately, even if TF is delayed.
    // Older queued detections must not reappear after this loss message.
    pending_cups_.erase(std::remove_if(pending_cups_.begin(),
                            pending_cups_.end(),
                            [&](const PendingCup & pending) {
                              const auto & old = *pending.observation;
                              return old.cup_id == input->cup_id && old.source == input->source &&
                                     rclcpp::Time(old.header.stamp) <= stamp;
                            }),
        pending_cups_.end());
    observe_cup(*input);
    return;
  }
  // Keep multiple timestamps: replacing each delayed frame with the newest one can
  // starve the consumer forever when TF consistently trails the camera stream.
  constexpr size_t maximum_pending = 32;
  if (pending_cups_.size() >= maximum_pending) {
    pending_cups_.pop_front();
  }
  pending_cups_.push_back({std::move(input), Clock::now()});
  process_pending_cups();
}

void RobotContext::process_pending_cups() {
  for (auto it = pending_cups_.begin(); it != pending_cups_.end();) {
    const auto & input = *it->observation;
    const auto stamp = rclcpp::Time(input.header.stamp);
    const double age = (node_.now() - stamp).seconds();
    if (age < -.1 || age > config_.state_max_age || elapsed(it->received) > config_.state_max_age) {
      RCLCPP_WARN_THROTTLE(node_.get_logger(),
          *node_.get_clock(),
          2000,
          "Cup TF wait expired: id=%s/%s frame=%s age=%.3f s limit=%.3f s",
          input.cup_id.c_str(),
          input.source.c_str(),
          input.header.frame_id.c_str(),
          age,
          config_.state_max_age);
      it = pending_cups_.erase(it);
      continue;
    }
    // Nonblocking lookup: a blocking wait here would prevent this same executor
    // from processing the TF callbacks that we are waiting for.
    if (input.header.frame_id != config_.planning_frame &&
        !buffer_.canTransform(
            config_.planning_frame, input.header.frame_id, stamp, rclcpp::Duration(0, 0))) {
      ++it; // Other cameras/frames can still make progress.
      continue;
    }
    // Keep input receipt time for phase/freshness checks; TF completion is not a new image.
    observe_cup(input, it->received);
    it = pending_cups_.erase(it);
  }
}

std::optional<CupState> RobotContext::cup_observation(
    const std::string & id, const std::string & source) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cups_.find(id + "/" + source);
  if (it == cups_.end()) {
    std::string known;
    for (const auto & item : cups_) {
      known += item.first + " ";
    }
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "Cup cache missing: requested=%s/%s topic=%s publishers=%zu cached=[%s]",
        id.c_str(),
        source.c_str(),
        cup_subscription_->get_topic_name(),
        cup_subscription_->get_publisher_count(),
        known.c_str());
    return {};
  }
  auto copy = it->second;
  double age = (node_.now() - rclcpp::Time(copy.observation.header.stamp)).seconds();
  if (age < -.1 || age > config_.state_max_age) {
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "Cup cache expired: id=%s/%s age=%.3f s limit=%.3f s",
        id.c_str(),
        source.c_str(),
        age,
        config_.state_max_age);
    return {};
  }
  return copy;
}

bool RobotContext::cup_acquired_after(const CupState & state, TimePoint after) const {
  const double age = (node_.now() - rclcpp::Time(state.observation.header.stamp)).seconds();
  return age >= -.1 && age <= elapsed(after);
}

std::optional<CupState> RobotContext::tracked_head_cup(
    const std::string & id, double maximum_age, std::string & reason) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto saved = head_cup_tracker_.last_known(id);
  if (!saved) {
    reason = "No saved head cup: waiting for three valid, stable BODY observations";
    return {};
  }
  const double image_age = (node_.now() - rclcpp::Time(saved->observation.header.stamp)).seconds();
  const double receipt_age = elapsed(saved->received);
  if (!std::isfinite(maximum_age) || maximum_age <= 0 || image_age < -.1 ||
      image_age > maximum_age || receipt_age > maximum_age) {
    reason = "Saved head cup expired: image age=" + std::to_string(image_age) +
        " s, receipt age=" + std::to_string(receipt_age) +
        " s, limit=" + std::to_string(maximum_age) + " s; reobserve the cup";
    return {};
  }
  reason.clear();
  return saved;
}

bool RobotContext::observe_cup(const msg::CupObservation & input, TimePoint received) {
  if (input.cup_id.empty() ||
      (input.source != "head" && input.source != "wrist_left" && input.source != "wrist_right")) {
    return false;
  }
  auto stamp = rclcpp::Time(input.header.stamp);
  double age = (node_.now() - stamp).seconds();
  if (stamp.nanoseconds() <= 0 || age < -.1 || age > config_.state_max_age) {
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "Cup input timestamp rejected: id=%s/%s age=%.3f s limit=%.3f s stamp=%.3f now=%.3f",
        input.cup_id.c_str(),
        input.source.c_str(),
        age,
        config_.state_max_age,
        stamp.seconds(),
        node_.now().seconds());
    return false;
  }
  auto key = input.cup_id + "/" + input.source;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cups_.find(key);
    if (it != cups_.end() && stamp <= rclcpp::Time(it->second.observation.header.stamp)) {
      return false;
    }
    if (!input.valid) {
      cups_[key] = {input, received, 0};
      head_cup_tracker_.observe(cups_[key]);
      return true;
    }
  }
  if ((input.source == "head" && input.observation_type != msg::CupObservation::BODY) ||
      ((input.source == "wrist_left" || input.source == "wrist_right") &&
          input.observation_type != 2)) {
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "Cup input type rejected: id=%s/%s type=%u",
        input.cup_id.c_str(),
        input.source.c_str(),
        static_cast<unsigned>(input.observation_type));
    return false;
  }
  const auto & p = input.body_position;
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
      !std::isfinite(input.inner_radius) || input.inner_radius <= 0 ||
      !std::isfinite(input.position_error) || input.position_error < 0 ||
      !std::isfinite(input.confidence) || input.confidence < 0 || input.confidence > 1) {
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "Cup input geometry rejected: id=%s/%s xyz=[%.3f %.3f %.3f] radius=%.3f error=%.3f "
        "confidence=%.3f",
        input.cup_id.c_str(),
        input.source.c_str(),
        p.x,
        p.y,
        p.z,
        input.inner_radius,
        input.position_error,
        input.confidence);
    return false;
  }
  geometry_msgs::msg::PoseStamped pose;
  pose.header = input.header;
  pose.pose.position = input.body_position;
  pose.pose.orientation.w = 1;
  auto transformed = transform_pose(pose);
  if (!transformed) {
    RCLCPP_WARN_THROTTLE(node_.get_logger(),
        *node_.get_clock(),
        2000,
        "Cup input TF rejected: id=%s/%s frame=%s target=%s age=%.3f s",
        input.cup_id.c_str(),
        input.source.c_str(),
        input.header.frame_id.c_str(),
        config_.planning_frame.c_str(),
        age);
    return false;
  }
  auto output = input;
  output.header.frame_id = config_.planning_frame;
  output.body_position = transformed->pose.position;
  std::lock_guard<std::mutex> lock(mutex_);
  int stable = 1;
  auto it = cups_.find(key);
  // Match wrist receipt cadence to the existing 0.6 s image-stamp window.
  const double maximum_gap = input.source == "head" ? .3 : .6;
  if (it != cups_.end() && it->second.observation.valid &&
      it->second.observation.observation_type == output.observation_type &&
      received >= it->second.received &&
      std::chrono::duration<double>(received - it->second.received).count() < maximum_gap &&
      (rclcpp::Time(output.header.stamp) - rclcpp::Time(it->second.observation.header.stamp))
              .seconds() < maximum_gap) {
    const auto & previous = it->second.observation.body_position;
    const auto & current = output.body_position;
    if (std::hypot(std::hypot(previous.x - current.x, previous.y - current.y),
            previous.z - current.z) < .02) {
      stable = it->second.stable_frames + 1;
    }
  }
  cups_[key] = {output, received, stable};
  head_cup_tracker_.observe(cups_[key]);
  return true;
}
} // namespace demo_1f
