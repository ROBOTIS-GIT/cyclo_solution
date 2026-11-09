#include "demo_1f/runtime/state/robot_context.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>

namespace demo_1f {
void RobotContext::observe_joints(const sensor_msgs::msg::JointState & msg) {
  if (msg.name.size() != msg.position.size() ||
      (!msg.velocity.empty() && msg.velocity.size() != msg.name.size()) ||
      (!msg.effort.empty() && msg.effort.size() != msg.name.size())) {
    return;
  }
  const auto stamp = rclcpp::Time(msg.header.stamp).nanoseconds();
  const double age = (node_.now().nanoseconds() - stamp) * 1e-9;
  if (stamp <= 0 || age > config_.state_max_age || age < -0.1) {
    return;
  }
  std::set<std::string> names;
  for (size_t i = 0; i < msg.name.size(); ++i) {
    if (msg.name[i].empty() || !names.insert(msg.name[i]).second ||
        !std::isfinite(msg.position[i]) ||
        (!msg.velocity.empty() && !std::isfinite(msg.velocity[i]))) {
      return;
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < msg.name.size(); ++i) {
    const auto old = state_.joints.find(msg.name[i]);
    if (old != state_.joints.end() && stamp <= old->second.stamp_ns) {
      continue;
    }
    JointObservation observation;
    observation.position = msg.position[i];
    observation.stamp_ns = stamp;
    observation.received = Clock::now();
    if (!msg.velocity.empty()) {
      observation.velocity = msg.velocity[i];
    } else if (old != state_.joints.end()) {
      const double dt = (stamp - old->second.stamp_ns) * 1e-9;
      if (dt > 0 && dt <= config_.state_max_age) {
        observation.velocity = (msg.position[i] - old->second.position) / dt;
      }
    }
    // JointStateBroadcaster uses NaN for unprovided effort interfaces.
    // Effort is optional; never turn an unknown effort into a measured force.
    if (!msg.effort.empty() && std::isfinite(msg.effort[i])) {
      observation.effort = msg.effort[i];
    }
    const auto history_it = gripper_history_.find(msg.name[i]);
    if (history_it != gripper_history_.end() && config_.gripper_stop_mode == "position_window") {
      auto & history = history_it->second;
      const double max_gap = std::min(config_.state_max_age, config_.gripper_stop_max_gap);
      if (!history.empty() &&
          (std::chrono::duration<double>(observation.received - history.back().received).count() >
                  max_gap ||
              (stamp - history.back().stamp_ns) * 1e-9 > max_gap)) {
        history.clear();
      }
      history.push_back({observation.position, stamp, observation.received});
      const auto covers_window = [&](const PositionSample & start) {
        return std::chrono::duration<double>(observation.received - start.received).count() >=
                   config_.gripper_stop_window &&
               (stamp - start.stamp_ns) * 1e-9 >= config_.gripper_stop_window;
      };
      // Retain the sample just before the window boundary. Both receipt time
      // and message time must span the window; a buffered burst is insufficient.
      while (history.size() > 1 && covers_window(history[1])) {
        history.pop_front();
      }
      const double completion_window = config_.gripper_completion_window > 0
                                           ? config_.gripper_completion_window
                                           : config_.gripper_stop_window;
      auto first = history.begin();
      for (auto it = history.begin(); it != history.end(); ++it) {
        if (std::chrono::duration<double>(observation.received - it->received).count() >=
                completion_window &&
            (stamp - it->stamp_ns) * 1e-9 >= completion_window) {
          first = it;
        } else {
          break;
        }
      }
      if (std::distance(first, history.end()) >= 3 &&
          std::chrono::duration<double>(observation.received - first->received).count() >=
              completion_window &&
          (stamp - first->stamp_ns) * 1e-9 >= completion_window) {
        const auto limits = std::minmax_element(
            first, history.end(), [](const PositionSample & a, const PositionSample & b) {
              return a.position < b.position;
            });
        observation.completion_window_span = limits.second->position - limits.first->position;
        observation.completion_window_start = first->received;
      }
      if (history.size() >= 3 && covers_window(history.front())) {
        auto limits = std::minmax_element(
            history.begin(), history.end(), [](const PositionSample & a, const PositionSample & b) {
              return a.position < b.position;
            });
        observation.position_window_span = limits.second->position - limits.first->position;
        observation.position_window_start = history.front().received;
      }
    }
    state_.joints[msg.name[i]] = observation;
  }
}

bool RobotContext::joint_fresh(const JointObservation & joint) const {
  const double age = (node_.now().nanoseconds() - joint.stamp_ns) * 1e-9;
  return elapsed(joint.received) <= config_.state_max_age && age >= -0.1 &&
         age <= config_.state_max_age;
}

bool RobotContext::is_gripper_joint(const std::string & name) const {
  return std::any_of(config_.arms.begin(), config_.arms.end(), [&](const auto & entry) {
    return entry.second.gripper_joint == name;
  });
}

bool RobotContext::joint_ready_for_motion(const std::string & name,
    const JointObservation & joint, std::string & reason, bool normal_completion) const {
  if (!config_.require_passive_gripper_stationary && is_gripper_joint(name)) {
    reason = joint_fresh(joint) ? "" : "Missing fresh gripper feedback: " + name;
    return reason.empty();
  }
  return joint_stationary(name, joint, reason, TimePoint{}, normal_completion);
}

bool RobotContext::joint_stationary(const std::string & name,
    const JointObservation & joint,
    std::string & reason,
    TimePoint after,
    bool normal_completion) const {
  if (!joint_fresh(joint) || joint.received < after) {
    reason = "Missing fresh feedback after command: " + name;
    return false;
  }
  const bool gripper = is_gripper_joint(name);
  if (gripper && config_.gripper_stop_mode == "position_window") {
    const double age = (node_.now().nanoseconds() - joint.stamp_ns) * 1e-9;
    if (elapsed(joint.received) > config_.gripper_stop_max_gap ||
        age > config_.gripper_stop_max_gap) {
      reason = "Gripper position feedback interrupted: " + name;
      return false;
    }
    const auto & span =
        normal_completion ? joint.completion_window_span : joint.position_window_span;
    const auto start =
        normal_completion ? joint.completion_window_start : joint.position_window_start;
    if (!span || start < after) {
      reason = "Waiting for continuous gripper position feedback: " + name + " (window=" +
               std::to_string(normal_completion && config_.gripper_completion_window > 0
                                  ? config_.gripper_completion_window
                                  : config_.gripper_stop_window) +
               " s)";
      return false;
    }
    if (*span > config_.gripper_stop_position_tolerance) {
      reason = "Gripper position is not stationary: " + name + " (range=" + std::to_string(*span) +
               " rad, limit=" + std::to_string(config_.gripper_stop_position_tolerance) + " rad)";
      return false;
    }
  } else if (!joint.velocity || std::abs(*joint.velocity) > config_.stopped_velocity) {
    reason = "Joint velocity is not stationary: " + name +
             " (velocity=" + (joint.velocity ? std::to_string(*joint.velocity) : "unknown") +
             " rad/s, limit=" + std::to_string(config_.stopped_velocity) + " rad/s)";
    return false;
  }
  reason.clear();
  return true;
}

bool RobotContext::ready(std::string & reason) const {
  const auto state = snapshot();
  for (const auto & [name, arm] : config_.arms) {
    auto joints = arm.joints;
    joints.push_back(arm.gripper_joint);
    for (const auto & joint : joints) {
      const auto it = state.joints.find(joint);
      if (it == state.joints.end() || !joint_fresh(it->second)) {
        reason = "Missing/stale joint state: " + joint;
        return false;
      }
    }
    if (config_.require_tcp_tf && !pose_of(arm.tcp)) {
      reason = "Missing/stale TCP transform: " + name + "/" + arm.tcp;
      return false;
    }
  }
  reason.clear();
  return true;
}

std::optional<std::string> RobotContext::capture_targets(std::string & reason) const {
  if (!ready(reason)) {
    return std::nullopt;
  }
  const auto state = snapshot();
  std::ostringstream out;
  out << std::setprecision(9);
  for (const auto & [name, arm] : config_.arms) {
    auto required = arm.joints;
    required.push_back(arm.gripper_joint);
    for (const auto & joint : required) {
      const auto & observed = state.joints.at(joint);
      if (!joint_stationary(joint, observed, reason)) {
        return std::nullopt;
      }
    }
    out << name << ":\n  joint_target: [";
    for (size_t i = 0; i < arm.joints.size(); ++i) {
      if (i) {
        out << ", ";
      }
      out << state.joints.at(arm.joints[i]).position;
    }
    out << "]\n  gripper_target: [" << state.joints.at(arm.gripper_joint).position << "]\n";
    const auto pose = pose_of(arm.tcp);
    if (pose) {
      const auto & p = pose->pose.position;
      const auto & q = pose->pose.orientation;
      out << "  frame: " << config_.planning_frame << "\n  linear_target: [" << p.x << ", " << p.y
          << ", " << p.z << ", " << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "]\n";
    }
  }
  return out.str();
}
} // namespace demo_1f
