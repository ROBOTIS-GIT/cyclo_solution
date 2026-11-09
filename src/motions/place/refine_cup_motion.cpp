#include "demo_1f/motions/place/refine_cup_motion.hpp"
#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <algorithm>
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
namespace {
double distance(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b) {
  return std::hypot(std::hypot(a.x - b.x, a.y - b.y), a.z - b.z);
}
} // namespace

geometry_msgs::msg::Pose tilt_tcp_target(
    const geometry_msgs::msg::Pose & pose, double degrees, double ceiling_degrees) {
  const auto pitch = tcp_pitch(pose.orientation);
  const auto yaw = tcp_yaw(pose.orientation);
  if (!pitch || !yaw || !std::isfinite(degrees) || degrees <= 0 ||
      !std::isfinite(ceiling_degrees) || ceiling_degrees <= 0 || ceiling_degrees > 80 ||
      *pitch + degrees * M_PI / 180 > ceiling_degrees * M_PI / 180 + 1e-9) {
    throw std::invalid_argument("Downward pitch target is invalid or exceeds observation ceiling");
  }
  tf2::Quaternion rotation;
  tf2::fromMsg(pose.orientation, rotation);
  // Increase hand pitch about the horizontal left axis for this heading.
  // This preserves hand yaw/roll even when hand roll is nonzero (forward=-TCP Z).
  const tf2::Vector3 axis(-std::sin(*yaw), std::cos(*yaw), 0);
  const tf2::Quaternion delta(axis, degrees * M_PI / 180);
  auto target = pose;
  target.orientation = tf2::toMsg((delta * rotation).normalized());
  return target; // TCP position stays exactly fixed.
}

bool RefineCupMotion::dispatch_refinement(const geometry_msgs::msg::Pose & pose) {
  Step step;
  step.name = phase_;
  const bool correction = phase_ == "CORRECT_TO_CUP_CENTER";
  step.type = correction ? "pose" : "linear";
  if (correction) {
    step.orientation_slack = 10.0 * M_PI / 180.0;
    step.tcp_path_margin = planning_retries_ >= 2 ? .06 : .04;
    step.planning_time = planning_retries_ == 0 ? 3.0 : 5.0;
    step.tcp_path_min_z = estimated_cup_pose_.pose.position.z + .015;
  }
  step.arm = arm_;
  step.frame = robot_.planning_frame;
  const auto & p = pose.position;
  const auto & q = pose.orientation;
  step.target = {p.x, p.y, p.z, q.x, q.y, q.z, q.w};
  step.timeout = 30;
  // Cartesian LIN can amplify joint acceleration near difficult arm configurations.
  step.velocity_scale = correction ? config_.refine_correction_speed : config_.refine_speed;
  step.acceleration_scale =
      correction ? config_.refine_correction_acceleration : step.velocity_scale * .25;
  approach_target_.pose = pose;
  return execute_step(step);
}

void RefineCupMotion::tilt_to_observe() {
  if (!execute_) {
    fail("RIM_NOT_CONFIRMED", "Rim not confirmed; preview performs no pitch adjustment");
    return;
  }
  const auto tcp = context_.pose_of(robot_.arms.at(arm_).tcp);
  if (!tcp) {
    fail("TCP_TF_MISSING", "Cannot compute fixed-position pitch adjustment");
    return;
  }
  if (!tilt_origin_) {
    tilt_origin_ = tcp;
  }
  if (distance(tcp->pose.position, tilt_origin_->pose.position) > .01) {
    fail("ROBOT_MOVED", "TCP position changed during observation adjustments");
    return;
  }
  geometry_msgs::msg::Pose target;
  double next_tilt = 0;
  try {
    const auto origin_pitch = tcp_pitch(tilt_origin_->pose.orientation);
    if (!origin_pitch) throw std::invalid_argument("Cannot determine starting TCP pitch");
    const double available = config_.refine_pitch_ceiling_deg - *origin_pitch * 180 / M_PI;
    next_tilt = std::min(tilt_progress_deg_ + config_.refine_pitch_step_deg, available);
    if (next_tilt <= tilt_progress_deg_ + 1e-6) {
      throw std::invalid_argument("Rim not confirmed at the observation pitch ceiling");
    }
    // Anchor every target to the original pose; the last increment stops at the ceiling.
    target = tilt_tcp_target(tilt_origin_->pose, next_tilt, config_.refine_pitch_ceiling_deg);
  } catch (const std::invalid_argument & error) {
    fail("PITCH_LIMIT", error.what());
    return;
  }
  tilt_progress_deg_ = next_tilt;
  phase("TILT_TO_OBSERVE");
  approach_target_ = *tcp;
  dispatch_refinement(target);
  return;
}

void RefineCupMotion::reset() {
  absent_frames_ = planning_retries_ = stable_progress_ = 0;
  tilt_progress_deg_ = 0;
  search_started_ = Clock::now();
  correction_start_joints_.clear();
  last_wrist_stamp_ = 0;
  view_started_.reset();
  gripper_settle_started_.reset();
  refinement_start_.reset();
  tilt_origin_.reset();
}

void RefineCupMotion::tick() {
  if (!running()) {
    return;
  }
  if (execute_) {
    executor_.tick();
  }
  if (result_.state == "RUNNING" &&
      (phase_ == "FIND_WRIST_RIM" || phase_ == "TILT_TO_OBSERVE") &&
      elapsed(search_started_) >= config_.refine_search_timeout) {
    const std::string detail = "Wrist rim search exceeded " +
        std::to_string(config_.refine_search_timeout) + " s";
    if (!moving_) {
      fail("RIM_SEARCH_TIMEOUT", detail);
      return;
    }
    const auto state = executor_.result().state;
    if (state == ExecutionState::Running) {
      result_.detail = detail + "; waiting for pitch stop confirmation";
      executor_.cancel("RIM_SEARCH_TIMEOUT", detail);
    } else if (state == ExecutionState::Succeeded) {
      moving_ = false;
      fail("RIM_SEARCH_TIMEOUT", detail);
      return;
    }
    // An in-flight pitch must confirm its stop before this motion returns a result.
  }
  // Between motions, allow a bounded settling interval without assuming a secure grasp.
  if (execute_ && !moving_ && result_.state == "RUNNING") {
    std::string reason;
    if (!context_.ready(reason)) {
      fail("ROBOT_STATE_INVALID", reason);
      return;
    }
    const auto snapshot = context_.snapshot();
    for (const auto & [name, arm] : robot_.arms) {
      const auto joint = snapshot.joints.find(arm.gripper_joint);
      if (joint == snapshot.joints.end()) {
        fail("ROBOT_STATE_INVALID", "Missing gripper feedback: " + arm.gripper_joint);
        return;
      }
      if (!context_.joint_ready_for_motion(arm.gripper_joint, joint->second, reason,
              executor_.result().state == ExecutionState::Succeeded)) {
        if (!gripper_settle_started_) gripper_settle_started_ = Clock::now();
        if (elapsed(*gripper_settle_started_) >= 2.0) {
          fail("ROBOT_NOT_READY", "Gripper did not settle within 2 s; inspect grasp: " + reason);
        } else {
          result_.detail = "Waiting for gripper to settle (up to 2 s; grasp unverified): " + reason;
        }
        return;
      }
    }
    if (gripper_settle_started_) result_.detail.clear();
    gripper_settle_started_.reset();
  }
  if (scene_pending_) {
    const auto outcome = scene_.poll();
    if (outcome.state == ExecutionState::Running) {
      return;
    }
    scene_pending_ = false;
    if (outcome.state != ExecutionState::Succeeded) {
      fail(outcome.code, phase_ + ": " + outcome.detail);
      return;
    }
    if (result_.state == "CANCELLING") {
      result_.state = "CANCELLED";
      result_.code = "CANCELLED";
      return;
    }
    const auto current = context_.pose_of(robot_.arms.at(arm_).tcp);
    if (!current || !refinement_start_ ||
        distance(current->pose.position, refinement_start_->pose.position) > .01) {
      fail("ROBOT_MOVED", "TCP moved while preparing correction");
      return;
    }
    const auto & a = current->pose.orientation;
    const auto & b = refinement_start_->pose.orientation;
    const double dot = std::abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
    if (2 * std::acos(std::min(1., dot)) > .05) {
      fail("ROBOT_MOVED", "TCP orientation changed while preparing correction");
      return;
    }
    phase("CORRECT_TO_CUP_CENTER");
    dispatch_refinement(approach_target_.pose);
    return;
  }
  std::string reason;
  if (execute_ && !context_.ready(reason)) {
    if (moving_) {
      executor_.cancel("ROBOT_STATE_INVALID", reason);
    } else {
      fail("ROBOT_STATE_INVALID", reason);
      return;
    }
  }
  if (moving_) {
    const auto outcome = executor_.result();
    if (outcome.state == ExecutionState::Running || outcome.state == ExecutionState::Cancelling) {
      return;
    }
    moving_ = false;
    if (outcome.state == ExecutionState::Failed &&
        (outcome.code == "PLANNING_TIMEOUT" || outcome.code == "PLANNING_FAILED") &&
        phase_ == "CORRECT_TO_CUP_CENTER" && result_.state != "CANCELLING" &&
        planning_retries_ < 2) {
      const auto snapshot = context_.snapshot();
      for (const auto & [name, position] : correction_start_joints_) {
        const auto joint = snapshot.joints.find(name);
        if (joint == snapshot.joints.end() ||
            std::abs(joint->second.position - position) > robot_.continuous_start_tolerance ||
            !context_.joint_ready_for_motion(name, joint->second, reason)) {
          fail("ROBOT_MOVED", "Cannot replan correction: " + name + "; " + reason);
          return;
        }
      }
      const auto tcp = context_.pose_of(robot_.arms.at(arm_).tcp);
      if (!tcp || !refinement_start_ ||
          distance(tcp->pose.position, refinement_start_->pose.position) > .005) {
        fail("ROBOT_MOVED", "TCP moved before correction replan");
        return;
      }
      ++planning_retries_;
      result_.detail = "Replanning correction " + std::to_string(planning_retries_) +
                       "/2 after planning failure: " + outcome.detail;
      dispatch_refinement(approach_target_.pose);
      return;
    }
    if (outcome.state == ExecutionState::Failed ||
        outcome.state == ExecutionState::StopUnconfirmed) {
      fail(outcome.code, phase_ + ": " + outcome.detail);
      return;
    }
    if (result_.state == "CANCELLING" || outcome.state == ExecutionState::Cancelled) {
      result_.state = "CANCELLED";
      result_.code = "CANCELLED";
      return;
    }
    if (outcome.state != ExecutionState::Succeeded) {
      fail("EXECUTOR_STATE", "Unexpected correction executor state");
      return;
    }
    result_.last_completed_step = phase_;
    if (phase_ == "TILT_TO_OBSERVE") {
      absent_frames_ = stable_progress_ = 0;
      view_started_.reset();
      last_wrist_stamp_ = 0;
      phase("FIND_WRIST_RIM"); // Require images acquired after tilt completion.
      return;
    }
    if (phase_ == "CORRECT_TO_CUP_CENTER") {
      Step release;
      release.name = "OPEN_GRIPPER";
      release.type = "gripper";
      release.arm = arm_;
      release.target = {config_.refine_gripper_open};
      release.duration = .6;
      release.timeout = 12;
      phase("OPEN_GRIPPER");
      execute_step(release);
      return;
    }
    at_observation_pose_ = true;
    result_.state = "SUCCEEDED";
    result_.sequence_completed = true;
    result_.detail = "TCP centered, lowered and gripper opened; object landing is not verified";
    phase("RELEASE_COMPLETE");
    return;
  }
  if (phase_ != "FIND_WRIST_RIM") {
    return;
  }
  find_wrist_target();
}

void RefineCupMotion::find_wrist_target() {
  const auto observation = context_.cup_observation(cup_id_, "wrist_left");
  if (!observation || elapsed(observation->received) > .8 ||
      observation->received < phase_started_ ||
      !context_.cup_acquired_after(*observation, phase_started_)) {
    view_started_.reset();
    absent_frames_ = stable_progress_ = 0;
    result_.detail =
        "Waiting for fresh wrist observation after request/tilt; no blind tilt on missing input";
    if (elapsed(phase_started_) > config_.refine_wait) {
      fail("WRIST_INPUT_MISSING", result_.detail);
    }
    return;
  }
  const auto & input = observation->observation;
  const int64_t stamp = rclcpp::Time(input.header.stamp).nanoseconds();
  if (stamp == last_wrist_stamp_) {
    if (elapsed(view_started_.value_or(phase_started_)) > config_.refine_wait) {
      fail("WRIST_NOT_READY", result_.detail);
    }
    return;
  }
  last_wrist_stamp_ = stamp;
  // Match the known-radius fit's 75-degree minimum: confidence is arc / 360.
  const bool geometry_ready = input.valid && input.observation_type == 2 &&
                              input.header.frame_id == robot_.planning_frame &&
                              input.position_error <= .015 && input.confidence >= 75.0 / 360.0 &&
                              std::abs(input.inner_radius - config_.inner_radius) <= .003;
  const bool confirmed = geometry_ready && observation->stable_frames >= 3;
  if (!confirmed) {
    result_.detail =
        input.valid ? "Waiting for stable, dimension-matched wrist RIM observation" : input.detail;
    int progress = 0;
    if (input.detail.rfind("RIM_UNSTABLE: 1/3", 0) == 0) progress = 1;
    if (input.detail.rfind("RIM_UNSTABLE: 2/3", 0) == 0) progress = 2;
    if (geometry_ready) progress = 2 + observation->stable_frames;
    if (progress > stable_progress_) {
      stable_progress_ = progress;
      view_started_ = Clock::now();
      absent_frames_ = 0;
      return; // New stability progress gets a full observation interval at this angle.
    }
    const bool usable_image = input.valid || input.detail.rfind("NO_CANDIDATE:", 0) == 0 ||
                              input.detail.rfind("RIM_UNCERTAIN:", 0) == 0 ||
                              input.detail.rfind("RIM_UNSTABLE:", 0) == 0;
    if (usable_image) {
      if (!view_started_) {
        view_started_ = Clock::now();
      }
      ++absent_frames_;
      if (absent_frames_ >= 3 && elapsed(*view_started_) >= config_.refine_view_wait) {
        tilt_to_observe();
        return;
      }
    } else {
      view_started_.reset();
      absent_frames_ = stable_progress_ = 0;
    }
    if (elapsed(view_started_.value_or(phase_started_)) > config_.refine_wait) {
      fail("RIM_NOT_CONFIRMED", result_.detail);
    }
    return;
  }
  view_started_.reset();
  absent_frames_ = 0;
  const auto tcp = context_.pose_of(robot_.arms.at(arm_).tcp);
  if (!tcp) {
    fail("TCP_TF_MISSING", "Cannot preserve measured TCP orientation");
    return;
  }
  refinement_start_ = tcp;
  correction_start_joints_.clear();
  const auto snapshot = context_.snapshot();
  for (const auto & [side, arm] : robot_.arms) {
    auto names = arm.joints;
    names.push_back(arm.gripper_joint);
    for (const auto & name : names) {
      const auto joint = snapshot.joints.find(name);
      if (joint == snapshot.joints.end()) {
        fail("ROBOT_NOT_READY", "Missing joint feedback: " + name);
        return;
      }
      correction_start_joints_[name] = joint->second.position;
    }
  }
  estimated_cup_pose_.header = input.header;
  estimated_cup_pose_.pose.position = input.body_position;
  estimated_cup_pose_.pose.orientation.w = 1;
  approach_target_ = *tcp;
  approach_target_.pose.position.x = input.body_position.x;
  approach_target_.pose.position.y = input.body_position.y;
  approach_target_.pose.position.z =
      std::max(tcp->pose.position.z, input.body_position.z + config_.refine_clearance) -
      config_.refine_lower_distance;
  if (std::min(tcp->pose.position.z, approach_target_.pose.position.z) <
      input.body_position.z + .015) {
    fail("RELEASE_CLEARANCE", "Correction must stay at least 15mm above measured rim");
    return;
  }
  if (distance(approach_target_.pose.position, tcp->pose.position) >
      config_.refine_max_correction) {
    fail("CORRECTION_TOO_FAR", "Rim target exceeds local correction distance limit");
    return;
  }
  if (!execute_) {
    result_.state = "SUCCEEDED";
    result_.sequence_completed = true;
    result_.detail = "RIM correction preview; execution will move directly to lowered center and "
                     "open gripper; no preview motion";
    phase("CORRECTION_PREVIEW_READY");
    return;
  }
  // Replace only the demo cup model using measured rim height. Keep collision checking.
  auto body = estimated_cup_pose_.pose;
  body.position.z -= config_.cup_height / 2;
  phase("PREPARE_REFINED_SCENE");
  scene_pending_ = scene_.apply(make_approach_scene(robot_, config_, body, input.position_error));
  if (!scene_pending_) {
    fail("SCENE_UNAVAILABLE", "Cannot prepare refined cup geometry");
  }
}
} // namespace demo_1f
