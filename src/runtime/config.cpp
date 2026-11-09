#include "demo_1f/runtime/config.hpp"
#include <cmath>
#include <set>
#include <stdexcept>

namespace demo_1f {
const char * to_string(ExecutionState s) {
  switch (s) {
  case ExecutionState::Idle:
    return "IDLE";
  case ExecutionState::Running:
    return "RUNNING";
  case ExecutionState::Cancelling:
    return "CANCELLING";
  case ExecutionState::Succeeded:
    return "SUCCEEDED";
  case ExecutionState::Failed:
    return "FAILED";
  case ExecutionState::Cancelled:
    return "CANCELLED";
  case ExecutionState::StopUnconfirmed:
    return "STOP_UNCONFIRMED";
  }
  return "UNKNOWN";
}

std::string validate_motion_sequence(const MotionSequence & sequence, const Config & config) {
  if (sequence.steps.empty()) {
    return "No taught steps configured";
  }
  std::set<std::string> names;
  for (const auto & step : sequence.steps) {
    const std::string prefix = step.name + ": ";
    if (step.name.empty() || !names.insert(step.name).second) {
      return prefix + "Duplicate/empty step name";
    }
    if (!std::isfinite(step.planning_time) || step.planning_time < 0 ||
        step.planning_time >= step.timeout) {
      return prefix + "Planning time must be zero (inherit) or positive and below timeout";
    }
    if (!std::isfinite(step.velocity_scale) || step.velocity_scale < 0 || step.velocity_scale > 1 ||
        !std::isfinite(step.acceleration_scale) || step.acceleration_scale < 0 ||
        step.acceleration_scale > 1) {
      return prefix + "Per-step scales must be zero (inherit) or in (0, 1]";
    }
    if (!std::isfinite(step.timeout) || step.timeout <= 0 || !std::isfinite(step.duration) ||
        step.duration <= 0 || !std::isfinite(step.tolerance) || step.tolerance <= 0) {
      return prefix + "Duration, timeout and tolerance must be positive and finite";
    }
    if (!std::isfinite(step.planning_tolerance) || step.planning_tolerance < 0 ||
        step.planning_tolerance > step.tolerance) {
      return prefix +
             "Planning tolerance must be zero (inherit) or no larger than completion tolerance";
    }
    if (step.gripper_wait_only && step.type != "gripper") {
      return prefix + "Gripper wait-only completion requires a gripper step";
    }
    if (step.position_only && step.type != "pose") {
      return prefix + "Position-only goals require a pose step";
    }
    if (!std::isfinite(step.tcp_roll_limit) || step.tcp_roll_limit < 0 ||
        (step.tcp_roll_limit > 0 && (!step.position_only || step.tcp_roll_limit <= M_PI / 180 ||
                                        step.tcp_roll_limit > M_PI / 2))) {
      return "Invalid TCP roll bounds; only position-only pose steps support roll constraints";
    }
    if ((step.tcp_pitch_downward_only && step.tcp_pitch_limit <= 0) ||
        !std::isfinite(step.tcp_pitch_limit) || step.tcp_pitch_limit < 0 ||
        (step.tcp_pitch_limit > 0 && (!step.position_only || step.tcp_pitch_limit <= M_PI / 180 ||
                                         step.tcp_pitch_limit >= M_PI / 2))) {
      return "Invalid TCP pitch bounds; only position-only pose steps support tilt constraints";
    }
    if (!std::isfinite(step.tcp_pitch_min) || step.tcp_pitch_min < 0 ||
        (step.tcp_pitch_min > 0 &&
            (!step.tcp_pitch_downward_only ||
                step.tcp_pitch_min + 2 * M_PI / 180 >= step.tcp_pitch_limit))) {
      return "Invalid minimum downward TCP pitch";
    }
    if (!std::isfinite(step.tcp_yaw_limit) || step.tcp_yaw_limit < 0 ||
        (step.tcp_yaw_limit > 0 && (!step.position_only || step.tcp_yaw_limit <= M_PI / 180 ||
                                       step.tcp_yaw_limit > M_PI))) {
      return "Invalid target TCP yaw tolerance";
    }
    if (step.tcp_yaw_inward_only &&
        (!step.position_only || (step.arm != "left" && step.arm != "right"))) {
      return "Inward TCP yaw requires a left/right position-only pose step";
    }
    if (step.type == "wait") {
      if (step.timeout < step.duration) {
        return prefix + "Timeout is shorter than wait";
      }
      continue;
    }
    if (step.type == "dual_joint" || step.type == "dual_joint_gripper") {
      if (config.coordinated_group.empty()) {
        return prefix + "Missing coordinated planning group";
      }
      for (const auto & side : {std::string("left"), std::string("right")}) {
        const auto & target = side == "left" ? step.left_target : step.right_target;
        if (config.arms.count(side) == 0 || target.size() != config.arms.at(side).joints.size()) {
          return prefix + "Missing calibrated dual-arm target: " + side;
        }
        for (double value : target) {
          if (!std::isfinite(value)) {
            return prefix + "Non-finite target";
          }
        }
      }
      if (step.type == "dual_joint_gripper") {
        if (config.arms.count(step.arm) == 0) {
          return prefix + "Unknown gripper arm";
        }
        if (step.gripper_target.size() != 1 || !std::isfinite(step.gripper_target[0])) {
          return prefix + "Missing gripper target";
        }
      }
      continue;
    }
    if (step.type == "joint_group") {
      const std::set<std::string> joints(step.joint_names.begin(), step.joint_names.end());
      if (step.group.empty() || step.joint_names.empty() || joints.count("") ||
          joints.size() != step.joint_names.size() || step.target.size() != joints.size()) {
        return prefix + "Missing/duplicate joint group targets";
      }
      for (double value : step.target) {
        if (!std::isfinite(value)) return prefix + "Non-finite joint group target";
      }
      continue;
    }
    if (config.arms.count(step.arm) == 0) {
      return prefix + "Unknown arm";
    }
    size_t count = 0;
    if (step.type == "joint" || step.type == "joint_gripper") {
      count = config.arms.at(step.arm).joints.size();
    } else if ((step.type == "linear" || step.type == "pose")) {
      count = 7;
    } else if (step.type == "gripper") {
      count = 1;
    } else {
      return prefix + "Unknown motion type";
    }
    if (step.target.size() != count) {
      return prefix + "Missing calibrated target (expected " + std::to_string(count) + " values)";
    }
    for (double value : step.target) {
      if (!std::isfinite(value)) {
        return prefix + "Non-finite target";
      }
    }
    if (step.type == "joint_gripper" &&
        (config.coordinated_group.empty() || step.gripper_target.size() != 1 ||
            !std::isfinite(step.gripper_target[0]))) {
      return prefix + "Missing gripper target or coordinated group";
    }
    if ((step.type == "linear" || step.type == "pose")) {
      if (step.frame.empty()) {
        return prefix + "Missing reference frame";
      }
      double norm = 0;
      for (size_t i = 3; i < 7; ++i) {
        norm += step.target[i] * step.target[i];
      }
      if (std::abs(norm - 1.0) > 0.01) {
        return prefix + "Quaternion must be normalized";
      }
    }
    if (step.type == "gripper" && step.timeout <= step.duration + config.settle_time) {
      return prefix + "Timeout must allow motion and settling";
    }
  }
  for (size_t i = 0; i < sequence.steps.size(); ++i) {
    if (!sequence.steps[i].blend_with_next) {
      continue;
    }
    const auto coordinated = [](const Step & step) {
      return step.type == "dual_joint" || step.type == "dual_joint_gripper";
    };
    if (!config.smooth_motion || i + 1 == sequence.steps.size() ||
        !coordinated(sequence.steps[i]) || !coordinated(sequence.steps[i + 1])) {
      return "Only adjacent coordinated arm moves can be blended with smooth_motion enabled";
    }
  }
  return {};
}

Config load_config(rclcpp::Node & node) {
  Config c;
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  auto text = [&](const std::string & key, const std::string & value) {
    return node.declare_parameter<std::string>(key, value, descriptor);
  };
  auto number = [&](const std::string & key, double value) {
    double result = node.declare_parameter<double>(key, value, descriptor);
    if (!std::isfinite(result) || result <= 0) {
      throw std::invalid_argument(key + " must be positive and finite");
    }
    return result;
  };
  c.planning_frame = text("planning_frame", c.planning_frame);
  c.joint_states_topic = text("joint_states_topic", c.joint_states_topic);
  c.move_group_action = text("move_group_action", c.move_group_action);
  c.pipeline = text("planning_pipeline", c.pipeline);
  c.gripper_transport = text("gripper_transport", c.gripper_transport);
  c.coordinated_group = text("coordinated_group", c.coordinated_group);
  c.coordinated_pipeline = text("coordinated_pipeline", c.coordinated_pipeline);
  c.coordinated_planner = text("coordinated_planner", c.coordinated_planner);
  if (c.gripper_transport != "topic" && c.gripper_transport != "action") {
    throw std::invalid_argument("gripper_transport must be topic or action");
  }
  c.state_max_age = number("state_max_age", c.state_max_age);
  c.tf_max_age = number("tf_max_age", c.tf_max_age);
  c.settle_time = number("settle_time", c.settle_time);
  c.cancel_timeout = number("cancel_timeout", c.cancel_timeout);
  c.velocity_scale = number("velocity_scale", c.velocity_scale);
  c.acceleration_scale = number("acceleration_scale", c.acceleration_scale);
  if (c.velocity_scale > 1 || c.acceleration_scale > 1) {
    throw std::invalid_argument("Motion scaling must be in (0, 1]");
  }
  c.planning_time = number("planning_time", c.planning_time);
  c.ompl_planning_attempts = node.declare_parameter<int>(
      "ompl_planning_attempts", c.ompl_planning_attempts, descriptor);
  if (c.ompl_planning_attempts < 1 || c.ompl_planning_attempts > 8) {
    throw std::invalid_argument("ompl_planning_attempts must be in [1, 8]");
  }
  c.pose_position_tolerance = number("pose_position_tolerance", c.pose_position_tolerance);
  c.pose_orientation_tolerance = number("pose_orientation_tolerance", c.pose_orientation_tolerance);
  c.stopped_velocity = number("stopped_velocity", c.stopped_velocity);
  c.gripper_stop_mode = text("gripper_stop_mode", c.gripper_stop_mode);
  c.require_passive_gripper_stationary = node.declare_parameter<bool>(
      "require_passive_gripper_stationary", c.require_passive_gripper_stationary, descriptor);
  if (c.gripper_stop_mode != "position_window" && c.gripper_stop_mode != "velocity") {
    throw std::invalid_argument("gripper_stop_mode must be position_window or velocity");
  }
  c.gripper_stop_window = number("gripper_stop_window", c.gripper_stop_window);
  c.gripper_completion_window = node.declare_parameter<double>(
      "gripper_completion_window", c.gripper_completion_window, descriptor);
  c.gripper_settle_time =
      node.declare_parameter<double>("gripper_settle_time", c.gripper_settle_time, descriptor);
  c.gripper_stop_position_tolerance =
      number("gripper_stop_position_tolerance", c.gripper_stop_position_tolerance);
  c.gripper_goal_tolerance = number("gripper_goal_tolerance", c.gripper_goal_tolerance);
  c.gripper_stop_max_gap = number("gripper_stop_max_gap", c.gripper_stop_max_gap);
  if (c.gripper_stop_max_gap >= c.gripper_stop_window) {
    throw std::invalid_argument("gripper_stop_max_gap must be less than gripper_stop_window");
  }
  if (!std::isfinite(c.gripper_completion_window) || c.gripper_completion_window < 0 ||
      (c.gripper_completion_window > 0 &&
          (c.gripper_completion_window > c.gripper_stop_window ||
              c.gripper_completion_window <= c.gripper_stop_max_gap)) ||
      !std::isfinite(c.gripper_settle_time) || c.gripper_settle_time < 0 ||
      c.gripper_settle_time > c.settle_time) {
    throw std::invalid_argument("Invalid gripper normal completion timing");
  }
  c.require_tcp_tf = node.declare_parameter<bool>("require_tcp_tf", true, descriptor);
  c.smooth_motion = node.declare_parameter<bool>("smooth_motion", c.smooth_motion, descriptor);
  c.path_simplification_tolerance =
      number("path_simplification_tolerance", c.path_simplification_tolerance);
  c.trajectory_sample_period = number("trajectory_sample_period", c.trajectory_sample_period);
  c.continuous_start_tolerance = number("continuous_start_tolerance", c.continuous_start_tolerance);
  if (c.trajectory_sample_period > 0.02) {
    throw std::invalid_argument("trajectory_sample_period must not exceed 0.02 s");
  }
  for (const std::string side : {"left", "right"}) {
    ArmConfig a;
    a.name = side;
    const std::string suffix = side == "left" ? "l" : "r";
    const std::string p = "arms." + side + ".";
    a.group = text(p + "group", "arm_" + suffix);
    a.tcp = text(p + "tcp", "gripper_" + suffix + "_tcp");
    a.gripper_joint = text(p + "gripper_joint", "gripper_" + suffix + "_joint1");
    a.trajectory_action =
        text(p + "trajectory_action", "/arm_" + suffix + "_controller/follow_joint_trajectory");
    a.trajectory_topic = text(p + "trajectory_topic",
        "/leader/joint_trajectory_command_broadcaster_" + side + "/joint_trajectory");
    std::vector<std::string> joints;
    for (int i = 1; i <= 7; ++i) {
      joints.push_back("arm_" + suffix + "_joint" + std::to_string(i));
    }
    a.joints = node.declare_parameter<std::vector<std::string>>(p + "joints", joints, descriptor);
    const std::set<std::string> unique(a.joints.begin(), a.joints.end());
    if (a.joints.empty() || unique.size() != a.joints.size() || unique.count("") ||
        a.group.empty() || a.tcp.empty() || a.gripper_joint.empty() ||
        unique.count(a.gripper_joint)) {
      throw std::invalid_argument(p + "invalid group/joints/TCP/gripper configuration");
    }
    c.arms.emplace(side, a);
    auto controlled = a.joints;
    controlled.push_back(a.gripper_joint);
    for (const auto & joint : controlled) {
      const std::string limit = "robot_description_planning.joint_limits." + joint + ".";
      MotionLimits limits;
      limits.velocity = number(limit + "max_velocity", limits.velocity);
      limits.acceleration = number(limit + "max_acceleration", limits.acceleration);
      limits.jerk = number(limit + "max_jerk", limits.jerk);
      c.motion_limits.emplace(joint, limits);
    }
  }
  for (const std::string direction : {"left_to_right", "right_to_left", "initial_pose"}) {
    MotionSequence sequence;
    sequence.direction = direction;
    sequence.giver = direction == "left_to_right" ? "left" : "right";
    sequence.receiver = direction == "left_to_right" ? "right" : "left";
    const std::string p = "motion_sequences." + direction + ".";
    const auto names = node.declare_parameter<std::vector<std::string>>(
        p + "steps", std::vector<std::string>{}, descriptor);
    for (const auto & name : names) {
      Step step;
      step.name = name;
      const auto q = p + name + ".";
      step.type = text(q + "type", "joint");
      step.group = text(q + "group", "");
      step.joint_names = node.declare_parameter<std::vector<std::string>>(
          q + "joint_names", std::vector<std::string>{}, descriptor);
      step.arm = text(q + "arm", "giver");
      if (step.arm == "giver") {
        step.arm = sequence.giver;
      }
      if (step.arm == "receiver") {
        step.arm = sequence.receiver;
      }
      step.target = node.declare_parameter<std::vector<double>>(
          q + "target", std::vector<double>{}, descriptor);
      step.left_target = node.declare_parameter<std::vector<double>>(
          q + "left_target", std::vector<double>{}, descriptor);
      step.right_target = node.declare_parameter<std::vector<double>>(
          q + "right_target", std::vector<double>{}, descriptor);
      step.gripper_target = node.declare_parameter<std::vector<double>>(
          q + "gripper_target", std::vector<double>{}, descriptor);
      step.frame = text(q + "frame", c.planning_frame);
      step.gripper_wait_only = node.declare_parameter<bool>(q + "gripper_wait_only", false, descriptor);
      step.duration = node.declare_parameter<double>(q + "duration", 1.0, descriptor);
      step.timeout = node.declare_parameter<double>(q + "timeout", 15.0, descriptor);
      step.tolerance = node.declare_parameter<double>(q + "tolerance", 0.02, descriptor);
      step.planning_tolerance =
          node.declare_parameter<double>(q + "planning_tolerance", 0.0, descriptor);
      step.velocity_scale = node.declare_parameter<double>(q + "velocity_scale", 0.0, descriptor);
      step.acceleration_scale =
          node.declare_parameter<double>(q + "acceleration_scale", 0.0, descriptor);
      step.blend_with_next = node.declare_parameter<bool>(q + "blend_with_next", false, descriptor);
      sequence.steps.push_back(step);
    }
    sequence.validation_error = validate_motion_sequence(sequence, c);
    c.motion_sequences.emplace(direction, sequence);
  }
  return c;
}
} // namespace demo_1f
