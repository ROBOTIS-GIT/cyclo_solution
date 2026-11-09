#pragma once

#include <chrono>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace demo_1f {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline double elapsed(TimePoint start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

enum class ExecutionState {
  Idle,
  Running,
  Cancelling,
  Succeeded,
  Failed,
  Cancelled,
  StopUnconfirmed
};
const char * to_string(ExecutionState state);

struct ExecutionResult {
  ExecutionState state{ExecutionState::Idle};
  std::string code;
  std::string detail;
  std::string phase{};
};

struct MotionLimits {
  double velocity{2.0};
  double acceleration{2.0};
  double jerk{4.0};
};

struct ArmConfig {
  std::string name;
  std::string group;
  std::string tcp;
  std::string trajectory_action;
  std::string trajectory_topic;
  std::string gripper_joint;
  std::vector<std::string> joints;
};

struct Step {
  std::string name;
  // joint: MoveIt PTP; linear: MoveIt LIN; gripper: controller command; wait: timer.
  std::string type;
  std::string arm;
  std::vector<double> target;
  std::string frame;
  double duration{1.0};
  double timeout{15.0};
  double tolerance{0.02};       // Measured completion tolerance.
  double planning_tolerance{0}; // Optional tighter joint planning goal; zero inherits tolerance.
  // dual_joint: both arms in one synchronized MoveIt plan; grippers hold position.
  std::vector<double> left_target{};
  std::vector<double> right_target{};
  // joint_gripper: receiver arm + gripper in the SAME trajectory/controller goal.
  // dual_joint_gripper: both arm targets plus the selected arm's gripper target.
  std::vector<double> gripper_target{};
  // Zero scales inherit the global setting. Blending is only for coordinated arm moves.
  double velocity_scale{0.0};
  double acceleration_scale{0.0};
  bool blend_with_next{false};
  // pose only: constrain TCP XYZ; let OMPL choose the end-effector orientation.
  bool position_only{false};
  double orientation_slack{0};          // OMPL goal rotation-vector component tolerance (radians).
  double planning_time{0};              // Zero inherits global MoveIt planning timeout.
  double tcp_path_margin{0};            // TCP bounding box around start/goal, meters.
  std::optional<double> tcp_path_min_z; // Planning-frame floor; clips only the box bottom.
  double tcp_roll_limit{
      0}; // Goal/completion radians; zero disables. SG2 hand axes in planning_frame.
  double tcp_yaw_limit{0};         // Goal yaw tolerance around target heading; radians.
  double tcp_pitch_limit{0};       // Radians; zero disables pitch restriction.
  double tcp_pitch_min{0};         // Optional downward-only minimum elevation magnitude, radians.
  bool tcp_yaw_inward_only{false}; // Goal/completion: left -90..0, right 0..90 degrees.
  bool tcp_pitch_downward_only{false}; // Positive body ZYX pitch points the hand down.
  // joint_group: named MoveIt group with explicit joint targets (head/lift initialization).
  std::string group{};
  std::vector<std::string> joint_names{};
  // Gripper only: wait for command duration without checking physical closure.
  bool gripper_wait_only{false};
};

struct MotionSequence {
  std::string direction;
  std::string giver;
  std::string receiver;
  std::vector<Step> steps;
  std::string validation_error;
};

struct Config {
  std::string planning_frame{"base_link"};
  std::string joint_states_topic{"/joint_states"};
  std::string move_group_action{"/move_action"};
  std::string pipeline{"pilz_industrial_motion_planner"};
  std::string gripper_transport{"topic"};
  std::string coordinated_group{"handover_arms"};
  std::string coordinated_pipeline{"ompl"};
  std::string coordinated_planner{"RRTConnectkConfigDefault"};
  double state_max_age{0.5};
  double tf_max_age{0.5};
  double settle_time{0.25};
  double cancel_timeout{3.0};
  double velocity_scale{0.1};
  double acceleration_scale{0.1};
  double planning_time{5.0};
  int ompl_planning_attempts{3}; // Shared time budget; compare joint-space path lengths.
  double pose_position_tolerance{0.005};
  double pose_orientation_tolerance{0.03};
  double stopped_velocity{0.03};
  std::string gripper_stop_mode{"position_window"};
  bool require_passive_gripper_stationary{true};
  double gripper_stop_window{0.5};
  double gripper_completion_window{
      0};                        // Zero inherits stop window; cancellation always uses stop window.
  double gripper_settle_time{0}; // Zero inherits general settle time.
  double gripper_stop_position_tolerance{0.004};
  double gripper_goal_tolerance{0.004};
  double gripper_stop_max_gap{0.1};
  bool require_tcp_tf{true};
  bool smooth_motion{false};
  double path_simplification_tolerance{0.002};
  double trajectory_sample_period{0.01};
  double continuous_start_tolerance{0.01};
  std::map<std::string, MotionLimits> motion_limits;
  std::map<std::string, ArmConfig> arms;
  std::map<std::string, MotionSequence> motion_sequences;
};

struct JointObservation {
  double position{0.0};
  std::optional<double> velocity;
  std::optional<double> effort;
  int64_t stamp_ns{0};
  TimePoint received;
  // Only populated after a contiguous gripper position window is complete.
  // Keep the original reported velocity above for diagnostics.
  std::optional<double> position_window_span;
  TimePoint position_window_start{};
  std::optional<double> completion_window_span;
  TimePoint completion_window_start{};
};

struct Snapshot {
  std::map<std::string, JointObservation> joints;
  std::string task_state{"IDLE"};
  std::string direction;
  std::string phase{"IDLE"};
  std::string last_completed_step;
  // Actual possession is UNKNOWN in the initial scripted, no-perception demo.
  ExecutionResult execution;
  std::string error_code;
  std::string detail;
};
} // namespace demo_1f
