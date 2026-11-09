#pragma once
#include "demo_1f/core/types.hpp"
#include "demo_1f/msg/cup_observation.hpp"
#include "demo_1f/runtime/state/head_cup_tracker.hpp"
#include <deque>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace demo_1f {
// Shared data and coordinate queries. It never issues a robot command.
class RobotContext {
public:
  RobotContext(rclcpp::Node & node, const Config & config, HeadCupTrackConfig head_tracking = {});
  Snapshot snapshot() const;
  bool observe_cup(const msg::CupObservation & observation, TimePoint received = Clock::now());
  std::optional<CupState> cup_observation(const std::string & id, const std::string & source) const;
  std::optional<CupState> tracked_head_cup(
      const std::string & id, double maximum_age, std::string & reason) const;
  bool cup_acquired_after(const CupState & state, TimePoint after) const;
  void observe_joints(const sensor_msgs::msg::JointState & message);
  bool joint_fresh(const JointObservation & joint) const;
  bool is_gripper_joint(const std::string & name) const;
  bool joint_ready_for_motion(const std::string & name,
      const JointObservation & joint, std::string & reason,
      bool normal_completion = false) const;
  bool joint_stationary(const std::string & name,
      const JointObservation & joint,
      std::string & reason,
      TimePoint after = TimePoint{},
      bool normal_completion = false) const;
  bool ready(std::string & reason) const;
  std::optional<std::string> capture_targets(std::string & reason) const;
  std::optional<geometry_msgs::msg::PoseStamped> pose_of(const std::string & frame) const;
  std::optional<geometry_msgs::msg::PoseStamped> transform_pose(
      const geometry_msgs::msg::PoseStamped & input) const;
  void set_task(const std::string & state,
      const std::string & direction,
      const std::string & code = {},
      const std::string & detail = {});
  void set_phase(const std::string & phase, const std::string & completed);
  void set_execution(const ExecutionResult & result);

private:
  void queue_cup(msg::CupObservation::ConstSharedPtr observation);
  void process_pending_cups();

  struct PendingCup {
    msg::CupObservation::ConstSharedPtr observation;
    TimePoint received;
  };

  // Subscription and timer share the node's mutually exclusive callback group.
  std::deque<PendingCup> pending_cups_;
  rclcpp::TimerBase::SharedPtr cup_tf_timer_;
  rclcpp::Node & node_;
  const Config & config_;
  mutable std::mutex mutex_;
  Snapshot state_;
  std::map<std::string, CupState> cups_;
  HeadCupTracker head_cup_tracker_;
  rclcpp::Subscription<msg::CupObservation>::SharedPtr cup_subscription_, wrist_subscription_;

  struct PositionSample {
    double position;
    int64_t stamp_ns;
    TimePoint received;
  };

  std::map<std::string, std::deque<PositionSample>> gripper_history_;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};
} // namespace demo_1f
