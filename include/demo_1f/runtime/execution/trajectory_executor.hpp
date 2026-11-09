#pragma once
#include "demo_1f/core/executor_port.hpp"
#include "demo_1f/runtime/execution/trajectory_smoother.hpp"
#include "demo_1f/runtime/state/robot_context.hpp"
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <functional>
#include <future>
#include <moveit_msgs/action/execute_trajectory.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace demo_1f {
class TrajectoryExecutor final : public ExecutorPort {
public:
  TrajectoryExecutor(rclcpp::Node & node, const Config & config, RobotContext & context);
  bool ready(const MotionSequence & sequence, std::string & reason) const override;
  bool start(const Step & step) override;
  bool start_sequence(const std::vector<Step> & steps) override;
  void tick() override;
  void cancel(
      const std::string & code = "CANCELLED", const std::string & reason = "Cancelled") override;
  ExecutionResult result() const override;

private:
  using Move = moveit_msgs::action::MoveGroup;
  using Joint = control_msgs::action::FollowJointTrajectory;
  using Execute = moveit_msgs::action::ExecuteTrajectory;

  struct Operation {
    Step step;
    ExecutionResult result{ExecutionState::Running, {}, {}};
    TimePoint started{Clock::now()};
    TimePoint cancelling;
    std::optional<TimePoint> gripper_sent;
    std::optional<TimePoint> stable_since;
    std::function<void()> cancel_action;
    bool planning_only{false};
    bool action_terminal{false};
    bool action_success{false};
    bool cancel_requested{false};
    bool topic_command{false};
    bool hold_sent{false};
    std::string cancel_code;
    std::string cancel_reason;
    std::string completion_wait_reason;
    geometry_msgs::msg::PoseStamped pose_target;
    std::map<std::string, double> joint_targets;
    std::vector<Step> steps;
    std::vector<moveit_msgs::msg::RobotTrajectory> planned_segments;
    moveit_msgs::msg::RobotState planned_start;
    SmoothPath path;
    size_t progress_point{0};
    bool executing_path{false};
    std::future<SmoothPath> preparation;
    TimePoint preparation_started;
    std::map<std::string, double> preparation_positions;
  };

  void send_move(const std::shared_ptr<Operation> & op, bool plan_only = false);
  void execute_planned_pose(const std::shared_ptr<Operation> & op,
      const moveit_msgs::msg::RobotTrajectory & trajectory);
  void request_scene(const std::shared_ptr<Operation> & op, bool prepare_execution);
  void prepare_path(
      const std::shared_ptr<Operation> & op, const moveit_msgs::msg::PlanningScene & scene);
  void update_path_phase(const std::shared_ptr<Operation> & op);
  void load_robot_model();
  bool joint_group_ready(const Step & step, std::string & reason) const;
  void send_gripper(const std::shared_ptr<Operation> & op);
  template <class Action>
  void dispatch(const typename rclcpp_action::Client<Action>::SharedPtr & client,
      const typename Action::Goal & goal,
      const std::shared_ptr<Operation> & op,
      std::function<std::string(const typename Action::Result &)> error,
      std::function<void(const typename Action::Result &)> on_success = {});
  bool settled(const std::shared_ptr<Operation> & op, bool check_target);
  double goal_tolerance(const std::string & joint, const Step & step) const;
  void publish_hold(const std::shared_ptr<Operation> & op);
  void publish_state();
  void set_operation(const std::shared_ptr<Operation> & op);
  rclcpp::Node & node_;
  const Config & config_;
  RobotContext & context_;
  rclcpp_action::Client<Move>::SharedPtr move_client_;
  rclcpp_action::Client<Execute>::SharedPtr execute_client_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client_;
  moveit::core::RobotModelPtr robot_model_;
  planning_scene::PlanningScenePtr scene_template_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr planned_trajectory_publisher_;
  std::map<std::string, rclcpp_action::Client<Joint>::SharedPtr> joint_clients_;
  std::map<std::string, rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr>
      publishers_;
  std::shared_ptr<Operation> operation_;
  std::vector<std::shared_ptr<Operation>> retired_operations_;
};
} // namespace demo_1f
