#pragma once
#include "demo_1f/action/approach_cup.hpp"
#include "demo_1f/action/handover.hpp"
#include "demo_1f/action/run_demo.hpp"
#include <ffw_vision_msgs/action/pick_dynamixel.hpp>
#include "demo_1f/motions/common/cup_motion.hpp"
#include "demo_1f/motions/handover/handover_motion.hpp"
#include "demo_1f/motions/initial/initial_pose_motion.hpp"
#include "demo_1f/msg/demo_status.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace demo_1f {
// Modules are owned by the composition root and must outlive TaskManager.
struct CupMotionRegistration {
  std::string action_name;
  CupMotion & motion;
};

class TaskManager {
public:
  TaskManager(rclcpp::Node & node,
      RobotContext & context,
      HandoverMotion & motion,
      std::vector<CupMotionRegistration> cup_motions = {}, InitialPoseMotion * initial_motion = nullptr);
  void request_shutdown();
  bool can_shutdown() const;
  bool demo_faulted() const { return demo_faulted_; }

private:
  using DemoAction = demo_1f::action::RunDemo;
  using DemoHandle = rclcpp_action::ServerGoalHandle<DemoAction>;
  using PickAction = ffw_vision_msgs::action::PickDynamixel;
  using PickHandle = rclcpp_action::ClientGoalHandle<PickAction>;
  void setup_demo_server();
  void tick_demo();
  void start_demo_stage();
  void cancel_demo_motion();
  void finish_demo(bool success, const std::string & code, const std::string & detail,
      bool resumable = false);
  void fail_demo_stage(const std::string & code, const std::string & detail, bool retryable);
  CupMotion * demo_cup_module(const std::string & action);
  bool demo_busy() const;
  rclcpp_action::Server<DemoAction>::SharedPtr demo_server_;
  rclcpp_action::Client<PickAction>::SharedPtr pick_client_;
  std::shared_ptr<DemoHandle> demo_goal_;
  PickHandle::SharedPtr pick_handle_;
  PickAction::Result::SharedPtr pick_result_;
  bool pick_pending_{false}, pick_cancel_sent_{false}, demo_stage_active_{false};
  bool demo_retry_wait_{false}, demo_resumable_{false}, demo_pick_only_{false};
  bool demo_initial_only_{false};
  bool demo_faulted_{false}, demo_cancel_requested_{false};
  std::string demo_stage_, demo_completed_stage_;
  std::string demo_cup_id_, demo_detail_, pick_phase_;
  int demo_retries_{0}, demo_max_retries_{2};
  double demo_retry_delay_{1.0}, demo_pick_timeout_{180.0};
  TimePoint demo_deadline_{}, demo_stage_started_{};

  using Action = demo_1f::action::Handover;
  using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
  using ApproachAction = demo_1f::action::ApproachCup;
  using ApproachGoalHandle = rclcpp_action::ServerGoalHandle<ApproachAction>;
  rclcpp_action::Server<Action>::SharedPtr make_handover_server(
      const std::string & action_name, bool prepare_only);
  void setup_cup_servers();
  rclcpp_action::Server<ApproachAction>::SharedPtr make_cup_server(
      const std::string & name, CupMotion & module);
  void tick_cup_action();
  void publish_cup_target();
  void tick();
  void publish_status();
  rclcpp::Node & node_;
  RobotContext & context_;
  HandoverMotion & motion_;
  InitialPoseMotion * initial_motion_;
  std::vector<CupMotionRegistration> cup_motions_;
  CupMotion * active_cup_motion_{nullptr};
  bool cup_motion_blocked() const;
  std::vector<rclcpp_action::Server<ApproachAction>::SharedPtr> cup_servers_;
  std::shared_ptr<ApproachGoalHandle> cup_goal_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr approach_target_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr cup_estimate_publisher_;
  rclcpp_action::Server<Action>::SharedPtr server_, prepare_server_;
  rclcpp::Publisher<demo_1f::msg::DemoStatus>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_service_, reset_demo_service_;
  // Accepted request retained until its final action result is sent.
  std::shared_ptr<GoalHandle> active_goal_handle_;
  // A goal was accepted; its accepted callback has not started yet.
  bool goal_start_pending_{false};
  bool shutdown_requested_{false};
  TimePoint last_publish_{Clock::now()};
};
} // namespace demo_1f
