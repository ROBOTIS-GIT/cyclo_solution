#include "demo_1f/runtime/state/robot_context.hpp"

namespace demo_1f {
RobotContext::RobotContext(rclcpp::Node & node, const Config & config, HeadCupTrackConfig head_tracking)
    : node_(node), config_(config), head_cup_tracker_(head_tracking),
      buffer_(node.get_clock()), listener_(buffer_, &node, false) {
  for (const auto & [side, arm] : config_.arms) {
    (void)side;
    gripper_history_.try_emplace(arm.gripper_joint);
  }
  cup_subscription_ = node.create_subscription<msg::CupObservation>(
      "/perception/cup", rclcpp::QoS(5), [this](msg::CupObservation::ConstSharedPtr message) {
        RCLCPP_INFO_ONCE(node_.get_logger(),
            "Cup input received: topic=%s id=%s source=%s frame=%s",
            cup_subscription_->get_topic_name(),
            message->cup_id.c_str(),
            message->source.c_str(),
            message->header.frame_id.c_str());
        queue_cup(message);
      });
  wrist_subscription_ = node.create_subscription<msg::CupObservation>(
      "/perception/wrist_cup", rclcpp::QoS(5), [this](msg::CupObservation::ConstSharedPtr message) {
        if ((message->source == "wrist_left" || message->source == "wrist_right") &&
            message->observation_type == 2) {
          queue_cup(message);
        }
      });
  cup_tf_timer_ = node.create_wall_timer(std::chrono::milliseconds(10), [this]() {
    process_pending_cups();
  });
  subscription_ = node.create_subscription<sensor_msgs::msg::JointState>(config.joint_states_topic,
      rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
        observe_joints(*message);
      });
}

Snapshot RobotContext::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

void RobotContext::set_task(const std::string & state,
    const std::string & direction,
    const std::string & code,
    const std::string & detail) {
  std::lock_guard<std::mutex> lock(mutex_);
  state_.task_state = state;
  state_.direction = direction;
  state_.error_code = code;
  state_.detail = detail;
}

void RobotContext::set_phase(const std::string & phase, const std::string & completed) {
  std::lock_guard<std::mutex> lock(mutex_);
  state_.phase = phase;
  state_.last_completed_step = completed;
}

void RobotContext::set_execution(const ExecutionResult & result) {
  std::lock_guard<std::mutex> lock(mutex_);
  state_.execution = result;
}

} // namespace demo_1f
