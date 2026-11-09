#include "demo_1f/runtime/execution/trajectory_executor.hpp"
#include "action_dispatch.hpp"
#include "demo_1f/runtime/config.hpp"
#include <algorithm>
#include <cmath>

namespace demo_1f {
TrajectoryExecutor::TrajectoryExecutor(
    rclcpp::Node & node, const Config & config, RobotContext & context)
    : node_(node), config_(config), context_(context) {
  move_client_ = rclcpp_action::create_client<Move>(&node, config.move_group_action);
  execute_client_ = rclcpp_action::create_client<Execute>(&node, "/execute_trajectory");
  scene_client_ = node.create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  planned_trajectory_publisher_ = node.create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "~/planned_trajectory", rclcpp::QoS(1).reliable());
  if (config_.smooth_motion) {
    load_robot_model();
  }
  for (const auto & [name, arm] : config.arms) {
    joint_clients_[name] = rclcpp_action::create_client<Joint>(&node, arm.trajectory_action);
    publishers_[name] = node.create_publisher<trajectory_msgs::msg::JointTrajectory>(
        arm.trajectory_topic, rclcpp::QoS(1).reliable());
  }
}

bool TrajectoryExecutor::joint_group_ready(const Step & step, std::string & reason) const {
  if (step.type != "joint_group") return true;
  if (robot_model_) {
    const auto * group = robot_model_->getJointModelGroup(step.group);
    if (!group || group->getVariableCount() != step.joint_names.size()) {
      reason = "Joint group does not match robot model: " + step.group;
      return false;
    }
    const auto & names = group->getVariableNames();
    for (const auto & name : step.joint_names) {
      if (std::find(names.begin(), names.end(), name) == names.end()) {
        reason = "Joint is not in planning group: " + name;
        return false;
      }
    }
  }
  const auto snapshot = context_.snapshot();
  for (const auto & name : step.joint_names) {
    const auto it = snapshot.joints.find(name);
    if (it == snapshot.joints.end()) { reason = "Missing joint feedback: " + name; return false; }
    if (!context_.joint_stationary(name, it->second, reason)) return false;
  }
  return true;
}

bool TrajectoryExecutor::ready(const MotionSequence & sequence, std::string & reason) const {
  const auto state = result().state;
  if (state == ExecutionState::Running || state == ExecutionState::Cancelling ||
      state == ExecutionState::StopUnconfirmed) {
    reason = "Executor is busy or stop is unconfirmed";
    return false;
  }
  const auto snapshot = context_.snapshot();
  for (const auto & [name, arm] : config_.arms) {
    auto joints = arm.joints;
    joints.push_back(arm.gripper_joint);
    for (const auto & joint : joints) {
      const auto it = snapshot.joints.find(joint);
      if (it == snapshot.joints.end()) {
        reason = "Missing joint feedback before starting: " + joint;
        return false;
      }
      if (!context_.joint_ready_for_motion(
              joint, it->second, reason, state == ExecutionState::Succeeded)) {
        return false;
      }
    }
  }
  if (config_.smooth_motion && (!robot_model_ || !scene_client_->service_is_ready() ||
                                   !execute_client_->action_server_is_ready())) {
    reason = "Smoothed motion needs a robot model, planning scene and trajectory execution server";
    return false;
  }
  for (const auto & step : sequence.steps) {
    if (!joint_group_ready(step, reason)) return false;
    if ((step.type == "joint" || step.type == "pose" || step.type == "linear" ||
            step.type == "dual_joint" || step.type == "joint_gripper" ||
            step.type == "dual_joint_gripper" || step.type == "joint_group") &&
        !move_client_->action_server_is_ready()) {
      reason = "MoveIt action server is unavailable";
      return false;
    }
    if (step.type == "pose" && step.tcp_path_margin > 0 &&
        !execute_client_->action_server_is_ready()) {
      reason = "Trajectory execution server is unavailable for planned correction";
      return false;
    }
    if (step.type == "gripper") {
      bool available = config_.gripper_transport == "action"
                           ? joint_clients_.at(step.arm)->action_server_is_ready()
                           : publishers_.at(step.arm)->get_subscription_count() > 0;
      if (!available) {
        reason = "Gripper trajectory endpoint is unavailable: " + step.arm;
        return false;
      }
    }
  }
  reason.clear();
  return true;
}

ExecutionResult TrajectoryExecutor::result() const {
  return operation_ ? operation_->result : ExecutionResult{};
}

void TrajectoryExecutor::publish_state() {
  context_.set_execution(result());
}

void TrajectoryExecutor::set_operation(const std::shared_ptr<Operation> & op) {
  if (operation_ && operation_->preparation.valid() &&
      operation_->preparation.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    retired_operations_.push_back(operation_);
  }
  operation_ = op;
}

bool TrajectoryExecutor::start(const Step & step) {
  if (config_.smooth_motion && (step.type == "dual_joint" || step.type == "dual_joint_gripper" ||
                                   step.type == "joint_gripper")) {
    return start_sequence({step});
  }
  const auto previous = result().state;
  if (previous == ExecutionState::Running || previous == ExecutionState::Cancelling ||
      previous == ExecutionState::StopUnconfirmed) {
    return false;
  }
  MotionSequence single_step_sequence;
  single_step_sequence.steps = {step};
  const auto validation = validate_motion_sequence(single_step_sequence, config_);
  auto op = std::make_shared<Operation>();
  op->step = step;
  set_operation(op);
  std::string reason;
  if (!validation.empty() || !context_.ready(reason) || !joint_group_ready(step, reason)) {
    op->result = {
        ExecutionState::Failed, "INVALID_COMMAND", validation.empty() ? reason : validation};
    publish_state();
    return false;
  }
  try {
    if (step.type == "joint" || step.type == "pose" || step.type == "linear" ||
        step.type == "dual_joint" || step.type == "joint_gripper" ||
        step.type == "dual_joint_gripper" || step.type == "joint_group") {
      // Corridor-constrained corrections are planned first, then executed explicitly.
      send_move(op, step.type == "pose" && step.tcp_path_margin > 0);
    } else if (step.type == "gripper") {
      if (config_.smooth_motion) {
        request_scene(op, false);
      } else {
        send_gripper(op);
      }
    }
    // wait is intentionally non-blocking; tick() completes it.
  } catch (const std::exception & error) {
    op->result = {ExecutionState::Failed, "SEND_FAILED", error.what()};
  }
  publish_state();
  return op->result.state == ExecutionState::Running;
}

void TrajectoryExecutor::cancel(const std::string & code, const std::string & reason) {
  const auto op = operation_;
  if (!op || op->result.state != ExecutionState::Running) {
    return;
  }
  op->cancel_requested = true;
  op->cancel_code = code;
  op->cancel_reason = reason;
  op->cancelling = Clock::now();
  op->stable_since.reset();
  op->result = {ExecutionState::Cancelling, code, reason};
  if (op->step.type == "wait") {
    op->result.state = code == "CANCELLED" ? ExecutionState::Cancelled : ExecutionState::Failed;
  } else if (op->topic_command) {
    publish_hold(op);
  } else if (op->cancel_action && !op->action_terminal) {
    try {
      op->cancel_action();
    } catch (const std::exception & e) {
      op->cancel_reason += "; " + std::string(e.what());
    }
  }
  publish_state();
}

void TrajectoryExecutor::tick() {
  for (auto it = retired_operations_.begin(); it != retired_operations_.end();) {
    if ((*it)->preparation.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      try {
        (*it)->preparation.get();
      } catch (const std::exception &) {
      }
      it = retired_operations_.erase(it);
    } else {
      ++it;
    }
  }
  const auto op = operation_;
  if (!op) {
    return;
  }
  if (op->preparation.valid() &&
      op->preparation.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    try {
      auto path = op->preparation.get();
      RCLCPP_INFO(node_.get_logger(),
          "Prepared %s in %.3f s",
          op->step.name.c_str(),
          elapsed(op->preparation_started));
      if (!op->cancel_requested && op->result.state == ExecutionState::Running) {
        std::string reason;
        if (!context_.ready(reason)) {
          throw std::runtime_error(reason);
        }
        const auto snapshot = context_.snapshot();
        const auto & trajectory = path.trajectory.joint_trajectory;
        for (const auto & [name, planned] : op->preparation_positions) {
          const auto & measured = snapshot.joints.at(name);
          if (!context_.joint_ready_for_motion(name, measured, reason, true) ||
              std::abs(measured.position - planned) > config_.continuous_start_tolerance) {
            throw std::runtime_error(
                "Robot moved during trajectory preparation: " + name + "; " + reason);
          }
        }
        const double duration =
            rclcpp::Duration(trajectory.points.back().time_from_start).seconds();
        if (duration + config_.settle_time + config_.gripper_stop_window >=
            op->step.timeout - elapsed(op->started)) {
          throw std::runtime_error(
              "Motion timeout is too short for the prepared path and stop verification");
        }
        for (size_t joint = 0; joint < trajectory.joint_names.size(); ++joint) {
          const auto & name = trajectory.joint_names[joint];
          const auto & measured = snapshot.joints.at(name);
          if (!context_.joint_ready_for_motion(name, measured, reason, true) ||
              std::abs(measured.position - trajectory.points.front().positions[joint]) >
                  config_.continuous_start_tolerance) {
            throw std::runtime_error("Robot moved during planning: " + name + "; " + reason);
          }
        }
        op->path = std::move(path);
        if (op->step.type == "gripper") {
          send_gripper(op);
        } else {
          Execute::Goal goal;
          goal.trajectory = op->path.trajectory;
          if (!config_.require_passive_gripper_stationary) {
            // Planning still checks the measured finger geometry. Do not send a
            // position-hold goal to a current-limited gripper while moving arms.
            auto & command = goal.trajectory.joint_trajectory;
            for (size_t i = command.joint_names.size(); i-- > 0;) {
              const auto & name = command.joint_names[i];
              if (!context_.is_gripper_joint(name)) continue;
              const bool commanded = std::any_of(op->steps.begin(), op->steps.end(),
                  [&](const Step & step) {
                    return (step.type == "joint_gripper" || step.type == "dual_joint_gripper") &&
                           config_.arms.at(step.arm).gripper_joint == name;
                  });
              if (commanded) continue;
              command.joint_names.erase(command.joint_names.begin() + i);
              for (auto & point : command.points) {
                for (auto * values : {&point.positions, &point.velocities,
                         &point.accelerations, &point.effort}) {
                  if (!values->empty()) values->erase(values->begin() + i);
                }
              }
            }
          }
          planned_trajectory_publisher_->publish(goal.trajectory.joint_trajectory);
          dispatch<Execute>(execute_client_, goal, op, [](const Execute::Result & result) {
            return result.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS
                       ? std::string{}
                       : "MoveIt execution error " + std::to_string(result.error_code.val);
          });
          op->executing_path = true;
          RCLCPP_INFO(node_.get_logger(),
              "Executing %zu continuous segment(s), duration %.3f s",
              op->steps.size(),
              rclcpp::Duration(goal.trajectory.joint_trajectory.points.back().time_from_start)
                  .seconds());
        }
      }
    } catch (const std::exception & error) {
      if (!op->cancel_requested && op->result.state == ExecutionState::Running) {
        op->result = {ExecutionState::Failed, "PATH_PREPARATION_FAILED", error.what()};
      }
    }
  }
  update_path_phase(op);
  if (op->result.state == ExecutionState::Running) {
    if (elapsed(op->started) >= op->step.timeout) {
      cancel("TIMEOUT",
          "Step timed out: " + op->step.name + "; " + op->completion_wait_reason +
              "; action_terminal=" + (op->action_terminal ? "true" : "false") +
              ", action_success=" + (op->action_success ? "true" : "false"));
    } else if (op->step.type == "wait") {
      if (elapsed(op->started) >= op->step.duration) {
        op->result.state = ExecutionState::Succeeded;
      }
    } else {
      const bool finished = op->topic_command ? elapsed(op->started) >= op->step.duration
                                              : op->action_terminal && op->action_success;
      const bool wait_only = op->step.type == "gripper" && op->step.gripper_wait_only;
      const bool duration_elapsed = op->gripper_sent &&
          elapsed(*op->gripper_sent) >= op->step.duration;
      if (finished && (wait_only ? duration_elapsed : settled(op, true))) {
        op->result.state = ExecutionState::Succeeded;
      } else if (finished && wait_only) {
        op->completion_wait_reason = "Waiting for gripper command duration; grasp not verified";
      } else if (!finished) {
        op->completion_wait_reason = op->topic_command ? "Waiting for commanded duration"
                                                       : "Waiting for controller/action result";
      }
    }
  }
  if (op->result.state == ExecutionState::Cancelling) {
    if (op->topic_command && !op->hold_sent) {
      publish_hold(op);
    }
    const bool stopped_request =
        op->topic_command ? op->hold_sent && elapsed(op->cancelling) >= 0.2 : op->action_terminal;
    if (stopped_request && settled(op, false)) {
      op->result = {
          op->cancel_code == "CANCELLED" ? ExecutionState::Cancelled : ExecutionState::Failed,
          op->cancel_code,
          op->cancel_reason};
    } else if (elapsed(op->cancelling) >= config_.cancel_timeout) {
      // Latched until restart: a late acceptance/result must not unlock a new motion.
      op->result = {ExecutionState::StopUnconfirmed,
          "STOP_UNCONFIRMED",
          op->cancel_reason + "; stop not confirmed"};
    }
  }
  publish_state();
}
} // namespace demo_1f
