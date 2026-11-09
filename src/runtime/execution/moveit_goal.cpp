#include "action_dispatch.hpp"
#include "demo_1f/runtime/execution/trajectory_executor.hpp"
#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <algorithm>
#include <cmath>
#include <moveit/robot_state/conversions.hpp>
#include <set>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
void TrajectoryExecutor::send_move(const std::shared_ptr<Operation> & op, bool plan_only) {
  const auto & step = plan_only && !op->steps.empty()
                         ? op->steps.at(op->planned_segments.size()) : op->step;
  op->planning_only = plan_only;
  op->joint_targets.clear();
  const bool both_arms = step.type == "dual_joint" || step.type == "dual_joint_gripper";
  const bool move_gripper = step.type == "joint_gripper" || step.type == "dual_joint_gripper";
  const bool coordinated = both_arms || move_gripper;
  const bool joint_group = step.type == "joint_group";
  const auto & arm = config_.arms.at(step.type == "dual_joint" || joint_group ? "left" : step.arm);
  Move::Goal goal;
  auto & request = goal.request;
  request.group_name = joint_group ? step.group : coordinated ? config_.coordinated_group : arm.group;
  request.pipeline_id = coordinated ? config_.coordinated_pipeline : config_.pipeline;
  request.planner_id =
      coordinated ? config_.coordinated_planner : (step.type == "linear" ? "LIN" : "PTP");
  if (joint_group || step.position_only || step.orientation_slack > 0) {
    // Pilz PTP expects a complete pose. OMPL can sample unconstrained orientations.
    request.pipeline_id = "ompl";
    request.planner_id = "RRTConnectkConfigDefault";
  }
  // OMPL ParallelPlan compares candidates inside one allowed_planning_time budget.
  // sg2_moveit.launch.py enables path-length scoring and hybridization, so this
  // does not stop at the first successful planner. Pilz LIN/PTP remains single-shot.
  request.num_planning_attempts =
      request.pipeline_id == "ompl" ? config_.ompl_planning_attempts : 1;
  request.allowed_planning_time =
      step.planning_time > 0 ? step.planning_time : config_.planning_time;
  request.max_velocity_scaling_factor =
      step.velocity_scale > 0 ? step.velocity_scale : config_.velocity_scale;
  request.max_acceleration_scaling_factor =
      step.acceleration_scale > 0 ? step.acceleration_scale : config_.acceleration_scale;
  request.start_state.is_diff = true;
  if (plan_only && !op->steps.empty()) {
    request.start_state = op->planned_start;
    request.start_state.is_diff = false;
  }
  goal.planning_options.plan_only = plan_only;
  goal.planning_options.replan = false;
  goal.planning_options.planning_scene_diff.is_diff = true;
  goal.planning_options.planning_scene_diff.robot_state.is_diff = true;
  moveit_msgs::msg::Constraints constraints;
  std::set<std::string> held_joints;
  if (coordinated) {
    const auto snapshot = context_.snapshot();
    const auto current_position = [&](const std::string & name) {
      const auto & joints = op->planned_start.joint_state;
      const auto it = std::find(joints.name.begin(), joints.name.end(), name);
      if (plan_only && it != joints.name.end()) {
        return joints.position.at(it - joints.name.begin());
      }
      return snapshot.joints.at(name).position;
    };
    for (const auto & [side, settings] : config_.arms) {
      for (size_t i = 0; i < settings.joints.size(); ++i) {
        const auto & name = settings.joints[i];
        double target = current_position(name);
        if (both_arms) {
          target = (side == "left" ? step.left_target : step.right_target)[i];
        } else if (side == step.arm) {
          target = step.target[i];
        } else {
          held_joints.insert(name);
        }
        op->joint_targets[name] = target;
      }
      op->joint_targets[settings.gripper_joint] = move_gripper && side == step.arm
                                                      ? step.gripper_target[0]
                                                      : current_position(settings.gripper_joint);
      if (!move_gripper || side != step.arm) {
        held_joints.insert(settings.gripper_joint);
      }
    }
  } else if (joint_group) {
    for (size_t i = 0; i < step.joint_names.size(); ++i) {
      op->joint_targets[step.joint_names[i]] = step.target[i];
    }
  } else if (step.type == "joint") {
    for (size_t i = 0; i < arm.joints.size(); ++i) {
      op->joint_targets[arm.joints[i]] = step.target[i];
    }
  } else {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = step.frame;
    pose.pose.position.x = step.target[0];
    pose.pose.position.y = step.target[1];
    pose.pose.position.z = step.target[2];
    pose.pose.orientation.x = step.target[3];
    pose.pose.orientation.y = step.target[4];
    pose.pose.orientation.z = step.target[5];
    pose.pose.orientation.w = step.target[6];
    const auto transformed = context_.transform_pose(pose);
    if (!transformed) {
      throw std::runtime_error("Target frame transform unavailable/stale");
    }
    op->pose_target = *transformed;
    moveit_msgs::msg::PositionConstraint position;
    position.header.frame_id = config_.planning_frame;
    position.link_name = arm.tcp;
    position.weight = 1.0;
    shape_msgs::msg::SolidPrimitive region;
    region.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    // Reserve tracking margin: OMPL may choose any point inside the goal region.
    region.dimensions = {config_.pose_position_tolerance * (step.position_only ? .5 : 1.0)};
    position.constraint_region.primitives.push_back(region);
    position.constraint_region.primitive_poses.push_back(transformed->pose);
    constraints.position_constraints.push_back(position);
    if (step.tcp_path_margin > 0) {
      const auto start = context_.pose_of(arm.tcp);
      if (!start) {
        throw std::runtime_error("Missing TCP pose for correction corridor");
      }
      auto corridor = position;
      auto & shape = corridor.constraint_region.primitives.front();
      shape.type = shape_msgs::msg::SolidPrimitive::BOX;
      const auto & a = start->pose.position;
      const auto & b = transformed->pose.position;
      double bottom = std::min(a.z, b.z) - step.tcp_path_margin;
      const double top = std::max(a.z, b.z) + step.tcp_path_margin;
      if (step.tcp_path_min_z) {
        if (!std::isfinite(*step.tcp_path_min_z) || *step.tcp_path_min_z > std::min(a.z, b.z)) {
          throw std::runtime_error("TCP corridor floor excludes start or goal");
        }
        bottom = std::max(bottom, *step.tcp_path_min_z);
      }
      shape.dimensions = {std::abs(a.x - b.x) + 2 * step.tcp_path_margin,
          std::abs(a.y - b.y) + 2 * step.tcp_path_margin,
          top - bottom};
      auto & center = corridor.constraint_region.primitive_poses.front();
      center.position.x = (a.x + b.x) * .5;
      center.position.y = (a.y + b.y) * .5;
      center.position.z = (bottom + top) * .5;
      center.orientation = geometry_msgs::msg::Quaternion();
      center.orientation.w = 1;
      request.path_constraints.position_constraints.push_back(corridor);
    }
    if (step.tcp_pitch_min > 0) {
      constraints.position_constraints.push_back(tcp_min_pitch_constraint(arm.tcp,
          config_.planning_frame,
          transformed->pose.position,
          region.dimensions[0],
          step.tcp_pitch_min + M_PI / 180));
    }
    if (step.tcp_roll_limit > 0 || step.tcp_pitch_limit > 0) {
      // Goal-only tilt bounds: allow free transit, reserve one degree for tracking error.
      // The position and orientation constraints form ONE goal.
      const double roll_limit = step.tcp_roll_limit > 0 ? step.tcp_roll_limit - M_PI / 180 : M_PI;
      const double pitch_limit = step.tcp_pitch_limit > 0 ? step.tcp_pitch_limit - M_PI / 180 : 0;
      auto tilt = tcp_roll_constraint(arm.tcp, config_.planning_frame, roll_limit, pitch_limit);
      constraints.orientation_constraints.push_back(tilt);
      if (step.tcp_pitch_downward_only) {
        constraints.orientation_constraints.push_back(
            tcp_downward_constraint(arm.tcp, config_.planning_frame));
      }
    }
    if (step.tcp_yaw_inward_only) {
      constraints.orientation_constraints.push_back(
          tcp_inward_yaw_constraint(arm.tcp, config_.planning_frame, step.arm));
    }
    if (step.tcp_yaw_limit > 0) {
      const auto yaw = tcp_yaw(transformed->pose.orientation);
      if (!yaw || step.tcp_yaw_limit <= M_PI / 180 || step.tcp_yaw_limit > M_PI) {
        throw std::invalid_argument("Invalid TCP heading goal");
      }
      constraints.orientation_constraints.push_back(tcp_yaw_constraint(
          arm.tcp, config_.planning_frame, *yaw, step.tcp_yaw_limit - M_PI / 180));
    }
    if (!step.position_only) {
      moveit_msgs::msg::OrientationConstraint orientation;
      orientation.header.frame_id = config_.planning_frame;
      orientation.link_name = arm.tcp;
      orientation.orientation = transformed->pose.orientation;
      const double tolerance = step.orientation_slack > 0 ? step.orientation_slack - M_PI / 180
                                                          : config_.pose_orientation_tolerance;
      if (step.orientation_slack > 0) {
        orientation.parameterization = moveit_msgs::msg::OrientationConstraint::ROTATION_VECTOR;
      }
      orientation.absolute_x_axis_tolerance = tolerance;
      orientation.absolute_y_axis_tolerance = tolerance;
      orientation.absolute_z_axis_tolerance = tolerance;
      orientation.weight = 1.0;
      constraints.orientation_constraints.push_back(orientation);
    }
  }
  auto planning_step = step;
  if (step.planning_tolerance > 0) {
    planning_step.tolerance = step.planning_tolerance;
  }
  for (const auto & [name, target] : op->joint_targets) {
    moveit_msgs::msg::JointConstraint joint;
    joint.joint_name = name;
    joint.position = target;
    joint.tolerance_above = goal_tolerance(name, planning_step);
    joint.tolerance_below = goal_tolerance(name, planning_step);
    joint.weight = 1.0;
    constraints.joint_constraints.push_back(joint);
    if (held_joints.count(name)) {
      // A held joint needs a path constraint, not just the same endpoint.
      joint.tolerance_above = joint.tolerance_below = std::min(planning_step.tolerance, 0.001);
      request.path_constraints.joint_constraints.push_back(joint);
    }
  }
  request.goal_constraints.push_back(constraints);
  if (request.pipeline_id == "ompl") {
    RCLCPP_INFO(node_.get_logger(),
        "Planning %s: up to %d OMPL candidates, joint-space path length, %.2f s budget",
        step.name.c_str(), request.num_planning_attempts, request.allowed_planning_time);
  }
  dispatch<Move>(
      move_client_,
      goal,
      op,
      [](const Move::Result & result) {
        return result.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS
                   ? std::string{}
                   : "MoveIt error " + std::to_string(result.error_code.val);
      },
      plan_only ? std::function<void(const Move::Result &)>([this, op](
                                                                const Move::Result & result) {
        const auto & trajectory = result.planned_trajectory.joint_trajectory;
        if (trajectory.points.empty() || trajectory.joint_names.empty() ||
            trajectory.points.back().positions.size() != trajectory.joint_names.size() ||
            !std::all_of(trajectory.points.back().positions.begin(),
                trajectory.points.back().positions.end(),
                [](double value) {
                  return std::isfinite(value);
                })) {
          throw std::runtime_error("MoveIt returned an empty plan");
        }
        if (!result.executed_trajectory.joint_trajectory.points.empty() ||
            !result.executed_trajectory.multi_dof_joint_trajectory.points.empty()) {
          throw std::runtime_error("Unexpected execution reported for a plan-only request");
        }
        if (op->steps.empty()) {
          execute_planned_pose(op, result.planned_trajectory);
          return;
        }
        op->planned_segments.push_back(result.planned_trajectory);
        moveit::core::RobotState state(robot_model_);
        moveit::core::robotStateMsgToRobotState(op->planned_start, state);
        state.setVariablePositions(trajectory.joint_names, trajectory.points.back().positions);
        state.update();
        moveit::core::robotStateToRobotStateMsg(state, op->planned_start);
        if (op->planned_segments.size() < op->steps.size()) {
          send_move(op, true);
        } else {
          request_scene(op, true);
        }
      })
                : std::function<void(const Move::Result &)>{});
}
void TrajectoryExecutor::execute_planned_pose(const std::shared_ptr<Operation> & op,
    const moveit_msgs::msg::RobotTrajectory & planned) {
  if (op->cancel_requested || op->result.state != ExecutionState::Running) return;
  std::string reason;
  if (!context_.ready(reason)) throw std::runtime_error(reason);
  const auto snapshot = context_.snapshot();
  const auto & trajectory = planned.joint_trajectory;
  const auto & controlled = config_.arms.at(op->step.arm).joints;
  if (!planned.multi_dof_joint_trajectory.points.empty() ||
      trajectory.joint_names.size() != controlled.size()) {
    throw std::runtime_error("Unexpected joint group in correction plan");
  }
  std::set<std::string> names;
  for (size_t i = 0; i < trajectory.joint_names.size(); ++i) {
    const auto & name = trajectory.joint_names[i];
    if (!names.insert(name).second ||
        std::find(controlled.begin(), controlled.end(), name) == controlled.end()) {
      throw std::runtime_error("Unexpected joint in correction plan: " + name);
    }
    for (const auto & point : trajectory.points) {
      if (point.positions.size() != trajectory.joint_names.size() || !std::isfinite(point.positions[i])) {
        throw std::runtime_error("Invalid correction trajectory positions");
      }
    }
    const auto joint = snapshot.joints.find(name);
    if (joint == snapshot.joints.end() ||
        !context_.joint_ready_for_motion(name, joint->second, reason) ||
        std::abs(joint->second.position - trajectory.points.front().positions[i]) >
            config_.continuous_start_tolerance) {
      throw std::runtime_error("Robot moved before correction execution: " + name + "; " + reason);
    }
    op->joint_targets[name] = trajectory.points.back().positions[i];
  }
  const double duration = rclcpp::Duration(trajectory.points.back().time_from_start).seconds();
  if (!std::isfinite(duration) || duration <= 0 ||
      duration + config_.settle_time >= op->step.timeout - elapsed(op->started)) {
    throw std::runtime_error("Correction plan duration exceeds remaining motion timeout");
  }
  Execute::Goal goal;
  goal.trajectory = planned; // Execute exactly the validated MoveIt path; no extra smoothing.
  planned_trajectory_publisher_->publish(trajectory);
  RCLCPP_INFO(node_.get_logger(), "Executing planned %s, duration %.3f s", op->step.name.c_str(), duration);
  dispatch<Execute>(execute_client_, goal, op, [](const Execute::Result & result) {
    return result.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS
               ? std::string{} : "MoveIt execution error " + std::to_string(result.error_code.val);
  });
}
} // namespace demo_1f
