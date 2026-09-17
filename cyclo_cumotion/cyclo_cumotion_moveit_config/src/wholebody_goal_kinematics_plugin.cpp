// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Yeonguk Kim

#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <class_loader/class_loader.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/srv/get_position_fk.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace cyclo_cumotion_moveit_config
{
class WholebodyGoalKinematicsPlugin : public kinematics::KinematicsBase
{
public:
  bool initialize(
    const rclcpp::Node::SharedPtr & node, const moveit::core::RobotModel & robot_model,
    const std::string & group_name, const std::string & base_frame,
    const std::vector<std::string> & /*loader_tip_frames*/, double search_discretization) override
  {
    node_ = node;
    joint_model_group_ = robot_model.getJointModelGroup(group_name);
    arm_groups_[0] = robot_model.getJointModelGroup("arm_l");
    arm_groups_[1] = robot_model.getJointModelGroup("arm_r");
    if (!joint_model_group_ || !arm_groups_[0] || !arm_groups_[1] || group_name != "wholebody") {
      return false;
    }

    const std::vector<std::string> control_tips{
      "wholebody_control_l_link", "wholebody_control_r_link"
    };
    ik_link_names_ = {"end_effector_l_link", "end_effector_r_link"};
    if (!robot_model.hasLinkModel(control_tips[0]) || !robot_model.hasLinkModel(control_tips[1]) ||
      !robot_model.hasLinkModel(ik_link_names_[0]) || !robot_model.hasLinkModel(ik_link_names_[1]))
    {
      return false;
    }

    storeValues(robot_model, group_name, base_frame, control_tips, search_discretization);
    joint_names_ = joint_model_group_->getActiveJointModelNames();
    link_names_ = control_tips;
    const bool is_rviz = std::string(node->get_name()).find("rviz") != std::string::npos;
    if (is_rviz) {
      goal_spheres_client_ =
        node->create_client<moveit_msgs::srv::GetPositionFK>("/cumotion_query_goal_spheres");
    }
    RCLCPP_INFO(node_->get_logger(),
                "Wholebody goal IK base=%s control_tips=[%s, %s] ik_links=[%s, %s] variables=%u",
                base_frame_.c_str(), link_names_[0].c_str(), link_names_[1].c_str(),
                ik_link_names_[0].c_str(), ik_link_names_[1].c_str(),
        joint_model_group_->getVariableCount());
    return joint_names_.size() == 15;
  }

  bool supportsGroup(
    const moveit::core::JointModelGroup * group,
    std::string * error_text = nullptr) const override
  {
    const bool supported = group && group->getName() == "wholebody" &&
      group->getVariableCount() == 15;
    if (!supported && error_text) {
      *error_text = "The plugin is only for the 15-DOF FFW wholebody goal-state group";
    }
    return supported;
  }

  bool getPositionIK(
    const geometry_msgs::msg::Pose & pose, const std::vector<double> & seed,
    std::vector<double> & solution, moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options = {}) const override
  {
    return solve({pose}, seed, getDefaultTimeout(), {}, solution, {}, error_code, options, nullptr);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & pose, const std::vector<double> & seed, double timeout,
    std::vector<double> & solution, moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options = {}) const override
  {
    return solve({pose}, seed, timeout, {}, solution, {}, error_code, options, nullptr);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & pose, const std::vector<double> & seed, double timeout,
    const std::vector<double> & consistency_limits, std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options = {}) const override
  {
    return solve({pose}, seed, timeout, consistency_limits, solution, {}, error_code, options,
        nullptr);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & pose, const std::vector<double> & seed, double timeout,
    std::vector<double> & solution, const IKCallbackFn & callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options = {}) const override
  {
    return solve({pose}, seed, timeout, {}, solution, callback, error_code, options, nullptr);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & pose, const std::vector<double> & seed, double timeout,
    const std::vector<double> & consistency_limits, std::vector<double> & solution,
    const IKCallbackFn & callback, moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options = {}) const override
  {
    return solve({pose}, seed, timeout, consistency_limits, solution, callback, error_code, options,
        nullptr);
  }

  bool searchPositionIK(
    const std::vector<geometry_msgs::msg::Pose> & poses, const std::vector<double> & seed,
    double timeout, const std::vector<double> & consistency_limits, std::vector<double> & solution,
    const IKCallbackFn & callback, moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options = {},
    const moveit::core::RobotState * context_state = nullptr) const override
  {
    return solve(poses, seed, timeout, consistency_limits, solution, callback, error_code, options,
        context_state);
  }

  bool getPositionFK(
    const std::vector<std::string> & links, const std::vector<double> & joint_values,
    std::vector<geometry_msgs::msg::Pose> & poses) const override
  {
    if (!robot_model_ || joint_values.size() != joint_model_group_->getVariableCount()) {
      return false;
    }
    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();
    state.setJointGroupPositions(joint_model_group_, joint_values);
    state.update();
    const Eigen::Isometry3d base_inverse = state.getGlobalLinkTransform(base_frame_).inverse();
    poses.clear();
    for (const auto & link : links) {
      if (!robot_model_->hasLinkModel(link)) {
        return false;
      }
      poses.push_back(tf2::toMsg(base_inverse * state.getGlobalLinkTransform(link)));
    }
    return true;
  }

  const std::vector<std::string> & getJointNames() const override {return joint_names_;}
  const std::vector<std::string> & getLinkNames() const override {return link_names_;}

private:
  bool solve(
    const std::vector<geometry_msgs::msg::Pose> & poses, const std::vector<double> & seed,
    double timeout,
    const std::vector<double> & consistency_limits, std::vector<double> & solution,
    const IKCallbackFn & callback, moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options,
    const moveit::core::RobotState * context_state) const
  {
    (void)options;
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
    if (!robot_model_ || !joint_model_group_ ||
      seed.size() != joint_model_group_->getVariableCount() ||
      poses.empty() || poses.size() > link_names_.size())
    {
      return false;
    }

    moveit::core::RobotState state =
      context_state ? *context_state : moveit::core::RobotState(robot_model_);
    if (!context_state) {
      state.setToDefaultValues();
    }
    state.setJointGroupPositions(joint_model_group_, seed);
    state.update();

    std::vector<Eigen::Isometry3d> targets;
    targets.reserve(poses.size());
    for (const auto & pose : poses) {
      Eigen::Isometry3d target;
      tf2::fromMsg(pose, target);
      targets.push_back(target);
    }

    const auto started = std::chrono::steady_clock::now();
    const double available_time = timeout > 0.0 ? timeout : 0.25;
    const auto lift_it = std::find(joint_names_.begin(), joint_names_.end(), "lift_joint");
    if (lift_it == joint_names_.end()) {
      return false;
    }
    const std::size_t lift_index = std::distance(joint_names_.begin(), lift_it);
    const double seed_lift = seed[lift_index];
    const auto & lift_bounds = robot_model_->getVariableBounds("lift_joint");

    // Try the current lift first. Only expand into the lift workspace when one
    // of the serial-arm IK solvers cannot reach the requested tool pose.
    std::vector<double> lift_candidates{seed_lift};
    constexpr double lift_step = 0.02;
    for (double offset = lift_step;
      seed_lift + offset <= lift_bounds.max_position_ ||
      seed_lift - offset >= lift_bounds.min_position_;
      offset += lift_step)
    {
      if (seed_lift + offset <= lift_bounds.max_position_) {
        lift_candidates.push_back(seed_lift + offset);
      }
      if (seed_lift - offset >= lift_bounds.min_position_) {
        lift_candidates.push_back(seed_lift - offset);
      }
    }

    for (const double lift : lift_candidates) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
        available_time)
      {
        break;
      }
      moveit::core::RobotState candidate(state);
      std::vector<double> candidate_values(seed);
      candidate_values[lift_index] = lift;
      candidate.setJointGroupPositions(joint_model_group_, candidate_values);
      candidate.update();

      bool solved = true;
      for (std::size_t tip_index = 0; tip_index < poses.size(); ++tip_index) {
        const double remaining = available_time -
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (remaining <= 0.0 ||
          !candidate.setFromIK(arm_groups_[tip_index], targets[tip_index],
            ik_link_names_[tip_index],
                                 std::min(0.01, remaining)))
        {
          solved = false;
          break;
        }
      }
      if (!solved) {
        continue;
      }

      candidate.copyJointGroupPositions(joint_model_group_, solution);
      if (!consistency_limits.empty() && consistency_limits.size() == solution.size()) {
        bool consistent = true;
        for (std::size_t i = 0; i < solution.size(); ++i) {
          consistent = consistent && std::abs(solution[i] - seed[i]) <= consistency_limits[i];
        }
        if (!consistent) {
          continue;
        }
      }

      error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
      if (callback) {
        callback(poses.front(), solution, error_code);
      }
      if (error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        updateGoalSpheres(solution);
        RCLCPP_DEBUG(node_->get_logger(), "Wholebody goal IK solved with lift %.4f (seed %.4f)",
            lift, seed_lift);
        return true;
      }
    }
    RCLCPP_WARN(node_->get_logger(),
        "Wholebody goal IK found no solution across lift range [%.3f, %.3f]",
                lift_bounds.min_position_, lift_bounds.max_position_);
    return false;
  }

  void updateGoalSpheres(const std::vector<double> & solution) const
  {
    if (!goal_spheres_client_ || !goal_spheres_client_->service_is_ready()) {
      return;
    }
    if (goal_spheres_request_in_flight_->exchange(true)) {
      return;
    }
    auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
    request->robot_state.joint_state.header.frame_id = getGroupName();
    request->robot_state.joint_state.name = joint_names_;
    request->robot_state.joint_state.position = solution;
    try {
      goal_spheres_client_->async_send_request(
          request, [request_in_flight = goal_spheres_request_in_flight_](
          rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedFuture) {
          request_in_flight->store(false);
          });
    } catch (...) {
      goal_spheres_request_in_flight_->store(false);
      throw;
    }
  }

  const moveit::core::JointModelGroup * joint_model_group_{nullptr};
  const moveit::core::JointModelGroup * arm_groups_[2]{nullptr, nullptr};
  std::vector<std::string> joint_names_;
  std::vector<std::string> link_names_;
  std::vector<std::string> ik_link_names_;
  rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr goal_spheres_client_;
  std::shared_ptr<std::atomic_bool> goal_spheres_request_in_flight_{
    std::make_shared<std::atomic_bool>(false)
  };
};
}  // namespace cyclo_cumotion_moveit_config

PLUGINLIB_EXPORT_CLASS(cyclo_cumotion_moveit_config::WholebodyGoalKinematicsPlugin,
  kinematics::KinematicsBase)
