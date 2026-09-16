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

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.hpp>
#include <moveit_msgs/srv/get_position_fk.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace cyclo_cumotion_moveit_config
{
class GoalStateKDLKinematicsPlugin : public kdl_kinematics_plugin::KDLKinematicsPlugin
{
public:
  bool initialize(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModel& robot_model,
                  const std::string& group_name, const std::string& base_frame,
                  const std::vector<std::string>& tip_frames, double search_discretization) override
  {
    if (!KDLKinematicsPlugin::initialize(
            node, robot_model, group_name, base_frame, tip_frames, search_discretization))
      return false;
    group_name_ = group_name;
    const bool is_rviz = std::string(node->get_name()).find("rviz") != std::string::npos;
    if (is_rviz)
      goal_spheres_client_ =
          node->create_client<moveit_msgs::srv::GetPositionFK>("/cumotion_query_goal_spheres");
    return true;
  }

  bool getPositionIK(const geometry_msgs::msg::Pose& pose, const std::vector<double>& seed,
                     std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                     const kinematics::KinematicsQueryOptions& options = {}) const override
  {
    const bool solved = KDLKinematicsPlugin::getPositionIK(pose, seed, solution, error_code, options);
    if (solved)
      updateGoalSpheres(solution);
    return solved;
  }

  bool searchPositionIK(const geometry_msgs::msg::Pose& pose, const std::vector<double>& seed, double timeout,
                        std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options = {}) const override
  {
    const bool solved = KDLKinematicsPlugin::searchPositionIK(pose, seed, timeout, solution, error_code, options);
    if (solved)
      updateGoalSpheres(solution);
    return solved;
  }

  bool searchPositionIK(const geometry_msgs::msg::Pose& pose, const std::vector<double>& seed, double timeout,
                        const std::vector<double>& consistency_limits, std::vector<double>& solution,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options = {}) const override
  {
    const bool solved = KDLKinematicsPlugin::searchPositionIK(
        pose, seed, timeout, consistency_limits, solution, error_code, options);
    if (solved)
      updateGoalSpheres(solution);
    return solved;
  }

  bool searchPositionIK(const geometry_msgs::msg::Pose& pose, const std::vector<double>& seed, double timeout,
                        std::vector<double>& solution, const IKCallbackFn& callback,
                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options = {}) const override
  {
    const bool solved = KDLKinematicsPlugin::searchPositionIK(
        pose, seed, timeout, solution, callback, error_code, options);
    if (solved)
      updateGoalSpheres(solution);
    return solved;
  }

  bool searchPositionIK(const geometry_msgs::msg::Pose& pose, const std::vector<double>& seed, double timeout,
                        const std::vector<double>& consistency_limits, std::vector<double>& solution,
                        const IKCallbackFn& callback, moveit_msgs::msg::MoveItErrorCodes& error_code,
                        const kinematics::KinematicsQueryOptions& options = {}) const override
  {
    const bool solved = KDLKinematicsPlugin::searchPositionIK(
        pose, seed, timeout, consistency_limits, solution, callback, error_code, options);
    if (solved)
      updateGoalSpheres(solution);
    return solved;
  }

private:
  void updateGoalSpheres(const std::vector<double>& solution) const
  {
    if (!goal_spheres_client_ || !goal_spheres_client_->service_is_ready())
      return;
    if (goal_spheres_request_in_flight_->exchange(true))
      return;
    auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
    request->robot_state.joint_state.header.frame_id = group_name_;
    request->robot_state.joint_state.name = getJointNames();
    request->robot_state.joint_state.position = solution;
    try
    {
      goal_spheres_client_->async_send_request(
          request, [request_in_flight = goal_spheres_request_in_flight_](
                       rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedFuture) {
            request_in_flight->store(false);
          });
    }
    catch (...)
    {
      goal_spheres_request_in_flight_->store(false);
      throw;
    }
  }

  std::string group_name_;
  rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr goal_spheres_client_;
  std::shared_ptr<std::atomic_bool> goal_spheres_request_in_flight_{
    std::make_shared<std::atomic_bool>(false)
  };
};
}  // namespace cyclo_cumotion_moveit_config

PLUGINLIB_EXPORT_CLASS(cyclo_cumotion_moveit_config::GoalStateKDLKinematicsPlugin, kinematics::KinematicsBase)
