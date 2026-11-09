#pragma once
#include "demo_1f/motions/common/cup_config.hpp"
#include "demo_1f/runtime/state/robot_context.hpp"
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>

namespace demo_1f {
std::optional<geometry_msgs::msg::Pose> approach_tcp_target(const geometry_msgs::msg::Pose & body,
    const ApproachConfig & config,
    const std::string & arm,
    double uncertainty,
    std::string & reason);
geometry_msgs::msg::Pose tilt_tcp_target(
    const geometry_msgs::msg::Pose & pose, double degrees, double ceiling_degrees);
moveit_msgs::msg::PlanningScene make_approach_scene(const Config & robot,
    const ApproachConfig & config,
    const geometry_msgs::msg::Pose & body,
    double uncertainty);

class ApproachScenePort {
public:
  virtual ~ApproachScenePort() = default;
  virtual bool ready() const = 0;
  virtual bool apply(const moveit_msgs::msg::PlanningScene & scene) = 0;
  virtual ExecutionResult poll() = 0;
};

class ApproachScene final : public ApproachScenePort {
public:
  explicit ApproachScene(rclcpp::Node & node);
  bool ready() const override;
  bool apply(const moveit_msgs::msg::PlanningScene & scene) override;
  ExecutionResult poll() override;

private:
  rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr client_;
  rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedFuture future_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr query_client_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedFuture query_;
  moveit_msgs::msg::PlanningScene pending_scene_;
  TimePoint started_;
  bool timed_out_{false};
};

} // namespace demo_1f
