#include "demo_1f/runtime/execution/trajectory_smoother.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <rclcpp/duration.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

namespace demo_1f {
namespace {
class RosEnvironment : public ::testing::Environment {
  void SetUp() override {
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }

  void TearDown() override {
    rclcpp::shutdown();
  }
};

auto * environment = ::testing::AddGlobalTestEnvironment(new RosEnvironment);

moveit_msgs::msg::RobotTrajectory segment(std::vector<std::vector<double>> positions) {
  moveit_msgs::msg::RobotTrajectory result;
  result.joint_trajectory.joint_names = {"j1", "j2"};
  for (auto & values : positions) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = std::move(values);
    result.joint_trajectory.points.push_back(point);
  }
  return result;
}

class SmootherTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto urdf = urdf::parseURDF(R"(
      <robot name="smooth_test">
        <link name="base"/><link name="one"/>
        <link name="two"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
        <joint name="j1" type="revolute"><parent link="base"/><child link="one"/>
          <axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1.2" effort="10"/></joint>
        <joint name="j2" type="revolute"><parent link="one"/><child link="two"/>
          <origin xyz="0.5 0 0"/><axis xyz="0 1 0"/>
          <limit lower="-2" upper="2" velocity="1.2" effort="10"/></joint>
      </robot>)");
    auto srdf = std::make_shared<srdf::Model>();
    ASSERT_TRUE(srdf->initString(*urdf,
        R"(<robot name="smooth_test"><group name="arms"><joint name="j1"/><joint name="j2"/></group></robot>)"));
    model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    scene = std::make_unique<planning_scene::PlanningScene>(model);
    scene->getCurrentStateNonConst().setToDefaultValues();
    config.velocity_scale = config.acceleration_scale = 1.0;
    config.coordinated_group = "arms";
    config.motion_limits = {{"j1", {0.6, 0.8, 4.0}}, {"j2", {0.6, 0.8, 4.0}}};
    first.name = "face_each_other";
    first.type = "dual_joint";
    first.velocity_scale = 1.0;
    first.acceleration_scale = 1.0;
    second = first;
    second.name = "open_approach";
    second.velocity_scale = 0.4;
  }

  Config config;
  Step first, second;
  moveit::core::RobotModelPtr model;
  std::unique_ptr<planning_scene::PlanningScene> scene;
};

TEST_F(SmootherTest, ConnectsViaPointWithNonzeroVelocityAndStopsAtTransfer) {
  auto path = smooth_segments(
      {segment({{0, 0}, {0.2, 0.1}, {0.4, 0.2}}), segment({{0.4, 0.2}, {0.8, 0.35}})},
      {first, second},
      config,
      model);
  ASSERT_EQ(path.segment_end_times.size(), 2u);
  bool crossed = false;
  const auto & points = path.trajectory.joint_trajectory.points;
  for (const auto & point : points) {
    if (std::abs(rclcpp::Duration(point.time_from_start).seconds() - path.segment_end_times[0]) <
        1e-8) {
      EXPECT_NEAR(point.positions[0], 0.4, 1e-8);
      EXPECT_GT(std::abs(point.velocities[0]), 0.01);
      EXPECT_NEAR(point.accelerations[0], 0.0, 1e-8);
      crossed = true;
    }
  }
  EXPECT_TRUE(crossed);
  EXPECT_NEAR(points.front().velocities[0], 0.0, 1e-10);
  EXPECT_NEAR(points.back().velocities[0], 0.0, 1e-10);
  EXPECT_NEAR(points.back().accelerations[0], 0.0, 1e-10);
  EXPECT_NEAR(points.back().positions[0], 0.8, 1e-8);
  EXPECT_NO_THROW(validate_smooth_path(path, {first, second}, config, *scene));
}

TEST_F(SmootherTest, RespectsSlowerApproachAndUrdfVelocityCap) {
  config.motion_limits["j1"].velocity = 10;
  auto path = smooth_segments({segment({{0, 0}, {0.4, 0.2}}), segment({{0.4, 0.2}, {0.8, 0.4}})},
      {first, second},
      config,
      model);
  for (size_t i = 0; i < path.point_steps.size(); ++i) {
    const auto & point = path.trajectory.joint_trajectory.points[i];
    const double cap = path.point_steps[i] == 0 ? 1.2 : 0.48;
    EXPECT_LE(std::abs(point.velocities[0]), cap + 1e-8);
  }
  EXPECT_NO_THROW(validate_smooth_path(path, {first, second}, config, *scene));
}

TEST_F(SmootherTest, KeepsGripperExactlyFixedUntilItsOpeningSegment) {
  config.arms["left"].gripper_joint = "j2";
  second.type = "dual_joint_gripper";
  second.arm = "left";
  auto path = smooth_segments(
      {segment({{0, 0.8}, {0.2, 0.7992}, {0.4, 0.7995}}), segment({{0.4, 0.7995}, {0.8, 0.0}})},
      {first, second},
      config,
      model);
  for (size_t i = 0; i < path.point_steps.size(); ++i) {
    const auto & point = path.trajectory.joint_trajectory.points[i];
    if (path.point_steps[i] == 0) {
      EXPECT_NEAR(point.positions[1], 0.8, 1e-10);
      EXPECT_NEAR(point.velocities[1], 0.0, 1e-10);
    }
  }
  EXPECT_NEAR(path.trajectory.joint_trajectory.points.back().positions[1], 0.0, 1e-8);
  EXPECT_NO_THROW(validate_smooth_path(path, {first, second}, config, *scene));
  EXPECT_THROW(smooth_segments({segment({{0, 0.8}, {0.4, 0.79}})}, {first}, config, model),
      std::runtime_error);
}

TEST_F(SmootherTest, RejectsCollisionIntroducedOnTheExecutedPath) {
  auto path = smooth_segments({segment({{0, 0}, {0.4, 0.2}})}, {first}, config, model);
  moveit_msgs::msg::CollisionObject obstacle;
  obstacle.header.frame_id = "base";
  obstacle.id = "blocking_box";
  obstacle.operation = obstacle.ADD;
  shape_msgs::msg::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions = {10, 10, 10};
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1;
  obstacle.primitives.push_back(box);
  obstacle.primitive_poses.push_back(pose);
  ASSERT_TRUE(scene->processCollisionObjectMsg(obstacle));
  EXPECT_THROW(validate_smooth_path(path, {first}, config, *scene), std::runtime_error);
}

TEST_F(SmootherTest, RejectsBoundsAndDiscontinuousPlanningStarts) {
  auto path = smooth_segments({segment({{0, 0}, {2.1, 0.2}})}, {first}, config, model);
  EXPECT_THROW(validate_smooth_path(path, {first}, config, *scene), std::runtime_error);
  EXPECT_THROW(smooth_segments({segment({{0, 0}, {0.4, 0.2}}), segment({{0.5, 0.2}, {0.8, 0.3}})},
                   {first, second},
                   config,
                   model),
      std::runtime_error);
}

TEST_F(SmootherTest, RejectsMalformedAndNonfinitePaths) {
  auto malformed = segment({{0, 0}, {0.4}});
  EXPECT_THROW(smooth_segments({malformed}, {first}, config, model), std::runtime_error);
  auto invalid = segment({{0, 0}, {std::numeric_limits<double>::quiet_NaN(), 0.2}});
  EXPECT_THROW(smooth_segments({invalid}, {first}, config, model), std::runtime_error);
}

TEST_F(SmootherTest, RepeatedPositionsStillProduceStrictlyIncreasingTimestamps) {
  auto path = smooth_segments({segment({{0, 0}, {0, 0}, {0.4, 0.2}})}, {first}, config, model);
  double previous = -1;
  for (const auto & point : path.trajectory.joint_trajectory.points) {
    const double time = rclcpp::Duration(point.time_from_start).seconds();
    EXPECT_GT(time, previous);
    previous = time;
  }
  EXPECT_NO_THROW(validate_smooth_path(path, {first}, config, *scene));
}

TEST_F(SmootherTest, GripperDurationIsAMinimumAndLimitedProfileIsValidated) {
  first.type = "gripper";
  first.duration = 0.9;
  config.motion_limits["j1"] = {2.0, 2.0, 20.0};
  auto path = smooth_segments({segment({{0, 0}, {0.8, 0}})}, {first}, config, model);
  EXPECT_GE(
      rclcpp::Duration(path.trajectory.joint_trajectory.points.back().time_from_start).seconds(),
      first.duration);
  EXPECT_NO_THROW(validate_smooth_path(path, {first}, config, *scene));
  auto stationary = smooth_segments({segment({{0, 0}, {0, 0}})}, {first}, config, model);
  EXPECT_GE(rclcpp::Duration(stationary.trajectory.joint_trajectory.points.back().time_from_start)
                .seconds(),
      first.duration);
  EXPECT_NO_THROW(validate_smooth_path(stationary, {first}, config, *scene));
}
} // namespace
} // namespace demo_1f
