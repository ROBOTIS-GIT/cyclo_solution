#include "demo_1f/moveit_plugins/snap_start_state_bounds.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

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

class StartStateBoundsTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto urdf = urdf::parseURDF(R"(
      <robot name="bounds_test">
        <link name="base"/><link name="left_link"/><link name="right_link"/>
        <joint name="left" type="revolute">
          <parent link="base"/><child link="left_link"/><axis xyz="0 0 1"/>
          <limit lower="0" upper="3.14" effort="10" velocity="1"/>
        </joint>
        <joint name="right" type="revolute">
          <parent link="base"/><child link="right_link"/><axis xyz="0 0 1"/>
          <limit lower="-3.14" upper="0" effort="10" velocity="1"/>
        </joint>
      </robot>)");
    ASSERT_TRUE(urdf);
    auto srdf = std::make_shared<srdf::Model>();
    ASSERT_TRUE(srdf->initString(*urdf, R"(
      <robot name="bounds_test"><group name="arms">
        <joint name="left"/><joint name="right"/>
      </group></robot>)"));
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    scene = std::make_shared<planning_scene::PlanningScene>(model);
    scene->getCurrentStateNonConst().setToDefaultValues();
    request.group_name = "arms";
    request.start_state.is_diff = true;
  }

  void positions(double left, double right) {
    auto & state = scene->getCurrentStateNonConst();
    state.setVariablePosition("left", left);
    state.setVariablePosition("right", right);
    state.update();
  }

  moveit::core::RobotState requested_state() {
    auto state = scene->getCurrentState();
    EXPECT_TRUE(moveit::core::robotStateMsgToRobotState(
        scene->getTransforms(), request.start_state, state));
    return state;
  }

  planning_scene::PlanningScenePtr scene;
  planning_interface::MotionPlanRequest request;
  demo_1f::SnapStartStateBounds adapter;
};

TEST_F(StartStateBoundsTest, SnapsGazeboRoundoffWithoutChangingLimitsOrScene) {
  positions(-6.77264e-14, 7.24182e-14);
  EXPECT_EQ(adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  auto corrected = requested_state();
  EXPECT_DOUBLE_EQ(corrected.getVariablePosition("left"), 0.0);
  EXPECT_DOUBLE_EQ(corrected.getVariablePosition("right"), 0.0);
  EXPECT_TRUE(corrected.satisfiesBounds());
  EXPECT_DOUBLE_EQ(scene->getCurrentState().getVariablePosition("left"), -6.77264e-14);
  EXPECT_DOUBLE_EQ(scene->getCurrentState().getVariablePosition("right"), 7.24182e-14);
  EXPECT_DOUBLE_EQ(scene->getRobotModel()->getVariableBounds("left").min_position_, 0.0);
  EXPECT_DOUBLE_EQ(scene->getRobotModel()->getVariableBounds("right").max_position_, 0.0);
}

TEST_F(StartStateBoundsTest, LeavesValidRequestUnchanged) {
  positions(0.2, -0.3);
  auto original = request;
  EXPECT_EQ(adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  EXPECT_EQ(request, original);
}

TEST_F(StartStateBoundsTest, UsesExplicitStartStateAndPreservesVelocity) {
  positions(0.2, -0.3);
  request.start_state.joint_state.name = {"left", "right"};
  request.start_state.joint_state.position = {-6.77264e-14, -0.4};
  request.start_state.joint_state.velocity = {0.1, -0.2};
  EXPECT_EQ(adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  auto corrected = requested_state();
  EXPECT_DOUBLE_EQ(corrected.getVariablePosition("left"), 0.0);
  EXPECT_DOUBLE_EQ(corrected.getVariablePosition("right"), -0.4);
  EXPECT_DOUBLE_EQ(corrected.getVariableVelocity("left"), 0.1);
  EXPECT_DOUBLE_EQ(corrected.getVariableVelocity("right"), -0.2);
}

TEST_F(StartStateBoundsTest, RejectsLargerViolationsWithoutPartialCorrection) {
  for (const double error : {1.0001e-8, 1e-5, 0.1}) {
    for (const bool left_bad : {true, false}) {
      positions(left_bad ? -error : -1e-13, left_bad ? 1e-13 : error);
      auto original = request;
      EXPECT_EQ(adapter.adapt(scene, request).val,
          moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID);
      EXPECT_EQ(request, original);
    }
  }
}

TEST_F(StartStateBoundsTest, SnapsAtToleranceAndBothOtherBounds) {
  positions(-1e-8, 1e-8);
  EXPECT_EQ(adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  EXPECT_TRUE(requested_state().satisfiesBounds());
  request.start_state = moveit_msgs::msg::RobotState();
  request.start_state.is_diff = true;
  positions(3.14 + 1e-13, -3.14 - 1e-13);
  EXPECT_EQ(adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  auto corrected = requested_state();
  EXPECT_DOUBLE_EQ(corrected.getVariablePosition("left"), 3.14);
  EXPECT_DOUBLE_EQ(corrected.getVariablePosition("right"), -3.14);
}

TEST_F(StartStateBoundsTest, RejectsNonFinitePosition) {
  for (const double value : {std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()}) {
    positions(value, 0.0);
    EXPECT_EQ(
        adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID);
  }
}

TEST_F(StartStateBoundsTest, RejectsUnknownGroup) {
  request.group_name = "missing";
  EXPECT_EQ(
      adapter.adapt(scene, request).val, moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME);
}
} // namespace
