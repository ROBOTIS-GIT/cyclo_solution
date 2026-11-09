#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <gtest/gtest.h>
#include <moveit/constraint_samplers/constraint_sampler_allocator.hpp>
#include <moveit/kinematic_constraints/kinematic_constraint.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <pluginlib/class_loader.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

namespace demo_1f {
namespace {
Eigen::Matrix3d hand_tcp_rotation(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitY()))
      .toRotationMatrix();
}

geometry_msgs::msg::Quaternion message(const Eigen::Matrix3d & rotation) {
  const Eigen::Quaterniond q(rotation);
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}
} // namespace

TEST(TcpRoll, MoveItMobileFrameConstraintMatchesBodyRollAcrossYawAndPitch) {
  const std::string urdf = R"(<robot name="roll_test"><link name="base_link"/>
    <link name="tcp"/><joint name="wrist" type="floating"><parent link="base_link"/>
    <child link="tcp"/></joint></robot>)";
  auto parsed = urdf::parseURDF(urdf);
  auto srdf = std::make_shared<srdf::Model>();
  ASSERT_TRUE(srdf->initString(*parsed, "<robot name='roll_test'/>"));
  auto model = std::make_shared<moveit::core::RobotModel>(parsed, srdf);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  kinematic_constraints::OrientationConstraint constraint(model);
  ASSERT_TRUE(constraint.configure(tcp_roll_constraint("tcp", "base_link", M_PI / 3),
      moveit::core::Transforms(model->getModelFrame())));
  for (double yaw : {-179., -90., -25., 0., 60., 179.}) {
    for (double pitch : {-80., -45., 0., 40., 80.}) {
      for (double roll : {-170., -90., -61., -59., 0., 59., 61., 90., 170.}) {
        auto rotation = hand_tcp_rotation(roll * M_PI / 180, pitch * M_PI / 180, yaw * M_PI / 180);
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = rotation;
        state.setJointPositions(model->getJointModel("wrist"), pose);
        state.update();
        const auto measured = tcp_roll(message(rotation));
        ASSERT_TRUE(measured);
        EXPECT_NEAR(*measured, roll * M_PI / 180, 1e-9);
        EXPECT_EQ(constraint.decide(state).satisfied, std::abs(roll) < 60)
            << "roll=" << roll << " pitch=" << pitch << " yaw=" << yaw;
      }
    }
  }
}

TEST(TcpRoll, VerticalForwardAndInvalidQuaternionAreNotAcceptedAsZeroRoll) {
  EXPECT_FALSE(tcp_roll(message(hand_tcp_rotation(0, M_PI / 2, 0))));
  EXPECT_FALSE(tcp_roll(message(hand_tcp_rotation(0, -M_PI / 2, 0))));
  EXPECT_FALSE(tcp_roll(geometry_msgs::msg::Quaternion{}));
}

TEST(TcpRoll, SamplerPluginChecksRollAndPreservesValidityCallback) {
  const auto parsed = urdf::parseURDF(R"(<robot name="sample_test"><link name="base_link"/>
    <link name="end_effector_l_link"/><joint name="wrist" type="floating">
    <parent link="base_link"/><child link="end_effector_l_link"/></joint></robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  ASSERT_TRUE(srdf->initString(*parsed,
      "<robot name='sample_test'><group name='arm_l'><joint name='wrist'/></group></robot>"));
  auto model = std::make_shared<moveit::core::RobotModel>(parsed, srdf);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  pluginlib::ClassLoader<constraint_samplers::ConstraintSamplerAllocator> loader(
      "moveit_core", "constraint_samplers::ConstraintSamplerAllocator");
  auto allocator = loader.createSharedInstance("demo_1f/RollSamplerAllocator");
  moveit_msgs::msg::Constraints constraints;
  constraints.orientation_constraints.push_back(
      tcp_roll_constraint("end_effector_l_link", "base_link", M_PI / 3));
  ASSERT_TRUE(allocator->canService(scene, "arm_l", constraints));
  EXPECT_FALSE(allocator->canService(scene, "handover_arms", constraints));
  auto sampler = allocator->alloc(scene, "arm_l", constraints);
  ASSERT_TRUE(sampler);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  ASSERT_TRUE(sampler->sample(state, 100));
  const auto roll = tcp_roll(message(state.getGlobalLinkTransform("end_effector_l_link").linear()));
  ASSERT_TRUE(roll);
  EXPECT_LE(std::abs(*roll), M_PI / 3);
  sampler->setGroupStateValidityCallback(
      [](moveit::core::RobotState *, const moveit::core::JointModelGroup *, const double *) {
        return false;
      });
  EXPECT_FALSE(sampler->sample(state, 100));
}

TEST(TcpRoll, DownwardPitchBandKeepsYawFree) {
  const auto parsed = urdf::parseURDF(R"(<robot name="tilt_test"><link name="base_link"/>
    <link name="tcp"/><joint name="wrist" type="floating"><parent link="base_link"/>
    <child link="tcp"/></joint></robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  ASSERT_TRUE(srdf->initString(*parsed, "<robot name='tilt_test'/>"));
  auto model = std::make_shared<moveit::core::RobotModel>(parsed, srdf);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  kinematic_constraints::OrientationConstraint band(model), downward(model);
  auto transforms = moveit::core::Transforms(model->getModelFrame());
  ASSERT_TRUE(band.configure(
      tcp_roll_constraint("tcp", "base_link", 30 * M_PI / 180, 40 * M_PI / 180), transforms));
  ASSERT_TRUE(downward.configure(tcp_downward_constraint("tcp", "base_link"), transforms));
  for (double yaw : {-179., -90., -25., 0., 60., 179.}) {
    for (double pitch : {-60., -39., -20., -1., 0., 1., 20., 39., 41., 80.}) {
      for (double roll : {-31., -29., -1., 0., 1., 29., 31.}) {
        auto rotation = hand_tcp_rotation(roll * M_PI / 180, pitch * M_PI / 180, yaw * M_PI / 180);
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = rotation;
        state.setJointPositions(model->getJointModel("wrist"), pose);
        state.update();
        EXPECT_NEAR(*tcp_pitch(message(rotation)), pitch * M_PI / 180, 1e-9);
        const bool accepted = band.decide(state).satisfied && downward.decide(state).satisfied;
        // Exactly horizontal is the half-space boundary and can be rejected by Euler
        // singularity handling. The motion aligns starts below one degree before planning.
        if (pitch != 0) {
          EXPECT_EQ(accepted, std::abs(roll) < 30 && pitch > 0 && pitch < 40)
              << "roll=" << roll << " pitch=" << pitch << " yaw=" << yaw;
        }
        if (accepted) {
          EXPECT_LE((-rotation.col(2)).z(), 1e-12);
        }
        if (pitch > 0) {
          EXPECT_LT((-rotation.col(2)).z(), 0);
        }
      }
    }
    const auto aligned =
        leveled_tcp(message(hand_tcp_rotation(.4, -.3, yaw * M_PI / 180)), true, 20 * M_PI / 180);
    EXPECT_NEAR(*tcp_pitch(aligned), 20 * M_PI / 180, 1e-9);
    EXPECT_NEAR(*tcp_roll(aligned), 0, 1e-9);
  }
}

TEST(TcpRoll, InwardYawMatchesHandHeadingForBothArmsAcrossTilt) {
  const auto parsed = urdf::parseURDF(R"(<robot name="yaw_test"><link name="base_link"/>
    <link name="tcp"/><joint name="wrist" type="floating"><parent link="base_link"/>
    <child link="tcp"/></joint></robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  ASSERT_TRUE(srdf->initString(*parsed, "<robot name='yaw_test'/>"));
  auto model = std::make_shared<moveit::core::RobotModel>(parsed, srdf);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  for (const std::string arm : {"left", "right"}) {
    kinematic_constraints::OrientationConstraint constraint(model);
    ASSERT_TRUE(constraint.configure(tcp_inward_yaw_constraint("tcp", "base_link", arm),
        moveit::core::Transforms(model->getModelFrame())));
    for (double yaw :
        {-179., -100., -91., -90., -88., -45., -2., 0., 2., 45., 88., 90., 91., 100., 179.}) {
      for (double pitch : {-80., -20., 0., 20., 40., 80.}) {
        for (double roll : {-170., -30., 0., 30., 170.}) {
          const auto rotation =
              hand_tcp_rotation(roll * M_PI / 180, pitch * M_PI / 180, yaw * M_PI / 180);
          Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
          pose.linear() = rotation;
          state.setJointPositions(model->getJointModel("wrist"), pose);
          state.update();
          const auto measured = tcp_yaw(message(rotation));
          ASSERT_TRUE(measured);
          EXPECT_NEAR(*measured, yaw * M_PI / 180, 1e-9);
          const double inward = arm == "left" ? -yaw : yaw;
          EXPECT_EQ(constraint.decide(state).satisfied, inward > 1 && inward < 89)
              << arm << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll;
          EXPECT_EQ(tcp_yaw_is_inward(*measured, arm), inward >= 0 && inward <= 90);
        }
      }
    }
  }
  EXPECT_FALSE(tcp_yaw(message(hand_tcp_rotation(0, M_PI / 2, 0))));
  EXPECT_FALSE(tcp_yaw_is_inward(0, "unknown"));
}

TEST(TcpRoll, TargetYawBandMatchesAcrossTilt) {
  const auto parsed = urdf::parseURDF(R"(<robot name="yaw_test"><link name="base_link"/>
    <link name="tcp"/><joint name="wrist" type="floating"><parent link="base_link"/>
    <child link="tcp"/></joint></robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  ASSERT_TRUE(srdf->initString(*parsed, "<robot name='yaw_test'/>"));
  auto model = std::make_shared<moveit::core::RobotModel>(parsed, srdf);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  for (double center : {-170., -30., 0., 40., 170.}) {
    kinematic_constraints::OrientationConstraint constraint(model);
    ASSERT_TRUE(constraint.configure(
        tcp_yaw_constraint("tcp", "base_link", center * M_PI / 180, 14 * M_PI / 180),
        moveit::core::Transforms(model->getModelFrame())));
    for (double yaw :
        {-179., -100., -91., -90., -88., -45., -2., 0., 2., 45., 88., 90., 91., 100., 179.}) {
      for (double pitch : {-80., -20., 0., 20., 40., 80.}) {
        for (double roll : {-170., -30., 0., 30., 170.}) {
          const auto rotation =
              hand_tcp_rotation(roll * M_PI / 180, pitch * M_PI / 180, yaw * M_PI / 180);
          Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
          pose.linear() = rotation;
          state.setJointPositions(model->getJointModel("wrist"), pose);
          state.update();
          const auto measured = tcp_yaw(message(rotation));
          ASSERT_TRUE(measured);
          EXPECT_NEAR(*measured, yaw * M_PI / 180, 1e-9);
          const double error = std::abs(std::remainder(yaw - center, 360.));
          EXPECT_EQ(constraint.decide(state).satisfied, error < 14)
              << "center=" << center << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll;
        }
      }
    }
  }
  EXPECT_FALSE(tcp_yaw(message(hand_tcp_rotation(0, M_PI / 2, 0))));
  EXPECT_FALSE(tcp_yaw_is_inward(0, "unknown"));
}

TEST(TcpRoll, MinimumPitchIncludesTcpPositionToleranceAcrossYawAndRoll) {
  const auto parsed = urdf::parseURDF(R"(<robot name="pitch_test"><link name="base_link"/>
    <link name="tcp"/><joint name="wrist" type="floating"><parent link="base_link"/>
    <child link="tcp"/></joint></robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  ASSERT_TRUE(srdf->initString(*parsed, "<robot name='pitch_test'/>"));
  auto model = std::make_shared<moveit::core::RobotModel>(parsed, srdf);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  geometry_msgs::msg::Point goal;
  goal.x = .6;
  goal.y = .2;
  goal.z = 1.0;
  constexpr double radius = .005;
  const auto transforms = moveit::core::Transforms(model->getModelFrame());
  kinematic_constraints::PositionConstraint minimum(model);
  kinematic_constraints::OrientationConstraint band(model);
  ASSERT_TRUE(minimum.configure(
      tcp_min_pitch_constraint("tcp", "base_link", goal, radius, 21 * M_PI / 180), transforms));
  ASSERT_TRUE(band.configure(
      tcp_roll_constraint("tcp", "base_link", 9 * M_PI / 180, 59 * M_PI / 180), transforms));
  for (double yaw : {-179., -90., -45., 0., 45., 90., 179.}) {
    for (double roll : {-8., 0., 8.}) {
      for (double pitch : {-30., 0., 19., 20., 20.9, 22., 40., 58., 60., 65., 80.}) {
        for (double dz : {-radius, 0., radius}) {
          Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
          pose.translation() = Eigen::Vector3d(goal.x, goal.y, goal.z + dz);
          pose.linear() =
              hand_tcp_rotation(roll * M_PI / 180, pitch * M_PI / 180, yaw * M_PI / 180);
          state.setJointPositions(model->getJointModel("wrist"), pose);
          state.update();
          const bool accepted = minimum.decide(state).satisfied && band.decide(state).satisfied;
          EXPECT_EQ(accepted, pitch >= 22 && pitch <= 58)
              << "yaw=" << yaw << " roll=" << roll << " pitch=" << pitch << " dz=" << dz;
        }
      }
    }
  }
}
} // namespace demo_1f
