#include "demo_1f/motions/pick/pick_cup_motion.hpp"
#include "demo_1f/motions/place/approach_cup_motion.hpp"
#include "demo_1f/motions/place/refine_cup_motion.hpp"
#include "demo_1f/motions/place/withdraw_cup_motion.hpp"
#include "demo_1f/runtime/execution/trajectory_smoother.hpp"
#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include "demo_1f/tasks/task_manager.hpp"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <thread>
using namespace std::chrono_literals;

namespace demo_1f {
namespace {
class Environment : public ::testing::Environment {
  void SetUp() override {
    initialize_moveit_logging();
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }

  void TearDown() override {
    rclcpp::shutdown();
  }
};

auto * environment = ::testing::AddGlobalTestEnvironment(new Environment);

TEST(ApproachGeometry, PreservesHeightAndCenterAssumptionWithoutExtraPadding) {
  ApproachConfig config;
  geometry_msgs::msg::Pose body;
  body.position.z = .8;
  body.orientation.w = 1;
  std::string reason;
  auto target = approach_tcp_target(body, config, "left", .09, reason);
  ASSERT_TRUE(target) << reason;
  EXPECT_NEAR(target->position.z, .965, 1e-10);
  EXPECT_FALSE(approach_tcp_target(body, config, "left", .2, reason));
  auto scene = make_approach_scene({}, config, body, .09);
  const auto & cup = scene.world.collision_objects.front();
  EXPECT_NEAR(cup.primitives.front().dimensions[0], .165, 1e-10);
  EXPECT_NEAR(cup.primitive_poses.front().position.z, .8, 1e-10);
  EXPECT_NEAR(cup.primitives.front().dimensions[1], .14, 1e-10);
  EXPECT_TRUE(scene.robot_state.attached_collision_objects.empty());
}

TEST(ApproachGeometry, RejectsInvalidConfiguration) {
  ApproachConfig config;
  EXPECT_FALSE(config.validate("unknown").empty());
  for (double roll : {-1., 1., 91., std::numeric_limits<double>::quiet_NaN()}) {
    config.tcp_roll_limit_deg = roll;
    EXPECT_FALSE(config.validate("left").empty());
  }
}

TEST(ApproachGeometry, LowersTargetWithoutMovingCupCollisionGeometry) {
  ApproachConfig config;
  config.offset_z = -.05;
  geometry_msgs::msg::Pose body;
  body.position.x = .5;
  body.position.y = .2;
  body.position.z = .8;
  body.orientation.w = 1;
  std::string reason;
  for (const std::string arm : {"left", "right"}) {
    const auto target = approach_tcp_target(body, config, arm, .01, reason);
    ASSERT_TRUE(target) << reason;
    EXPECT_NEAR(target->position.x, .5, 1e-10);
    EXPECT_NEAR(target->position.y, .2, 1e-10);
    EXPECT_NEAR(target->position.z, .915, 1e-10);
  }
  const auto scene = make_approach_scene({}, config, body, .01);
  const auto & cup = scene.world.collision_objects.front();
  EXPECT_NEAR(cup.primitives.front().dimensions[0], .165, 1e-10);
  EXPECT_NEAR(cup.primitive_poses.front().position.z, .8, 1e-10);
  for (double offset : {-.31, .31, std::numeric_limits<double>::quiet_NaN()}) {
    config.offset_z = offset;
    EXPECT_FALSE(approach_tcp_target(body, config, "left", .01, reason));
  }
}

TEST(ApproachGeometry, RejectsInconsistentMinimumPitch) {
  ApproachConfig config;
  config.tcp_pitch_limit_deg = 60;
  config.tcp_pitch_min_deg = 20;
  EXPECT_TRUE(config.validate("left").empty());
  config.tcp_pitch_downward_only = false;
  EXPECT_FALSE(config.validate("left").empty());
  config.tcp_pitch_downward_only = true;
  for (double minimum : {-1., 58., 60., 70., std::numeric_limits<double>::quiet_NaN()}) {
    config.tcp_pitch_min_deg = minimum;
    EXPECT_FALSE(config.validate("left").empty());
  }
}

class Executor : public ExecutorPort {
public:
  bool ready(const MotionSequence &, std::string &) const override {
    return stationary;
  }

  bool start(const Step & s) override {
    steps.push_back(s);
    current = {ExecutionState::Running, {}, {}};
    return true;
  }

  void tick() override {}

  void cancel(const std::string & code, const std::string & reason) override {
    current = {ExecutionState::Cancelled, code, reason};
  }

  ExecutionResult result() const override {
    return current;
  }

  bool stationary = true;
  ExecutionResult current;
  std::vector<Step> steps;
};

class Scene : public ApproachScenePort {
public:
  bool ready() const override {
    return !pending;
  }

  bool apply(const moveit_msgs::msg::PlanningScene & s) override {
    scenes.push_back(s);
    pending = true;
    return true;
  }

  ExecutionResult poll() override {
    if (!finish) {
      return {ExecutionState::Running, {}, {}};
    }
    pending = false;
    return outcome;
  }

  bool pending = false, finish = true;
  ExecutionResult outcome{ExecutionState::Succeeded, {}, {}};
  std::vector<moveit_msgs::msg::PlanningScene> scenes;
};

class ApproachFixture : public ::testing::Test {
protected:
  void SetUp() override {
    static int index = 0;
    node = std::make_shared<rclcpp::Node>("approach_test_" + std::to_string(++index));
    robot.require_tcp_tf = false;
    robot.gripper_stop_mode = "velocity";
    robot.state_max_age = 1;
    for (const std::string side : {"left", "right"}) {
      ArmConfig a;
      a.name = side;
      a.joints = {side + "_joint"};
      a.gripper_joint = side + "_grip";
      a.tcp = node->get_name() + std::string("_") + side + "_tcp";
      robot.arms[side] = a;
    }
    config = ApproachConfig{};
    config.refine_view_wait = .001;
    config.inner_radius = .08;
    config.observation_timeout = .2;
    config.tcp_yaw_inward_only = false;
    config.tcp_roll_limit_deg = 0;
    config.tcp_pitch_limit_deg = 0;
    config.tcp_pitch_downward_only = false;
    context = std::make_unique<RobotContext>(*node, robot,
        HeadCupTrackConfig{config.inner_radius, config.wall_thickness, config.max_position_error});
    motion = std::make_unique<ApproachCupMotion>(robot, config, *context, executor, scene);
    refinement = std::make_unique<RefineCupMotion>(robot, config, *context, executor, scene);
    withdrawal = std::make_unique<WithdrawCupMotion>(robot, config, *context, executor, scene);
    broadcaster = std::make_unique<tf2_ros::StaticTransformBroadcaster>(node);
    ros.add_node(node);
    feedback();
  }

  void feedback(double grip = .8, double gripper_velocity = 0) {
    sensor_msgs::msg::JointState j;
    j.header.stamp = node->now();
    for (const auto & [side, a] : robot.arms) {
      j.name.push_back(a.joints[0]);
      j.position.push_back(0);
      j.velocity.push_back(0);
      j.name.push_back(a.gripper_joint);
      j.position.push_back(grip);
      j.velocity.push_back(gripper_velocity);
    }
    context->observe_joints(j);
  }

  void observe(const std::string & source,
      double shift = 0,
      bool valid = true,
      double error = .09,
      int frames = 1,
      const std::string & detail = "") {
    msg::CupObservation o;
    o.observation_type = msg::CupObservation::BODY;
    o.header.frame_id = robot.planning_frame;
    o.source = source;
    o.cup_id = "cup";
    o.valid = valid;
    o.body_position.x = .54 + shift;
    o.body_position.z = .84;
    o.inner_radius = .08;
    o.position_error = error;
    o.confidence = 1;
    o.detail = detail;
    for (int i = 0; i < frames; ++i) {
      o.header.stamp = node->now();
      context->observe_cup(o);
      std::this_thread::sleep_for(1ms);
    }
  }

  void capture_initial() {
    for (int i = 0; i < 3; ++i) {
      observe("head");
      motion->tick();
    }
  }

  bool relaxed_pick_arrival = false;

  void complete() {
    if (executor.current.state == ExecutionState::Running) {
      auto s = executor.steps.back();
      if (s.target.size() == 7) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.frame_id = robot.planning_frame;
        tf.child_frame_id = robot.arms.at(s.arm).tcp;
        tf.header.stamp = node->now();
        tf.transform.translation.x = s.target[0];
        tf.transform.translation.y = s.target[1];
        tf.transform.translation.z = s.target[2];
        tf.transform.rotation.x = s.target[3];
        tf.transform.rotation.y = s.target[4];
        tf.transform.rotation.z = s.target[5];
        tf.transform.rotation.w = s.target[6];
        if (s.tcp_roll_limit > 0 && s.tcp_yaw_limit == 0) {
          tf.transform.rotation = geometry_msgs::msg::Quaternion{};
          tf.transform.rotation.y = -std::sin(M_PI / 4);
          tf.transform.rotation.w = std::cos(M_PI / 4);
        }
        if (relaxed_pick_arrival && s.name == "PICK_PREGRASP") {
          const double yaw = *tcp_yaw(tf.transform.rotation) + 8 * M_PI / 180;
          tf2::Quaternion hand, axes;
          hand.setRPY(4 * M_PI / 180, 5 * M_PI / 180, yaw);
          axes.setRPY(0, -M_PI / 2, 0);
          tf.transform.rotation = tf2::toMsg(hand * axes);
        }
        broadcaster->sendTransform(tf);
        for (int i = 0; i < 3; ++i) {
          ros.spin_some();
          std::this_thread::sleep_for(2ms);
        }
      }
      executor.current = {ExecutionState::Succeeded, {}, {}};
    }
  }

  bool start(const std::string & arm = "right") {
    std::string reason;
    return motion->start(arm, "cup", reason);
  }

  void advance(const std::string & desired) {
    for (int i = 0; i < 60 && motion->running(); ++i) {
      if (context->snapshot().phase == desired) {
        return;
      }
      feedback();
      observe("head");
      complete();
      motion->tick();
    }
    ASSERT_EQ(context->snapshot().phase, desired) << motion->result().detail;
  }

  void shoulder_for_pick() {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.frame_id = robot.planning_frame;
    tf.header.stamp = node->now();
    tf.transform.rotation.w = 1;
    for (const auto & side : {"l", "r"}) {
      tf.child_frame_id = std::string("arm_") + side + "_link1";
      tf.transform.translation.y = std::string(side) == "l" ? .1045 : -.1045;
      broadcaster->sendTransform(tf);
    }
    for (int i = 0; i < 10; ++i) {
      ros.spin_some();
      std::this_thread::sleep_for(2ms);
    }
  }

  void tcp_for_refinement(
      const std::string & side = "left", double x = .5, double y = 0, double z = 1.) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.frame_id = robot.planning_frame;
    tf.child_frame_id = robot.arms.at(side).tcp;
    tf.header.stamp = node->now();
    tf.transform.translation.x = x;
    tf.transform.translation.y = y;
    tf.transform.translation.z = z;
    tf2::Quaternion q;
    q.setRPY(0, -M_PI / 2, 0);
    tf.transform.rotation = tf2::toMsg(q);
    broadcaster->sendTransform(tf);
    tf.child_frame_id = side == "left" ? "arm_l_link6" : "arm_r_link6";
    // Distinct wrist origin verifies that pickup does not use TCP position.
    tf.transform.translation.x -= .1;
    broadcaster->sendTransform(tf);
    for (int i = 0; i < 10; ++i) {
      ros.spin_some();
      std::this_thread::sleep_for(2ms);
    }
  }

  void wrist(bool valid,
      const std::string & detail = "NO_CANDIDATE: empty",
      double x = .54,
      double age = 0,
      const std::string & arm = "left", double confidence = .7) {
    msg::CupObservation o;
    o.header.frame_id = robot.planning_frame;
    o.header.stamp = node->now() - rclcpp::Duration::from_seconds(age);
    o.source = "wrist_" + arm;
    o.cup_id = "cup";
    o.observation_type = 2;
    o.valid = valid;
    o.detail = detail;
    o.body_position.x = x;
    o.body_position.z = .9;
    o.inner_radius = config.inner_radius;
    o.position_error = .008;
    o.confidence = confidence;
    context->observe_cup(o);
    std::this_thread::sleep_for(1ms);
  }

  rclcpp::Node::SharedPtr node;
  Config robot;
  ApproachConfig config;
  std::unique_ptr<RobotContext> context;
  Executor executor;
  Scene scene;
  std::unique_ptr<ApproachCupMotion> motion;
  std::unique_ptr<RefineCupMotion> refinement;
  std::unique_ptr<WithdrawCupMotion> withdrawal;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster;
  rclcpp::executors::SingleThreadedExecutor ros;
};

TEST(RefinementGeometry, TiltKeepsPositionYawAndRollAndIncreasesDownwardPitch) {
  for (double yaw : {-1.2, 0., .8}) {
    for (double roll : {-.3, 0., .3}) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = .5;
      pose.position.y = .2;
      pose.position.z = 1.;
      tf2::Quaternion hand, axes;
      hand.setRPY(roll, 30 * M_PI / 180, yaw);
      axes.setRPY(0, -M_PI / 2, 0);
      pose.orientation = tf2::toMsg(hand * axes);
      auto target = tilt_tcp_target(pose, 5, 75);
      EXPECT_EQ(target.position, pose.position);
      EXPECT_NEAR(*tcp_pitch(target.orientation), 35 * M_PI / 180, 1e-9);
      EXPECT_NEAR(*tcp_yaw(target.orientation), yaw, 1e-9);
      EXPECT_NEAR(*tcp_roll(target.orientation), roll, 1e-9);
      EXPECT_NEAR(*tcp_pitch(tilt_tcp_target(pose, 11, 75).orientation),
          41 * M_PI / 180, 1e-9);
      EXPECT_THROW(tilt_tcp_target(pose, 5, 32), std::invalid_argument);
    }
  }
}

TEST_F(ApproachFixture, RefinementMissingOrOldInputNeverTilts) {
  config.refine_wait = .02;
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  wrist(false, "NO_CANDIDATE: empty", .54, .1);
  refinement->tick();
  std::this_thread::sleep_for(25ms);
  refinement->tick();
  EXPECT_EQ(refinement->result().code, "WRIST_INPUT_MISSING");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementWaitsForGripperBeforeTilting) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    feedback(.8, .1);
    wrist(false);
    refinement->tick();
  }
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(refinement->running());
  EXPECT_NE(refinement->result().detail.find("Waiting for gripper"), std::string::npos);
  for (int i = 0; i < 3; ++i) {
    feedback();
    wrist(false);
    refinement->tick();
  }
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps.front().name, "TILT_TO_OBSERVE");
}

TEST_F(ApproachFixture, CurrentLimitedGripperDoesNotAddRefinementSettleWait) {
  tcp_for_refinement();
  robot.require_passive_gripper_stationary = false;
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    feedback(.8, .1);
    wrist(false);
    refinement->tick();
  }
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps.front().name, "TILT_TO_OBSERVE");
}

TEST_F(ApproachFixture, RefinementGripperWaitIsBounded) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  const auto started = Clock::now();
  while (refinement->running() && elapsed(started) < 2.5) {
    feedback(.8, .1);
    refinement->tick();
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_EQ(refinement->result().code, "ROBOT_NOT_READY");
  EXPECT_FALSE(refinement->running());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementWaitsForPositionWindowAfterGripperShift) {
  tcp_for_refinement();
  robot.gripper_stop_mode = "position_window";
  robot.gripper_stop_window = .03;
  robot.gripper_completion_window = .03;
  robot.gripper_stop_position_tolerance = .004;
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 6; ++i) {
    feedback(i % 2 ? .804602 : .8);
    wrist(false);
    refinement->tick();
    std::this_thread::sleep_for(8ms);
  }
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(refinement->running());
  EXPECT_NE(refinement->result().detail.find("not stationary"), std::string::npos);
  for (int i = 0; i < 12 && executor.steps.empty(); ++i) {
    feedback(.804602);
    wrist(false);
    refinement->tick();
    std::this_thread::sleep_for(8ms);
  }
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps.front().name, "TILT_TO_OBSERVE");
}

TEST_F(ApproachFixture, RefinementCanCancelWhileWaitingForGripper) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  feedback(.8, .1);
  refinement->tick();
  refinement->cancel();
  refinement->tick();
  EXPECT_EQ(refinement->result().state, "CANCELLED");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementGripperWaitDoesNotHideStaleFeedback) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  feedback(.8, .1);
  refinement->tick();
  robot.state_max_age = .001;
  std::this_thread::sleep_for(5ms);
  refinement->tick();
  EXPECT_EQ(refinement->result().code, "ROBOT_STATE_INVALID");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementRepeatsUntilCeilingAndClampsLastIncrement) {
  tcp_for_refinement();
  config.refine_pitch_ceiling_deg = 12;
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int attempt = 0; attempt < 4; ++attempt) {
    for (int sample = 0; sample < 3; ++sample) {
      feedback();
      wrist(false);
      refinement->tick();
    }
    if (attempt < 3) {
      ASSERT_EQ(executor.steps.size(), size_t(attempt + 1));
      const auto & step = executor.steps.back();
      EXPECT_EQ(step.type, "linear");
      EXPECT_FALSE(step.position_only);
      EXPECT_NEAR(step.target[0], .5, 1e-8);
      EXPECT_NEAR(step.target[1], 0, 1e-8);
      EXPECT_NEAR(step.target[2], 1, 1e-8);
      geometry_msgs::msg::Quaternion q;
      q.x = step.target[3];
      q.y = step.target[4];
      q.z = step.target[5];
      q.w = step.target[6];
      EXPECT_NEAR(*tcp_pitch(q), std::min((attempt + 1) * 5, 12) * M_PI / 180, 1e-8);
      complete();
      refinement->tick();
    }
  }
  EXPECT_EQ(refinement->result().code, "PITCH_LIMIT");
  EXPECT_EQ(executor.steps.size(), 3u);
}

TEST_F(ApproachFixture, RefinementAmbiguousInputTiltsWithoutSelectingCandidate) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(false, "RIM_UNCERTAIN: ambiguous cup candidates");
    refinement->tick();
  }
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(context->snapshot().phase, "TILT_TO_OBSERVE");
  refinement->cancel();
  refinement->tick();
  EXPECT_EQ(refinement->result().state, "CANCELLED");
  EXPECT_EQ(executor.steps.size(), 1u);
}

TEST_F(ApproachFixture, AdvancingStabilityKeepsAngleUntilRimConfirmed) {
  tcp_for_refinement();
  config.refine_view_wait = .01;
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  wrist(false, "RIM_UNCERTAIN: ambiguous cup candidates");
  refinement->tick();
  wrist(false, "RIM_UNSTABLE: 1/3");
  refinement->tick();
  wrist(false, "RIM_UNSTABLE: 2/3");
  refinement->tick();
  EXPECT_TRUE(executor.steps.empty());
  std::this_thread::sleep_for(12ms);
  wrist(true); // Still lacks three stable observations in RobotContext.
  refinement->tick();
  EXPECT_TRUE(executor.steps.empty());
  for (int i = 0; i < 2; ++i) {
    wrist(true);
    refinement->tick();
  }
  ASSERT_EQ(scene.scenes.size(), 1u);
  refinement->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps[0].name, "CORRECT_TO_CUP_CENTER");
}

TEST_F(ApproachFixture, TransformErrorsNeverTriggerTimedViewChange) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 5; ++i) {
    wrist(false, "TF_UNAVAILABLE");
    refinement->tick();
  }
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementPreviewNeverTiltsAndCeilingBlocksMotion) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, false));
  for (int i = 0; i < 3; ++i) {
    wrist(false);
    refinement->tick();
  }
  EXPECT_FALSE(refinement->running());
  EXPECT_TRUE(executor.steps.empty());
  config.refine_pitch_ceiling_deg = 4;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(false);
    refinement->tick();
  }
  ASSERT_EQ(executor.steps.size(), 1u);
  complete();
  refinement->tick();
  for (int i = 0; i < 3; ++i) {
    wrist(false);
    refinement->tick();
  }
  EXPECT_EQ(refinement->result().code, "PITCH_LIMIT");
  EXPECT_EQ(executor.steps.size(), 1u);
}

TEST_F(ApproachFixture, StabilityProgressDoesNotResetOverallSearchDeadline) {
  tcp_for_refinement();
  config.refine_search_timeout = .02;
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  wrist(false, "RIM_UNSTABLE: 1/3");
  refinement->tick();
  std::this_thread::sleep_for(25ms);
  wrist(false, "RIM_UNSTABLE: 2/3");
  refinement->tick();
  EXPECT_EQ(refinement->result().code, "RIM_SEARCH_TIMEOUT");
  EXPECT_FALSE(refinement->running());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementStopsAfterReleaseWithoutWithdrawal) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(true);
    refinement->tick();
  }
  ASSERT_EQ(scene.scenes.size(), 1u);
  EXPECT_NEAR(scene.scenes[0].world.collision_objects[0].primitive_poses[0].position.z,
      .9 - config.cup_height / 2,
      1e-9);
  refinement->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  const auto & step = executor.steps[0];
  EXPECT_EQ(step.type, "pose");
  EXPECT_NEAR(step.orientation_slack, 10 * M_PI / 180, 1e-9);
  EXPECT_DOUBLE_EQ(step.tcp_path_margin, .04);
  EXPECT_DOUBLE_EQ(step.planning_time, 3.0);
  ASSERT_TRUE(step.tcp_path_min_z);
  EXPECT_NEAR(*step.tcp_path_min_z, .915, 1e-9);
  EXPECT_NEAR(step.target[0], .54, 1e-9);
  EXPECT_NEAR(step.target[2], .97, 1e-9);
  EXPECT_NEAR(step.target[4], -std::sin(M_PI / 4), 1e-9);
  wrist(false); // Occlusion after target capture does not change this short correction.
  complete();
  refinement->tick();
  ASSERT_EQ(executor.steps.size(), 2u);
  EXPECT_EQ(executor.steps[1].type, "gripper");
  EXPECT_EQ(executor.steps[1].target[0], 0.0);
  complete();
  refinement->tick();
  EXPECT_TRUE(refinement->result().sequence_completed) << refinement->result().detail;
  EXPECT_EQ(executor.steps.size(), 2u);
}

TEST_F(ApproachFixture, WithdrawalNeedsNoCupObservationAndKeepsGripperUnchanged) {
  tcp_for_refinement("left", .54, 0, .97);
  std::string reason;
  ASSERT_TRUE(withdrawal->start("left", "cup", reason, true));
  withdrawal->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps[0].name, "LIFT_FROM_CUP");
  EXPECT_EQ(executor.steps[0].type, "linear");
  EXPECT_NEAR(executor.steps[0].target[0], .54, 1e-9);
  EXPECT_NEAR(executor.steps[0].target[2], 1.07, 1e-9);
  EXPECT_DOUBLE_EQ(executor.steps[0].velocity_scale, config.refine_correction_speed);
  EXPECT_FALSE(withdrawal->result().sequence_completed);
  // The second target follows actual settled feedback, not the previous command.
  complete();
  tcp_for_refinement("left", .539, .001, 1.069);
  withdrawal->tick();
  ASSERT_EQ(executor.steps.size(), 2u);
  EXPECT_EQ(executor.steps[1].name, "RETREAT_FROM_CUP");
  EXPECT_EQ(executor.steps[1].type, "linear");
  EXPECT_NEAR(executor.steps[1].target[0], .459, 1e-9);
  EXPECT_NEAR(executor.steps[1].target[1], .001, 1e-9);
  EXPECT_NEAR(executor.steps[1].target[2], 1.069, 1e-9);
  EXPECT_NEAR(executor.steps[1].target[4], -std::sin(M_PI / 4), 1e-9);
  EXPECT_FALSE(withdrawal->result().sequence_completed);
  complete();
  withdrawal->tick();
  EXPECT_TRUE(withdrawal->result().sequence_completed) << withdrawal->result().detail;
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, WithdrawalPreviewNeverMovesOrChangesScene) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(withdrawal->start("left", "cup", reason, false));
  withdrawal->tick();
  EXPECT_TRUE(withdrawal->result().sequence_completed);
  EXPECT_NEAR(withdrawal->approach_target().pose.position.x, .42, 1e-9);
  EXPECT_NEAR(withdrawal->approach_target().pose.position.z, 1.10, 1e-9);
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, BothHandsWithdrawLeftThenRightWithoutGripperCommands) {
  tcp_for_refinement("left", .54, .2, .97);
  tcp_for_refinement("right", .55, -.2, .96);
  std::string reason;
  ASSERT_TRUE(withdrawal->start("both", "cup", reason, true));
  withdrawal->tick();
  complete();
  withdrawal->tick();
  complete();
  withdrawal->tick(); // Left finished; yield before the right hand.
  EXPECT_FALSE(withdrawal->result().sequence_completed);
  ASSERT_EQ(executor.steps.size(), 2u);
  withdrawal->tick();
  complete();
  withdrawal->tick();
  complete();
  withdrawal->tick();
  ASSERT_EQ(executor.steps.size(), 4u);
  const std::vector<std::string> arms{"left", "left", "right", "right"};
  for (size_t i = 0; i < arms.size(); ++i) {
    EXPECT_EQ(executor.steps[i].arm, arms[i]);
    EXPECT_EQ(executor.steps[i].type, "linear");
  }
  EXPECT_NEAR(executor.steps[0].target[2], 1.07, 1e-9);
  EXPECT_NEAR(executor.steps[1].target[0], .46, 1e-9);
  EXPECT_NEAR(executor.steps[2].target[2], 1.06, 1e-9);
  EXPECT_NEAR(executor.steps[3].target[0], .47, 1e-9);
  EXPECT_TRUE(withdrawal->result().sequence_completed);
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, BothHandsWithdrawalRejectsPreviewWithoutMotion) {
  std::string reason;
  EXPECT_FALSE(withdrawal->start("both", "cup", reason, false));
  EXPECT_FALSE(reason.empty());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, FailedOrCancelledWithdrawalNeverContinues) {
  for (size_t failed_step : {1u, 2u}) {
    for (bool cancel : {false, true}) {
      executor.steps.clear();
      tcp_for_refinement();
      tcp_for_refinement("right");
      std::string reason;
      ASSERT_TRUE(withdrawal->start("both", "cup", reason, true));
      withdrawal->tick();
      if (failed_step == 2) {
        tcp_for_refinement("left", .5, 0, 1.1);
        complete();
        withdrawal->tick();
      }
      ASSERT_EQ(executor.steps.size(), failed_step);
      if (cancel) withdrawal->cancel();
      else executor.current = {ExecutionState::Failed, "EXECUTION_FAILED", "withdrawal failed"};
      withdrawal->tick();
      EXPECT_EQ(executor.steps.size(), failed_step);
      EXPECT_FALSE(withdrawal->result().sequence_completed);
      EXPECT_EQ(withdrawal->result().state, cancel ? "CANCELLED" : "FAILED");
    }
  }
}

TEST_F(ApproachFixture, WithdrawalWithoutTcpDoesNotMove) {
  std::string reason;
  ASSERT_TRUE(withdrawal->start("left", "cup", reason, true));
  withdrawal->tick();
  EXPECT_EQ(withdrawal->result().code, "TCP_TF_MISSING");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, DownwardRetreatIsRejectedBeforeAnyWithdrawal) {
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = robot.planning_frame;
  tf.child_frame_id = robot.arms.at("left").tcp;
  tf.header.stamp = node->now();
  tf.transform.translation.z = 1;
  tf.transform.rotation.x = 1; // +TCP Z points down.
  tf.transform.rotation.w = 0;
  broadcaster->sendTransform(tf);
  for (int i = 0; i < 10; ++i) {
    ros.spin_some();
    std::this_thread::sleep_for(2ms);
  }
  std::string reason;
  ASSERT_TRUE(withdrawal->start("left", "cup", reason, true));
  withdrawal->tick();
  EXPECT_EQ(withdrawal->result().code, "RETREAT_DIRECTION");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, SeventyFiveDegreeRimCanProduceCorrectionPreview) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, false));
  for (int i = 0; i < 3; ++i) {
    wrist(true, "VALID_RIM", .54, 0, "left", 75.0 / 360.0);
    refinement->tick();
  }
  EXPECT_TRUE(refinement->result().sequence_completed) << refinement->result().detail;
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, BelowSeventyFiveDegreeRimCannotAuthorizeCorrection) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, false));
  for (int i = 0; i < 3; ++i) {
    wrist(true, "VALID_RIM", .54, 0, "left", 74.0 / 360.0);
    refinement->tick();
  }
  EXPECT_FALSE(refinement->result().sequence_completed);
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, WristStabilityAcceptsFourHundredMillisecondCadence) {
  const auto stamp = node->now();
  const auto received = Clock::now();
  for (const auto & source : {"head", "wrist_left"}) {
    msg::CupObservation input;
    input.header.frame_id = robot.planning_frame;
    input.source = source;
    input.cup_id = "cup";
    input.observation_type = std::string(source) == "head" ? 1 : 2;
    input.valid = true;
    input.body_position.z = .9;
    input.inner_radius = config.inner_radius;
    input.position_error = .009;
    input.confidence = .28;
    for (int i = 0; i < 3; ++i) {
      const double age = .8 - i * .4;
      input.header.stamp = stamp - rclcpp::Duration::from_seconds(age);
      ASSERT_TRUE(context->observe_cup(input, received -
          std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(age))));
    }
    const auto observation = context->cup_observation("cup", source);
    ASSERT_TRUE(observation);
    EXPECT_EQ(observation->stable_frames, std::string(source) == "head" ? 1 : 3);
  }
}

TEST_F(ApproachFixture, CorrectionPlanningFailuresAreBoundedAndKeepTarget) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(true);
    refinement->tick();
  }
  refinement->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  const auto target = executor.steps.front().target;
  for (int i = 0; i < 3; ++i) {
    feedback();
    executor.current = {ExecutionState::Failed, "PLANNING_FAILED", "MoveIt error 99999"};
    refinement->tick();
    ASSERT_EQ(executor.steps.size(), size_t(std::min(i + 2, 3))) << refinement->result().detail;
    EXPECT_EQ(executor.steps.back().target, target);
  }
  EXPECT_EQ(executor.steps[1].planning_time, 5.0);
  EXPECT_EQ(executor.steps[1].tcp_path_margin, .04);
  EXPECT_EQ(executor.steps[2].tcp_path_margin, .06);
  EXPECT_EQ(refinement->result().code, "PLANNING_FAILED");
  EXPECT_FALSE(refinement->result().sequence_completed);
}

TEST_F(ApproachFixture, CorrectionPlanningFailureDoesNotRetryAfterJointMovement) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(true);
    refinement->tick();
  }
  refinement->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  sensor_msgs::msg::JointState moved;
  moved.header.stamp = node->now();
  moved.name = {"left_joint"};
  moved.position = {.1};
  moved.velocity = {0};
  context->observe_joints(moved);
  executor.current = {ExecutionState::Failed, "PLANNING_FAILED", "MoveIt error 99999"};
  refinement->tick();
  EXPECT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(refinement->result().code, "ROBOT_MOVED");
}

TEST_F(ApproachFixture, CombinedCorrectionRejectsExcessiveLoweringBeforeStart) {
  tcp_for_refinement();
  config.refine_lower_distance = .09;
  std::string reason;
  EXPECT_FALSE(refinement->start("left", "cup", reason, true));
  EXPECT_FALSE(reason.empty());
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, FailedOrCancelledCombinedCorrectionNeverOpensGripper) {
  for (bool cancel : {false, true}) {
    executor.steps.clear();
    tcp_for_refinement();
    std::string reason;
    ASSERT_TRUE(refinement->start("left", "cup", reason, true));
    for (int i = 0; i < 3; ++i) {
      wrist(true);
      refinement->tick();
    }
    refinement->tick();
    ASSERT_EQ(executor.steps.size(), 1u);
    if (cancel) {
      refinement->cancel();
    }
    executor.current = {
        cancel ? ExecutionState::Cancelled : ExecutionState::Failed, "TEST", "test"};
    refinement->tick();
    EXPECT_EQ(executor.steps.size(), 1u);
    EXPECT_FALSE(refinement->result().sequence_completed);
  }
}

TEST_F(ApproachFixture, RefinementPreviewAndCancellationNeverDispatchLateCommands) {
  tcp_for_refinement();
  std::string reason;
  ASSERT_TRUE(refinement->start("left", "cup", reason, false));
  for (int i = 0; i < 3; ++i) {
    wrist(true);
    refinement->tick();
  }
  EXPECT_TRUE(refinement->result().sequence_completed);
  EXPECT_TRUE(scene.scenes.empty());
  EXPECT_TRUE(executor.steps.empty());
  scene.finish = false;
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(true);
    refinement->tick();
  }
  refinement->cancel();
  scene.finish = true;
  refinement->tick();
  EXPECT_EQ(refinement->result().state, "CANCELLED");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, RefinementRejectsDistantTargetAndUnsupportedArm) {
  tcp_for_refinement();
  std::string reason;
  EXPECT_FALSE(refinement->start("right", "cup", reason, true));
  ASSERT_TRUE(refinement->start("left", "cup", reason, true));
  for (int i = 0; i < 3; ++i) {
    wrist(true, "", .8);
    refinement->tick();
  }
  EXPECT_EQ(refinement->result().code, "CORRECTION_TOO_FAR");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, PreviewBothArmsNeverSendsSceneOrMotion) {
  config.offset_xy = {.03, -.04};
  executor.stationary = false;
  scene.pending = true;
  for (const std::string arm : {"left", "right"}) {
    std::string reason;
    ASSERT_TRUE(motion->start(arm, "cup", reason, false)) << reason;
    capture_initial();
    EXPECT_TRUE(motion->result().sequence_completed);
    EXPECT_FALSE(motion->at_observation_pose());
    EXPECT_NEAR(motion->approach_target().pose.position.x, .57, 1e-9);
    EXPECT_NEAR(motion->approach_target().pose.position.y, -.04, 1e-9);
    EXPECT_NEAR(motion->approach_target().pose.position.z, .84 + config.cup_height, 1e-9);
  }
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, ReachesTargetOnceWithoutGripperOrAttachmentChanges) {
  ASSERT_TRUE(start("left"));
  advance("OBSERVATION_READY");
  EXPECT_TRUE(motion->at_observation_pose());
  EXPECT_TRUE(motion->result().sequence_completed);
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps[0].type, "pose");
  EXPECT_TRUE(executor.steps[0].position_only);
  EXPECT_EQ(executor.steps[0].arm, "left");
  EXPECT_NEAR(executor.steps[0].target[2], .84 + config.cup_height, 1e-9);
  ASSERT_EQ(scene.scenes.size(), 1u);
  EXPECT_TRUE(scene.scenes[0].robot_state.attached_collision_objects.empty());
  for (int i = 0; i < 10; ++i) {
    motion->tick();
  }
  EXPECT_EQ(executor.steps.size(), 1u);
  std::string reason;
  EXPECT_TRUE(motion->is_ready_to_start("right", "cup", reason));
}

TEST_F(ApproachFixture, OutsideRollBandApproachesDirectlyWithoutAlignment) {
  config.tcp_pitch_min_deg = 20;
  config.tcp_pitch_limit_deg = 60;
  config.tcp_pitch_downward_only = true;
  config.tcp_roll_limit_deg = 60;
  config.tcp_yaw_inward_only = true;
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = robot.planning_frame;
  tf.child_frame_id = robot.arms.at("left").tcp;
  tf.header.stamp = node->now();
  tf.transform.translation.x = .2;
  tf.transform.translation.z = 1.2;
  tf2::Quaternion roll, neutral;
  roll.setRPY(M_PI / 2, 0, 0);
  neutral.setRPY(0, -M_PI / 2, 0);
  tf.transform.rotation = tf2::toMsg(roll * neutral);
  broadcaster->sendTransform(tf);
  for (int i = 0; i < 20; ++i) {
    ros.spin_some();
    std::this_thread::sleep_for(2ms);
  }
  std::string reason;
  ASSERT_TRUE(context->pose_of(tf.child_frame_id));
  ASSERT_TRUE(motion->start("left", "cup", reason, true));
  advance("OBSERVATION_READY");
  ASSERT_TRUE(motion->result().sequence_completed) << motion->result().detail;
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(executor.steps[0].name, "MOVE_TO_OBSERVE");
  EXPECT_NEAR(executor.steps[0].tcp_roll_limit, M_PI / 3, 1e-9);
  EXPECT_TRUE(executor.steps[0].tcp_yaw_inward_only);
  EXPECT_NEAR(executor.steps[0].tcp_pitch_min, 20 * M_PI / 180, 1e-9);
  EXPECT_NEAR(executor.steps[0].tcp_pitch_limit, 60 * M_PI / 180, 1e-9);
  EXPECT_NEAR(executor.steps[0].target[2], .84 + config.cup_height, 1e-9);
}

TEST_F(ApproachFixture, ApproachRejectsStaleOrUncertainHeadWithoutCommands) {
  config.observation_timeout = .01;
  std::string reason;
  ASSERT_TRUE(motion->start("right", "cup", reason, false));
  observe("head", 0, true, .09, 3);
  config.max_age = .000001;
  std::this_thread::sleep_for(12ms);
  motion->tick();
  EXPECT_EQ(motion->result().code, "CUP_NOT_FOUND");
  config.max_age = .5;
  config.max_position_error = .08;
  ASSERT_TRUE(motion->start("right", "cup", reason, true));
  observe("head", 0, true, .09, 3);
  std::this_thread::sleep_for(12ms);
  motion->tick();
  EXPECT_EQ(motion->result().code, "CUP_NOT_FOUND");
  EXPECT_TRUE(scene.scenes.empty());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, HeadCollectorUsesExactlyThreeNewFramesWithoutNestedWarmup) {
  ASSERT_TRUE(start());
  for (int i = 0; i < 3; ++i) {
    observe("head", .003 * i, true, .09, 1);
    motion->tick();
    if (i < 2) {
      EXPECT_TRUE(scene.scenes.empty());
    }
  }
  ASSERT_EQ(scene.scenes.size(), 1u);
  EXPECT_NEAR(motion->estimated_cup_pose().pose.position.x, .543, 1e-9);
  EXPECT_TRUE(executor.steps.empty()); // Scene acknowledgement still precedes motion.
}

TEST_F(ApproachFixture, HeadTrackIsCollectedBeforeRequestsAndSurvivesOcclusion) {
  HeadCupObserver collector(*context, config);
  geometry_msgs::msg::PoseStamped estimate;
  double uncertainty = 0;
  std::string reason;
  for (int i = 0; i < 4; ++i) {
    observe("head", .003 * i, true, .09, 1);
    EXPECT_EQ(collector.poll("cup", estimate, uncertainty, reason), i >= 2);
  }
  EXPECT_NEAR(estimate.pose.position.x, .546, 1e-9);
  const auto stamp = estimate.header.stamp;
  const auto saved = context->tracked_head_cup("cup", config.max_age, reason);
  ASSERT_TRUE(saved);
  observe("head", 0, false, .09, 1, "Cup occluded during handover");
  ASSERT_TRUE(collector.poll("cup", estimate, uncertainty, reason));
  EXPECT_EQ(estimate.header.stamp, stamp);
  EXPECT_NEAR(estimate.pose.position.x, .546, 1e-9);
  EXPECT_EQ(context->tracked_head_cup("cup", config.max_age, reason)->received, saved->received);
  ASSERT_TRUE(motion->start("left", "cup", reason, false));
  motion->tick();
  EXPECT_TRUE(motion->result().sequence_completed);
  EXPECT_NEAR(motion->approach_target().pose.position.x, .546, 1e-9);
  EXPECT_TRUE(scene.scenes.empty());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, HeadTrackRetainsOldPositionUntilNewLocationIsConfirmed) {
  for (int i = 0; i < 3; ++i) observe("head");
  std::string reason;
  auto saved = context->tracked_head_cup("cup", config.max_age, reason);
  ASSERT_TRUE(saved);
  for (int i = 0; i < 3; ++i) {
    observe("head", .1);
    const auto current = context->tracked_head_cup("cup", config.max_age, reason);
    ASSERT_TRUE(current);
    EXPECT_NEAR(current->observation.body_position.x, i < 2 ? .54 : .64, 1e-9);
  }
  // Wrist RIM observations do not update the head BODY track.
  wrist(true, "VALID_RIM", .9);
  EXPECT_NEAR(context->tracked_head_cup("cup", config.max_age, reason)->observation.body_position.x,
      .64, 1e-9);
}

TEST_F(ApproachFixture, RepeatedReadsAndReplayCannotRefreshStoredHeadCup) {
  for (int i = 0; i < 3; ++i) observe("head");
  std::string reason;
  auto saved = context->tracked_head_cup("cup", config.max_age, reason);
  ASSERT_TRUE(saved);
  EXPECT_FALSE(context->observe_cup(saved->observation));
  const auto reread = context->tracked_head_cup("cup", config.max_age, reason);
  ASSERT_TRUE(reread);
  EXPECT_EQ(reread->received, saved->received);
  EXPECT_EQ(reread->observation.header.stamp, saved->observation.header.stamp);
  std::this_thread::sleep_for(2ms);
  EXPECT_FALSE(context->tracked_head_cup("cup", .000001, reason));
  EXPECT_NE(reason.find("expired"), std::string::npos);
}

TEST_F(ApproachFixture, InitialSamplesAreDistinctAndAveragedWithConservativeGeometry) {
  ASSERT_TRUE(start());
  observe("head", -.006);
  motion->tick();
  for (int i = 0; i < 5; ++i) {
    motion->tick(); // Repeated reads of the same image must not fill the sample window.
  }
  EXPECT_TRUE(scene.scenes.empty());
  observe("head", .003);
  motion->tick();
  EXPECT_TRUE(scene.scenes.empty());
  observe("head", .006);
  motion->tick();
  ASSERT_EQ(scene.scenes.size(), 1u);
  EXPECT_NEAR(motion->approach_target().pose.position.x, .541, 1e-9);
  const auto & cup = scene.scenes[0].world.collision_objects[0];
  EXPECT_NEAR(cup.primitive_poses[0].position.x, .541, 1e-9);
  EXPECT_NEAR(cup.primitives[0].dimensions[1], .08 + .005 + .097, 1e-9);
  EXPECT_TRUE(executor.steps.empty()); // Still awaiting scene acknowledgement.
}

TEST_F(ApproachFixture, InvalidInitialSampleResetsCollectionAndTimeoutSendsNothing) {
  ASSERT_TRUE(start());
  observe("head");
  motion->tick();
  observe("head", 0, false);
  motion->tick();
  for (int i = 0; i < 2; ++i) {
    observe("head");
    motion->tick();
  }
  EXPECT_TRUE(scene.scenes.empty());
  config.observation_timeout = .001;
  motion->tick();
  EXPECT_EQ(motion->result().code, "CUP_NOT_FOUND");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, FrozenCupIgnoresShiftUncertaintyAndLossAfterCapture) {
  ASSERT_TRUE(start());
  capture_initial();
  const auto saved = motion->approach_target();
  scene.finish = false;
  observe("head", .04, true, .14);
  motion->tick();
  EXPECT_TRUE(executor.steps.empty());
  observe("head", 0, false, .09, 1, "Cup occluded by hand");
  scene.finish = true;
  motion->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(scene.scenes.size(), 1u);
  EXPECT_DOUBLE_EQ(executor.steps[0].target[0], saved.pose.position.x);
  EXPECT_DOUBLE_EQ(executor.steps[0].target[2], saved.pose.position.z);
  EXPECT_EQ(context->snapshot().phase, "MOVE_TO_OBSERVE");
  complete();
  motion->tick();
  EXPECT_TRUE(motion->at_observation_pose());
  EXPECT_TRUE(motion->result().sequence_completed);
  EXPECT_EQ(executor.steps.size(), 1u);
}

TEST_F(ApproachFixture, FrozenCupDoesNotRequireFreshCameraButRequiresRobotFeedback) {
  ASSERT_TRUE(start());
  capture_initial();
  config.max_age = .000001;
  motion->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  robot.state_max_age = .000001;
  motion->tick();
  EXPECT_FALSE(motion->result().sequence_completed);
  EXPECT_EQ(executor.current.code, "ROBOT_STATE_INVALID");
  EXPECT_FALSE(motion->at_observation_pose());
}

TEST_F(ApproachFixture, InvalidRobotFeedbackBeforeDispatchStillBlocksMotion) {
  ASSERT_TRUE(start());
  capture_initial();
  robot.state_max_age = .000001;
  motion->tick();
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_EQ(motion->result().code, "ROBOT_NOT_READY");
}

TEST_F(ApproachFixture, NewRequestUsesLatestCupCollectedWhileIdle) {
  std::string reason;
  ASSERT_TRUE(motion->start("left", "cup", reason, false));
  capture_initial();
  const auto previous = motion->approach_target();
  for (int i = 0; i < 3; ++i) observe("head", .04);
  ASSERT_TRUE(motion->start("left", "cup", reason, false));
  motion->tick();
  EXPECT_TRUE(motion->result().sequence_completed);
  EXPECT_NEAR(motion->approach_target().pose.position.x, previous.pose.position.x + .04, 1e-9);
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, ApproachCancelDuringMotionDoesNotReleaseOrClaimArrival) {
  std::string reason;
  ASSERT_TRUE(motion->start("right", "cup", reason, true));
  capture_initial();
  motion->tick();
  ASSERT_EQ(executor.steps.size(), 1u);
  motion->cancel();
  motion->tick();
  EXPECT_EQ(motion->result().state, "CANCELLED");
  EXPECT_FALSE(motion->at_observation_pose());
  EXPECT_FALSE(motion->blocks_other_motion());
  EXPECT_EQ(executor.steps.size(), 1u);
}

TEST_F(ApproachFixture, FailedInitialSceneCannotDispatchMotion) {
  ASSERT_TRUE(start());
  capture_initial();
  ASSERT_EQ(scene.scenes.size(), 1u);
  scene.outcome = {ExecutionState::Failed, "SCENE_FAILED", "scene rejected"};
  motion->tick();
  EXPECT_EQ(motion->result().code, "SCENE_FAILED");
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_FALSE(motion->blocks_other_motion());
}

TEST_F(ApproachFixture, RejectsUnknownObservationTypeAndBodyFromWrist) {
  msg::CupObservation observation;
  observation.header.frame_id = robot.planning_frame;
  observation.header.stamp = node->now();
  observation.source = "right";
  observation.cup_id = "cup";
  observation.valid = true;
  observation.inner_radius = .08;
  observation.confidence = 1;
  EXPECT_FALSE(context->observe_cup(observation));
  observation.observation_type = msg::CupObservation::BODY;
  EXPECT_FALSE(context->observe_cup(observation));
  EXPECT_FALSE(context->cup_observation("cup", "right"));
}

TEST_F(ApproachFixture, CancelPendingSceneCannotDispatchLateMotion) {
  scene.finish = false;
  ASSERT_TRUE(start());
  capture_initial();
  motion->cancel();
  motion->tick();
  EXPECT_TRUE(motion->running());
  scene.finish = true;
  motion->tick();
  EXPECT_FALSE(motion->running());
  EXPECT_FALSE(motion->blocks_other_motion());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, SceneTimeoutBlocksOtherMotions) {
  ASSERT_TRUE(start());
  capture_initial();
  scene.outcome = {ExecutionState::StopUnconfirmed, "SCENE_UNCONFIRMED", "timeout"};
  motion->tick();
  EXPECT_TRUE(motion->blocks_other_motion());
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, ReplayCannotRefreshObservation) {
  observe("head");
  auto old = context->cup_observation("cup", "head");
  ASSERT_TRUE(old);
  EXPECT_FALSE(context->observe_cup(old->observation));
  EXPECT_EQ(context->cup_observation("cup", "head")->stable_frames, old->stable_frames);
}

TEST(PickGeometry, SideGraspUsesRimHeightAndHorizontalJawOrientation) {
  PickCupConfig c;
  geometry_msgs::msg::Point rim;
  rim.x = .6;
  rim.y = .2;
  rim.z = 1.0;
  auto p = cup_grasp_pose(rim, .165, c, std::atan2(.2, .6));
  EXPECT_NEAR(p.position.z, .9, 1e-9);
  EXPECT_NEAR(p.position.x, .6 + .03 * std::cos(std::atan2(.2, .6)), 1e-9);
  EXPECT_NEAR(*tcp_pitch(p.orientation), 0, 1e-9);
  EXPECT_NEAR(*tcp_yaw(p.orientation), std::atan2(.2, .6), 1e-9);
  EXPECT_TRUE(c.validate(.165).empty());
  c.stand_off = .9;
  EXPECT_FALSE(c.validate(.165).empty());
}

TEST(PickGeometry, HeadingUsesInitialTcpAndNearCupFallback) {
  geometry_msgs::msg::Pose tcp;
  tcp.position.x = .2;
  tcp.position.y = .3;
  tf2::Quaternion q;
  q.setRPY(0, -M_PI / 2, -.4);
  tcp.orientation = tf2::toMsg(q);
  geometry_msgs::msg::Point cup;
  cup.x = .6;
  cup.y = .1;
  EXPECT_NEAR(*cup_approach_yaw(tcp, cup), std::atan2(-.2, .4), 1e-9);
  cup.x = .21;
  cup.y = .31;
  EXPECT_NEAR(*cup_approach_yaw(tcp, cup), -.4, 1e-9);
  tcp.orientation = geometry_msgs::msg::Quaternion{};
  EXPECT_FALSE(cup_approach_yaw(tcp, cup));
}

TEST_F(ApproachFixture, PickPreviewNeverOpensGripperOrChangesScene) {
  tcp_for_refinement();
  shoulder_for_pick();
  PickCupConfig pc;
  PickCupMotion pick(robot, config, pc, *context, executor, scene);
  std::string reason;
  ASSERT_TRUE(pick.start("left", "cup", reason, false));
  for (int i = 0; i < 10 && pick.running(); ++i) {
    observe("head");
    pick.tick();
  }
  EXPECT_TRUE(pick.result().sequence_completed) << pick.result().detail;
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(scene.scenes.empty());
}

TEST_F(ApproachFixture, PickHeadingIsFrozenWhileCollectingCupSamples) {
  tcp_for_refinement("left", .2, .3);
  PickCupConfig pc;
  PickCupMotion pick(robot, config, pc, *context, executor, scene);
  std::string reason;
  ASSERT_TRUE(pick.start("left", "cup", reason, false));
  pick.tick();
  tcp_for_refinement("left", .8, -.2);
  for (int i = 0; i < 10 && pick.running(); ++i) {
    observe("head");
    pick.tick();
  }
  ASSERT_TRUE(pick.result().sequence_completed) << pick.result().detail;
  EXPECT_NEAR(*tcp_yaw(pick.approach_target().pose.orientation), std::atan2(-.3, .44), 1e-9);
}

TEST_F(ApproachFixture, PickApproachStopsAfterOpenAndPregrasp) {
  tcp_for_refinement("left");
  relaxed_pick_arrival = true;
  PickCupConfig pc;
  PickCupMotion pick(robot, config, pc, *context, executor, scene);
  std::string reason;
  ASSERT_TRUE(pick.start("left", "cup", reason, true));
  for (int i = 0; i < 30 && pick.running(); ++i) {
    feedback(0);
    observe("head");
    complete();
    pick.tick();
  }
  ASSERT_TRUE(pick.result().sequence_completed) << pick.result().detail;
  ASSERT_EQ(executor.steps.size(), 2u);
  EXPECT_EQ(executor.steps[0].name, "PICK_OPEN");
  EXPECT_EQ(executor.steps[0].target, std::vector<double>{0.0});
  EXPECT_EQ(executor.steps[1].name, "PICK_PREGRASP");
  EXPECT_TRUE(executor.steps[1].position_only);
  EXPECT_NEAR(executor.steps[1].tcp_roll_limit, 10 * M_PI / 180, 1e-9);
  EXPECT_NEAR(executor.steps[1].tcp_pitch_limit, 10 * M_PI / 180, 1e-9);
  EXPECT_NEAR(executor.steps[1].tcp_yaw_limit, 15 * M_PI / 180, 1e-9);
  EXPECT_NEAR(pick.approach_target().pose.position.x, .45, 1e-8);
  EXPECT_NEAR(pick.approach_target().pose.position.z, .84 - .165 / 2 + .065, 1e-8);
  EXPECT_FALSE(pick.blocks_other_motion());
  for (int i = 0; i < 5; ++i) {
    pick.tick();
  }
  EXPECT_EQ(executor.steps.size(), 2u);
  for (const auto & update : scene.scenes) {
    EXPECT_TRUE(update.robot_state.attached_collision_objects.empty());
    for (const auto & row : update.allowed_collision_matrix.entry_values) {
      EXPECT_TRUE(std::none_of(row.enabled.begin(), row.enabled.end(), [](bool value) {
        return value;
      }));
    }
  }
}

TEST_F(ApproachFixture, PickCancelledDuringObservationDoesNotOpenGripper) {
  tcp_for_refinement();
  shoulder_for_pick();
  PickCupConfig pc;
  PickCupMotion pick(robot, config, pc, *context, executor, scene);
  std::string reason;
  ASSERT_TRUE(pick.start("left", "cup", reason, true));
  pick.tick();
  pick.cancel();
  pick.tick();
  EXPECT_EQ(pick.result().state, "CANCELLED");
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ApproachFixture, ActionDefaultsToPreviewAndRejectsConcurrentRequests) {
  config.observation_timeout = 5;
  HandoverMotion handover(robot, *context, executor);
  shoulder_for_pick();
  PickCupConfig pc;
  PickCupMotion pick(robot, config, pc, *context, executor, scene);
  TaskManager manager(*node,
      *context,
      handover,
      {{"/approach_cup", *motion}, {"/refine_cup", *refinement}, {"/pick_cup", pick}});
  auto pick_client = rclcpp_action::create_client<action::ApproachCup>(node, "/pick_cup");
  auto client = rclcpp_action::create_client<action::ApproachCup>(node, "/approach_cup");
  auto refine_client = rclcpp_action::create_client<action::ApproachCup>(node, "/refine_cup");
  auto handover_client = rclcpp_action::create_client<action::Handover>(node, "/handover");
  auto until = [&](auto condition) {
    const auto start = Clock::now();
    while (elapsed(start) < 2) {
      feedback();
      ros.spin_some();
      if (condition()) {
        return true;
      }
      std::this_thread::sleep_for(5ms);
    }
    return false;
  };
  ASSERT_TRUE(until([&] {
    return client->action_server_is_ready() && handover_client->action_server_is_ready();
  }));
  action::ApproachCup::Goal goal;
  EXPECT_FALSE(goal.execute);
  goal.arm = "left";
  goal.cup_id = "cup";
  auto sent = client->async_send_goal(goal);
  ASSERT_TRUE(until([&] {
    return sent.wait_for(0s) == std::future_status::ready;
  }));
  auto handle = sent.get();
  ASSERT_TRUE(handle);
  EXPECT_FALSE(manager.can_shutdown());
  auto concurrent = client->async_send_goal(goal);
  ASSERT_TRUE(until([&] {
    return concurrent.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_FALSE(concurrent.get());
  auto refine_busy = refine_client->async_send_goal(goal);
  ASSERT_TRUE(until([&] {
    return refine_busy.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_FALSE(refine_busy.get());
  auto pick_busy = pick_client->async_send_goal(goal);
  ASSERT_TRUE(until([&] {
    return pick_busy.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_FALSE(pick_busy.get());
  action::Handover::Goal handover_goal;
  handover_goal.direction = "left_to_right";
  auto other = handover_client->async_send_goal(handover_goal);
  ASSERT_TRUE(until([&] {
    return other.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_FALSE(other.get());
  auto result = client->async_get_result(handle);
  ASSERT_TRUE(until([&] {
    observe("head");
    return result.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_TRUE(executor.steps.empty());
  EXPECT_TRUE(scene.scenes.empty());
  EXPECT_TRUE(manager.can_shutdown());
  tcp_for_refinement();
  auto refine_sent = refine_client->async_send_goal(goal);
  ASSERT_TRUE(until([&] {
    return refine_sent.wait_for(0s) == std::future_status::ready;
  }));
  auto refine_handle = refine_sent.get();
  ASSERT_TRUE(refine_handle);
  EXPECT_TRUE(refinement->running());
  EXPECT_FALSE(motion->running());
  auto approach_busy = client->async_send_goal(goal);
  ASSERT_TRUE(until([&] {
    return approach_busy.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_FALSE(approach_busy.get());
  auto refine_result = refine_client->async_get_result(refine_handle);
  manager.request_shutdown();
  ASSERT_TRUE(until([&] {
    return refine_result.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_TRUE(manager.can_shutdown());
  EXPECT_TRUE(executor.steps.empty());
}

} // namespace
} // namespace demo_1f
