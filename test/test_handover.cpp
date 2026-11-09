#include "demo_1f/runtime/config.hpp"
#include "demo_1f/runtime/execution/trajectory_executor.hpp"
#include "demo_1f/tasks/task_manager.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <thread>

using namespace std::chrono_literals;

namespace demo_1f {
namespace {
class RosEnvironment : public ::testing::Environment {
  void SetUp() override {
    initialize_moveit_logging();
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }

  void TearDown() override {
    rclcpp::shutdown();
  }
};

auto * environment = ::testing::AddGlobalTestEnvironment(new RosEnvironment);

TEST(PlanningConfig, CandidateCountIsBoundedAndReadOnly) {
  auto defaults = std::make_shared<rclcpp::Node>("planning_config_default");
  EXPECT_EQ(load_config(*defaults).ompl_planning_attempts, 3);
  EXPECT_TRUE(defaults->describe_parameter("ompl_planning_attempts").read_only);
  for (int count : {0, 1, 3, 8, 9}) {
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("ompl_planning_attempts", count)});
    auto node = std::make_shared<rclcpp::Node>("planning_config_override", options);
    if (count == 0 || count == 9) {
      EXPECT_THROW(load_config(*node), std::invalid_argument);
    } else {
      EXPECT_EQ(load_config(*node).ompl_planning_attempts, count);
    }
  }
}

Config fixture_config(const std::string & prefix) {
  Config c;
  c.require_tcp_tf = false;
  c.state_max_age = 0.15;
  c.settle_time = 0.02;
  c.cancel_timeout = 0.15;
  c.gripper_stop_window = 0.04;
  c.gripper_stop_max_gap = 0.025;
  c.gripper_transport = "action";
  c.joint_states_topic = prefix + "/joint_states";
  c.move_group_action = prefix + "/move_action";
  for (const std::string name : {"left", "right"}) {
    ArmConfig arm;
    arm.name = name;
    arm.group = name;
    arm.tcp = prefix.substr(1) + "_" + name + "_tcp";
    arm.gripper_joint = name + "_gripper";
    arm.trajectory_action = prefix + "/" + name + "/follow_joint_trajectory";
    arm.trajectory_topic = prefix + "/" + name + "/joint_trajectory";
    for (int i = 0; i < 7; ++i) {
      arm.joints.push_back(name + "_joint" + std::to_string(i));
    }
    c.arms.emplace(name, arm);
  }
  for (const std::string direction : {"left_to_right", "right_to_left"}) {
    MotionSequence sequence;
    sequence.direction = direction;
    sequence.giver = direction == "left_to_right" ? "left" : "right";
    sequence.receiver = direction == "left_to_right" ? "right" : "left";
    sequence.steps = {
        {"present", "joint", sequence.giver, std::vector<double>(7, 0.1), "", 0.03, 1.0, 0.01},
        {"receiver_close", "gripper", sequence.receiver, {0.4}, "", 0.03, 1.0, 0.01},
        {"giver_release", "gripper", sequence.giver, {0.0}, "", 0.03, 1.0, 0.01},
        {"retreat", "joint", sequence.giver, std::vector<double>(7, 0.2), "", 0.03, 1.0, 0.01}};
    c.motion_sequences.emplace(direction, sequence);
  }
  return c;
}

void configure_five_step_sequences(Config & config) {
  for (auto & [direction, sequence] : config.motion_sequences) {
    (void)direction;
    Step ready;
    ready.name = "face_each_other";
    ready.type = "dual_joint";
    ready.timeout = 1.0;
    ready.left_target = std::vector<double>(7, 0.1);
    ready.right_target = std::vector<double>(7, -0.1);
    Step approach;
    approach.name = "open_approach";
    approach.type = "dual_joint_gripper";
    approach.arm = sequence.receiver;
    approach.left_target = std::vector<double>(7, 0.2);
    approach.right_target = std::vector<double>(7, -0.2);
    approach.gripper_target = {0.0};
    approach.timeout = 1.0;
    approach.tolerance = 0.017453292519943295;
    approach.planning_tolerance = 0.0005;
    Step close = sequence.steps[1];
    Step release = sequence.steps[2];
    Step retreat = ready;
    retreat.name = "retreat_together";
    retreat.left_target = std::vector<double>(7, 0.3);
    retreat.right_target = std::vector<double>(7, -0.3);
    sequence.steps = {ready, approach, close, release, retreat};
  }
}

class FakeExecutor : public ExecutorPort {
public:
  bool ready(const MotionSequence &, std::string &) const override {
    return true;
  }

  bool start(const Step & step) override {
    steps.push_back(step);
    current = {ExecutionState::Running, {}, {}};
    return true;
  }

  bool start_sequence(const std::vector<Step> & sequence) override {
    ++continuous_starts;
    steps.insert(steps.end(), sequence.begin(), sequence.end());
    current = {ExecutionState::Running, {}, {}, sequence.front().name};
    return true;
  }

  void tick() override {}

  void cancel(const std::string & code, const std::string & detail) override {
    ++cancel_count;
    current = {
        code == "CANCELLED" ? ExecutionState::Cancelled : ExecutionState::Failed, code, detail};
  }

  ExecutionResult result() const override {
    return current;
  }

  ExecutionResult current;
  std::vector<Step> steps;
  int cancel_count{0};
  int continuous_starts{0};
};

class ContextFixture : public ::testing::Test {
protected:
  void SetUp() override {
    static int sequence = 0;
    prefix = "/handover_test_" + std::to_string(++sequence);
    node = std::make_shared<rclcpp::Node>(prefix.substr(1));
    config = fixture_config(prefix);
    context = std::make_unique<RobotContext>(*node, config);
    for (const auto & [name, arm] : config.arms) {
      for (const auto & joint : arm.joints) {
        positions[joint] = 0.0;
      }
      positions[arm.gripper_joint] = 0.2;
    }
    refresh();
  }

  sensor_msgs::msg::JointState sample() {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = node->now();
    for (const auto & [joint, position] : positions) {
      msg.name.push_back(joint);
      msg.position.push_back(position);
      msg.velocity.push_back(velocities.count(joint) ? velocities.at(joint) : 0.0);
    }
    return msg;
  }

  void refresh() {
    context->observe_joints(sample());
  }

  void feed_for(double duration, const std::function<void(int)> & update = [](int) {}) {
    const auto start = Clock::now();
    int count = 0;
    do {
      update(count++);
      refresh();
      std::this_thread::sleep_for(5ms);
    } while (elapsed(start) < duration);
    refresh();
  }

  std::string prefix;
  std::shared_ptr<rclcpp::Node> node;
  Config config;
  std::unique_ptr<RobotContext> context;
  std::map<std::string, double> positions;
  std::map<std::string, double> velocities;
};

TEST_F(ContextFixture, NormalCompletionWindowDoesNotWeakenStopConfirmation) {
  config.gripper_stop_window = .12;
  config.gripper_completion_window = .04;
  const auto after = Clock::now();
  positions["right_gripper"] = .8;
  feed_for(.07);
  std::string reason;
  auto joint = context->snapshot().joints.at("right_gripper");
  EXPECT_TRUE(context->joint_stationary("right_gripper", joint, reason, after, true)) << reason;
  EXPECT_FALSE(context->joint_stationary("right_gripper", joint, reason, after, false));
  feed_for(.1);
  joint = context->snapshot().joints.at("right_gripper");
  EXPECT_TRUE(context->joint_stationary("right_gripper", joint, reason, after, false)) << reason;
  positions["right_gripper"] += .03;
  refresh();
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason, after, true));
  std::this_thread::sleep_for(35ms);
  refresh();
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason, after, true));
}

TEST_F(ContextFixture, GripperPositionWindowAcceptsRecordedJitterButArmVelocityStillBlocks) {
  std::string reason;
  const auto stopped = [&](const std::string & joint) {
    return context->joint_stationary(joint, context->snapshot().joints.at(joint), reason);
  };
  EXPECT_FALSE(stopped("right_gripper")); // one sample cannot confirm a stop
  feed_for(0.08, [&](int i) {
    positions["right_gripper"] = (i % 2 ? 1 : -1) * 0.0015339807878856412;
    velocities["right_gripper"] = i % 2 ? -0.083566 : 0.214885;
  });
  EXPECT_TRUE(stopped("right_gripper")) << reason;
  EXPECT_DOUBLE_EQ(
      context->snapshot().joints.at("right_gripper").velocity.value(), velocities["right_gripper"]);
  EXPECT_TRUE(context->capture_targets(reason).has_value()) << reason;
  velocities["right_joint0"] = 0.214885;
  refresh();
  EXPECT_FALSE(stopped("right_joint0"));
  EXPECT_FALSE(context->capture_targets(reason).has_value());
  config.gripper_stop_mode = "velocity";
  EXPECT_FALSE(stopped("right_gripper"));
}

TEST_F(ContextFixture, GripperPositionMotionCannotPassWithZeroReportedVelocity) {
  std::string reason;
  feed_for(0.08, [&](int i) {
    positions["right_gripper"] = 0.001 * i;
  });
  auto observed = context->snapshot().joints.at("right_gripper");
  ASSERT_TRUE(observed.position_window_span.has_value());
  EXPECT_FALSE(context->joint_stationary("right_gripper", observed, reason));
  // Returning to the original position must not conceal an excursion.
  feed_for(0.06, [&](int) {
    positions["right_gripper"] = 0.2;
  });
  positions["right_gripper"] = 0.23;
  refresh();
  std::this_thread::sleep_for(5ms);
  positions["right_gripper"] = 0.2;
  refresh();
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason));
}

TEST_F(ContextFixture, GripperStopNeedsContinuousFreshSamplesAfterCancellation) {
  std::string reason;
  feed_for(0.06);
  EXPECT_TRUE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason));
  const auto cancelled = Clock::now();
  refresh();
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason, cancelled));
  feed_for(0.06);
  EXPECT_TRUE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason, cancelled));
  std::this_thread::sleep_for(35ms); // a gap invalidates the old window
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason));
  refresh();
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason));
  auto replay = sample();
  context->observe_joints(replay);
  for (int i = 0; i < 10; ++i) {
    context->observe_joints(replay);
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason));
  feed_for(0.06);
  config.state_max_age = 0.01;
  std::this_thread::sleep_for(15ms);
  EXPECT_FALSE(context->joint_stationary(
      "right_gripper", context->snapshot().joints.at("right_gripper"), reason));
}

TEST_F(ContextFixture, PassiveGripperPolicyKeepsFreshnessArmAndCancellationChecks) {
  config.require_passive_gripper_stationary = false;
  config.gripper_stop_mode = "velocity";
  velocities["right_gripper"] = .2;
  velocities["left_joint0"] = .2;
  positions["left_joint0"] = 0;
  refresh();
  std::string reason;
  const auto state = context->snapshot();
  EXPECT_TRUE(context->joint_ready_for_motion("right_gripper", state.joints.at("right_gripper"), reason));
  EXPECT_FALSE(context->joint_stationary("right_gripper", state.joints.at("right_gripper"), reason));
  EXPECT_FALSE(context->joint_ready_for_motion("left_joint0", state.joints.at("left_joint0"), reason));
  config.state_max_age = .001;
  std::this_thread::sleep_for(5ms);
  EXPECT_FALSE(context->joint_ready_for_motion("right_gripper", state.joints.at("right_gripper"), reason));
}

TEST_F(ContextFixture, PartialUpdatesDoNotRefreshMissingJoints) {
  config.state_max_age = 0.02;
  std::this_thread::sleep_for(25ms);
  auto msg = sample();
  msg.name = {"left_gripper"};
  msg.position = {0.3};
  msg.velocity = {0.0};
  context->observe_joints(msg);
  std::string reason;
  EXPECT_FALSE(context->ready(reason));
  EXPECT_NE(reason.find("joint"), std::string::npos);
  EXPECT_TRUE(context->joint_fresh(context->snapshot().joints.at("left_gripper")));
}

TEST_F(ContextFixture, InvalidOrReplayedSamplesCannotOverwriteObservations) {
  auto old = sample();
  context->observe_joints(old);
  old.position[0] = 99.0;
  context->observe_joints(old);
  EXPECT_NE(context->snapshot().joints.at(old.name[0]).position, 99.0);
  auto malformed = sample();
  malformed.position[0] = std::numeric_limits<double>::quiet_NaN();
  context->observe_joints(malformed);
  EXPECT_TRUE(std::isfinite(context->snapshot().joints.at(old.name[0]).position));
  malformed = sample();
  malformed.velocity.resize(1);
  context->observe_joints(malformed);
  EXPECT_EQ(context->snapshot().joints.size(), positions.size());
}

TEST_F(ContextFixture, RejectsMissingCalibrationBeforeSendingCommands) {
  config.motion_sequences.at("left_to_right").steps[0].target.clear();
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  EXPECT_FALSE(motion.start("left_to_right", reason));
  EXPECT_NE(reason.find("calibrated"), std::string::npos);
  EXPECT_TRUE(executor.steps.empty());
}

TEST_F(ContextFixture, UnknownEffortDoesNotDiscardValidPositionFeedback) {
  auto msg = sample();
  msg.position[0] = 0.15;
  msg.effort.assign(msg.name.size(), std::numeric_limits<double>::quiet_NaN());
  context->observe_joints(msg);
  const auto observed = context->snapshot().joints.at(msg.name[0]);
  EXPECT_DOUBLE_EQ(observed.position, 0.15);
  EXPECT_FALSE(observed.effort.has_value());
  std::string reason;
  EXPECT_TRUE(context->ready(reason)) << reason;
}

TEST_F(ContextFixture, BothDirectionsUseCorrectArmAndNeverClaimPhysicalTransfer) {
  for (const std::string direction : {"left_to_right", "right_to_left"}) {
    FakeExecutor executor;
    HandoverMotion motion(config, *context, executor);
    std::string reason;
    ASSERT_TRUE(motion.start(direction, reason)) << reason;
    for (int i = 0; i < 20 && motion.running(); ++i) {
      refresh();
      if (executor.current.state == ExecutionState::Running) {
        executor.current.state = ExecutionState::Succeeded;
      }
      motion.tick();
    }
    ASSERT_TRUE(motion.result().sequence_completed);
    EXPECT_FALSE(motion.result().handover_confirmed);
    ASSERT_EQ(executor.steps.size(), 4u);
    EXPECT_EQ(executor.steps[0].arm, config.motion_sequences.at(direction).giver);
    EXPECT_EQ(executor.steps[1].arm, config.motion_sequences.at(direction).receiver);
    EXPECT_EQ(executor.steps[2].arm, config.motion_sequences.at(direction).giver);
  }
}

TEST_F(ContextFixture, ReceiverFailureDoesNotReleaseGiver) {
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason));
  motion.tick();
  executor.current.state = ExecutionState::Succeeded;
  motion.tick();
  motion.tick();
  ASSERT_EQ(executor.steps.back().name, "receiver_close");
  executor.current = {ExecutionState::Failed, "TIMEOUT", "receiver did not reach target"};
  motion.tick();
  motion.tick();
  EXPECT_EQ(executor.steps.size(), 2u);
  EXPECT_EQ(motion.result().state, "FAILED");
  EXPECT_EQ(motion.result().last_completed_step, "present");
}

TEST_F(ContextFixture, CancellationBetweenStepsDoesNotStartTheNextStep) {
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("right_to_left", reason));
  motion.tick();
  executor.current.state = ExecutionState::Succeeded;
  motion.tick();
  motion.cancel();
  motion.tick();
  EXPECT_EQ(executor.steps.size(), 1u);
  EXPECT_EQ(motion.result().state, "CANCELLED");
}

TEST_F(ContextFixture, StaleFeedbackCancelsActiveExecution) {
  config.state_max_age = 0.02;
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason));
  motion.tick();
  std::this_thread::sleep_for(25ms);
  motion.tick();
  EXPECT_EQ(executor.cancel_count, 1);
  EXPECT_EQ(motion.result().code, "ROBOT_STATE_INVALID");
  EXPECT_FALSE(motion.result().sequence_completed);
}

TEST_F(ContextFixture, ContinuousApproachKeepsTransferBarrierAndCorrectNextStep) {
  configure_five_step_sequences(config);
  config.smooth_motion = true;
  for (auto & [direction, sequence] : config.motion_sequences) {
    sequence.steps[0].blend_with_next = true;
  }
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason)) << reason;
  motion.tick();
  ASSERT_EQ(executor.continuous_starts, 1);
  ASSERT_EQ(executor.steps.size(), 2u);
  motion.tick();
  EXPECT_EQ(executor.steps.size(), 2u);
  executor.current.phase = "open_approach";
  motion.tick();
  EXPECT_EQ(context->snapshot().phase, "open_approach");
  executor.current.state = ExecutionState::Succeeded;
  motion.tick();
  EXPECT_EQ(motion.result().last_completed_step, "open_approach");
  EXPECT_EQ(executor.steps.size(), 2u);
  motion.tick();
  ASSERT_EQ(executor.steps.size(), 3u);
  EXPECT_EQ(executor.steps.back().name, "receiver_close");
}

TEST_F(ContextFixture, FailedContinuousApproachCannotCloseOrRelease) {
  configure_five_step_sequences(config);
  config.smooth_motion = true;
  config.motion_sequences.at("left_to_right").steps[0].blend_with_next = true;
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason));
  motion.tick();
  executor.current = {ExecutionState::Failed, "PATH_PREPARATION_FAILED", "collision"};
  motion.tick();
  motion.tick();
  EXPECT_FALSE(motion.running());
  EXPECT_FALSE(motion.result().sequence_completed);
  EXPECT_EQ(executor.steps.size(), 2u);
}

TEST_F(ContextFixture, BlendingCannotCrossGripperBarrierOrEndOfSequence) {
  configure_five_step_sequences(config);
  config.smooth_motion = true;
  auto sequence = config.motion_sequences.at("left_to_right");
  sequence.steps[1].blend_with_next = true;
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
  sequence.steps[1].blend_with_next = false;
  sequence.steps.back().blend_with_next = true;
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
  sequence.steps.back().blend_with_next = false;
  sequence.steps[0].velocity_scale = 1.1;
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
}

class RosFixture : public ContextFixture {
protected:
  using Move = moveit_msgs::action::MoveGroup;
  using Joint = control_msgs::action::FollowJointTrajectory;
  using MoveHandle = rclcpp_action::ServerGoalHandle<Move>;
  using JointHandle = rclcpp_action::ServerGoalHandle<Joint>;

  void SetUp() override {
    ContextFixture::SetUp();
    mock = std::make_shared<rclcpp::Node>(prefix.substr(1) + "_mock");
    state_pub = mock->create_publisher<sensor_msgs::msg::JointState>(
        config.joint_states_topic, rclcpp::SensorDataQoS());
    move_server = rclcpp_action::create_server<Move>(
        mock,
        config.move_group_action,
        [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const Move::Goal> goal) {
          requests.push_back("move:" + goal->request.group_name + ":" + goal->request.planner_id);
          move_goals.push_back(*goal);
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](std::shared_ptr<MoveHandle>) {
          ++cancels;
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](std::shared_ptr<MoveHandle> handle) {
          moves.push_back(handle);
        });
    for (const auto & [side, arm] : config.arms) {
      joint_servers[side] = rclcpp_action::create_server<Joint>(
          mock,
          arm.trajectory_action,
          [this, side](const rclcpp_action::GoalUUID &, std::shared_ptr<const Joint::Goal>) {
            requests.push_back("gripper:" + side);
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
          },
          [this](std::shared_ptr<JointHandle>) {
            ++cancels;
            return rclcpp_action::CancelResponse::ACCEPT;
          },
          [this](std::shared_ptr<JointHandle> handle) {
            grippers.push_back(handle);
          });
      topic_subs[side] =
          mock->create_subscription<trajectory_msgs::msg::JointTrajectory>(arm.trajectory_topic,
              rclcpp::QoS(1).reliable(),
              [this, side](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg) {
                requests.push_back("topic:" + side);
                if (follow_topic && !msg->points.empty()) {
                  for (size_t i = 0; i < msg->joint_names.size(); ++i) {
                    positions.at(msg->joint_names[i]) = msg->points.back().positions.at(i);
                  }
                }
              });
    }
    timer = mock->create_wall_timer(5ms, [this]() {
      auto feedback = sample();
      if (++feedback_count % 2) {
        for (size_t i = 0; i < feedback.name.size(); ++i) {
          for (const auto & [side, arm] : config.arms) {
            (void)side;
            if (feedback.name[i] == arm.gripper_joint &&
                (jitter_joint.empty() || jitter_joint == feedback.name[i])) {
              feedback.position[i] += gripper_position_jitter;
            }
          }
        }
      }
      state_pub->publish(feedback);
      for (auto it = moves.begin(); it != moves.end();) {
        auto result = std::make_shared<Move::Result>();
        if ((*it)->is_canceling()) {
          result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PREEMPTED;
          (*it)->canceled(result);
          it = moves.erase(it);
        } else if (auto_complete) {
          result->error_code.val = move_error;
          if (move_result_has_execution) {
            result->executed_trajectory.joint_trajectory.points.resize(1);
          }
          if (move_error == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
            const auto & goal = *(*it)->get_goal();
            if (goal.planning_options.plan_only) {
              result->trajectory_start = goal.request.start_state;
              auto & path = result->planned_trajectory.joint_trajectory;
              trajectory_msgs::msg::JointTrajectoryPoint initial, target;
              for (const auto & joint : goal.request.goal_constraints.front().joint_constraints) {
                path.joint_names.push_back(joint.joint_name);
                const auto & start = goal.request.start_state.joint_state;
                const auto found =
                    std::find(start.name.begin(), start.name.end(), joint.joint_name);
                initial.positions.push_back(found == start.name.end()
                                                ? positions.at(joint.joint_name)
                                                : start.position.at(found - start.name.begin()));
                target.positions.push_back(joint.position);
              }
              if (path.joint_names.empty() && !goal.request.goal_constraints.front().position_constraints.empty()) {
                for (const auto & name : config.arms.at("left").joints) {
                  path.joint_names.push_back(name);
                  initial.positions.push_back(positions.at(name) - plan_start_offset);
                  target.positions.push_back(positions.at(name) + .1);
                }
                initial.time_from_start = rclcpp::Duration::from_seconds(0);
                target.time_from_start = rclcpp::Duration::from_seconds(.2);
              }
              path.points = {initial, target};
            } else {
              for (const auto & constraint : goal.request.goal_constraints) {
                for (const auto & joint : constraint.joint_constraints) {
                  if (joint.joint_name != omitted_joint) {
                    positions.at(joint.joint_name) = joint.position;
                  }
                }
              }
            }
          }
          (*it)->succeed(result);
          it = moves.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = executions.begin(); it != executions.end();) {
        auto result = std::make_shared<moveit_msgs::action::ExecuteTrajectory::Result>();
        if ((*it)->is_canceling()) {
          result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PREEMPTED;
          (*it)->canceled(result);
          it = executions.erase(it);
        } else if (auto_execute) {
          result->error_code.val = execute_error;
          if (execute_error == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
            const auto & path = (*it)->get_goal()->trajectory.joint_trajectory;
            for (size_t i = 0; i < path.joint_names.size(); ++i) {
              positions.at(path.joint_names[i]) = path.points.back().positions[i];
            }
          }
          (*it)->succeed(result);
          it = executions.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = grippers.begin(); it != grippers.end();) {
        auto result = std::make_shared<Joint::Result>();
        if ((*it)->is_canceling()) {
          (*it)->canceled(result);
          it = grippers.erase(it);
        } else if (auto_gripper) {
          const auto & trajectory = (*it)->get_goal()->trajectory;
          for (size_t i = 0; i < trajectory.joint_names.size(); ++i) {
            const auto & name = trajectory.joint_names[i];
            const double target = trajectory.points.back().positions[i];
            positions.at(name) = name == blocked_gripper ? std::min(target, .25) : target;
          }
          result->error_code = Joint::Result::SUCCESSFUL;
          (*it)->succeed(result);
          it = grippers.erase(it);
        } else {
          ++it;
        }
      }
    });
    ros.add_node(node);
    ros.add_node(mock);
  }

  void TearDown() override {
    ros.remove_node(node);
    ros.remove_node(mock);
  }

  bool until(
      const std::function<bool()> & condition,
      double timeout = 2.0,
      const std::function<void()> & tick = []() {}) {
    const auto started = Clock::now();
    do {
      ros.spin_some();
      tick();
      if (condition()) {
        return true;
      }
      std::this_thread::sleep_for(2ms);
    } while (elapsed(started) < timeout);
    return false;
  }

  void warmup(ExecutorPort & executor) {
    std::string reason;
    ASSERT_TRUE(until([&]() {
      return context->ready(reason) &&
             executor.ready(config.motion_sequences.at("left_to_right"), reason);
    })) << reason;
  }

  void enable_smoothing() {
    config.smooth_motion = true;
    configure_five_step_sequences(config);
    for (auto & [direction, sequence] : config.motion_sequences) {
      for (auto & step : sequence.steps) {
        step.timeout = 5.0;
      }
    }
    config.motion_sequences.at("left_to_right").steps[0].blend_with_next = true;
    config.motion_sequences.at("right_to_left").steps[0].blend_with_next = true;
    std::ostringstream urdf, srdf;
    urdf << "<robot name='smoothing_mock'><link name='base_link'/>";
    srdf << "<robot name='smoothing_mock'><group name='" << config.coordinated_group << "'>";
    for (const auto & [name, position] : positions) {
      (void)position;
      urdf << "<link name='" << name << "_link'/><joint name='" << name
           << "' type='revolute'><parent link='base_link'/><child link='" << name
           << "_link'/><axis xyz='0 0 1'/><limit lower='-3' upper='3' effort='10' "
              "velocity='2'/></joint>";
      srdf << "<joint name='" << name << "'/>";
    }
    urdf << "</robot>";
    srdf << "</group></robot>";
    node->declare_parameter("robot_description", urdf.str());
    node->declare_parameter("robot_description_semantic", srdf.str());
    scene_server = mock->create_service<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene",
        [this](const moveit_msgs::srv::GetPlanningScene::Request::SharedPtr,
            moveit_msgs::srv::GetPlanningScene::Response::SharedPtr response) {
          response->scene.name = "mock_scene";
          response->scene.robot_model_name = "smoothing_mock";
          response->scene.robot_state.joint_state = sample();
        });
    execution_server = rclcpp_action::create_server<moveit_msgs::action::ExecuteTrajectory>(
        mock,
        "/execute_trajectory",
        [this](const rclcpp_action::GoalUUID &,
            std::shared_ptr<const moveit_msgs::action::ExecuteTrajectory::Goal> goal) {
          requests.push_back("execute:path");
          execution_goals.push_back(*goal);
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](auto) {
          ++cancels;
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](auto handle) {
          executions.push_back(handle);
        });
  }

  Step correction_step() {
    // The real planner is replaced by an action server; this test exercises dispatch only.
    enable_smoothing();
    config.smooth_motion = false;
    broadcaster = std::make_unique<tf2_ros::StaticTransformBroadcaster>(mock);
    geometry_msgs::msg::TransformStamped tf;
    tf.header.frame_id = config.planning_frame;
    tf.header.stamp = node->now();
    tf.child_frame_id = config.arms.at("left").tcp;
    tf.transform.translation.x = .5;
    tf.transform.translation.z = .8;
    tf.transform.rotation.w = 1;
    broadcaster->sendTransform(tf);
    EXPECT_TRUE(until([&]() { return context->pose_of(tf.child_frame_id).has_value(); }));
    Step step;
    step.name = "CORRECT_TO_CUP_CENTER";
    step.type = "pose";
    step.arm = "left";
    step.frame = config.planning_frame;
    step.target = {.5, 0., .8, 0., 0., 0., 1.};
    step.orientation_slack = .17;
    step.tcp_path_margin = .04;
    step.planning_time = 3.;
    step.timeout = 5.;
    return step;
  }

  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster;
  rclcpp::Service<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_server;
  rclcpp_action::Server<moveit_msgs::action::ExecuteTrajectory>::SharedPtr execution_server;
  std::vector<
      std::shared_ptr<rclcpp_action::ServerGoalHandle<moveit_msgs::action::ExecuteTrajectory>>>
      executions;
  std::vector<moveit_msgs::action::ExecuteTrajectory::Goal> execution_goals;
  int execute_error{moveit_msgs::msg::MoveItErrorCodes::SUCCESS};
  bool auto_execute{true};
  double plan_start_offset{0};
  std::shared_ptr<rclcpp::Node> mock;
  rclcpp::executors::SingleThreadedExecutor ros;
  rclcpp_action::Server<Move>::SharedPtr move_server;
  std::map<std::string, rclcpp_action::Server<Joint>::SharedPtr> joint_servers;
  std::map<std::string, rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr>
      topic_subs;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub;
  rclcpp::TimerBase::SharedPtr timer;
  std::vector<std::shared_ptr<MoveHandle>> moves;
  std::vector<std::shared_ptr<JointHandle>> grippers;
  std::vector<std::string> requests;
  std::vector<Move::Goal> move_goals;
  std::string omitted_joint;
  bool move_result_has_execution{false};
  int move_error{moveit_msgs::msg::MoveItErrorCodes::SUCCESS};
  bool auto_complete{true}, follow_topic{true}, auto_gripper{true};
  int cancels{0};
  int feedback_count{0};
  double gripper_position_jitter{0.0};
  std::string jitter_joint, blocked_gripper;
};

TEST_F(RosFixture, BodyJointGroupUsesMoveItAndChecksMeasuredCompletion) {
  positions["head_joint1"] = 0;
  positions["head_joint2"] = 0;
  refresh();
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  Step step;
  step.name = "head_observe";
  step.type = "joint_group";
  step.group = "head";
  step.joint_names = {"head_joint1", "head_joint2"};
  step.target = {.69, 0};
  step.timeout = 2;
  step.tolerance = .02;
  omitted_joint = "head_joint1";
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return !move_goals.empty(); }, 1, [&]() { executor.tick(); }));
  const auto & request = move_goals.front().request;
  EXPECT_EQ(request.group_name, "head");
  EXPECT_EQ(request.pipeline_id, "ompl");
  EXPECT_EQ(request.goal_constraints.front().joint_constraints.size(), 2u);
  EXPECT_FALSE(until([&]() { return executor.result().state == ExecutionState::Succeeded; },
      .08, [&]() { executor.tick(); }));
  positions["head_joint1"] = .69;
  ASSERT_TRUE(until([&]() { return executor.result().state == ExecutionState::Succeeded; },
      1, [&]() { executor.tick(); }));
  EXPECT_TRUE(grippers.empty());
}

TEST_F(RosFixture, BodyJointGroupRejectsMissingFeedbackBeforeDispatch) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  Step step;
  step.name = "lift_observe";
  step.type = "joint_group";
  step.group = "lift";
  step.joint_names = {"lift_joint"};
  step.target = {-.1};
  EXPECT_FALSE(executor.start(step));
  EXPECT_NE(executor.result().detail.find("lift_joint"), std::string::npos);
  EXPECT_TRUE(move_goals.empty());
}

TEST_F(RosFixture, RealActionClientsExecuteBothDirectionsSerially) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  for (const std::string direction : {"left_to_right", "right_to_left"}) {
    requests.clear();
    std::string reason;
    ASSERT_TRUE(motion.start(direction, reason)) << reason;
    ASSERT_TRUE(until(
        [&]() {
          return !motion.running();
        },
        3.0,
        [&]() {
          motion.tick();
        }));
    EXPECT_TRUE(motion.result().sequence_completed) << motion.result().detail;
    EXPECT_FALSE(motion.result().handover_confirmed);
    const auto & sequence = config.motion_sequences.at(direction);
    EXPECT_EQ(requests,
        (std::vector<std::string>{"move:" + sequence.giver + ":PTP",
            "gripper:" + sequence.receiver,
            "gripper:" + sequence.giver,
            "move:" + sequence.giver + ":PTP"}));
    for (const auto & goal : move_goals) {
      EXPECT_EQ(goal.request.num_planning_attempts, 1); // Pilz stays single-shot.
    }
  }
}

TEST_F(RosFixture, FiveStagesCoordinateBothArmsAndGripperInOnePlan) {
  configure_five_step_sequences(config);
  velocities["right_gripper"] = 0.214885;
  velocities["left_gripper"] = -0.083566;
  gripper_position_jitter = 0.0030679615757712823;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  for (const auto & direction : {"left_to_right", "right_to_left"}) {
    requests.clear();
    move_goals.clear();
    std::string reason;
    ASSERT_TRUE(motion.start(direction, reason)) << reason;
    ASSERT_TRUE(until(
        [&]() {
          return !motion.running();
        },
        3.0,
        [&]() {
          motion.tick();
        }));
    ASSERT_TRUE(motion.result().sequence_completed) << motion.result().detail;
    EXPECT_FALSE(motion.result().handover_confirmed);
    const auto & sequence = config.motion_sequences.at(direction);
    const auto move_request = "move:" + config.coordinated_group + ":" + config.coordinated_planner;
    EXPECT_EQ(requests,
        (std::vector<std::string>{move_request,
            move_request,
            "gripper:" + sequence.receiver,
            "gripper:" + sequence.giver,
            move_request}));
    ASSERT_EQ(move_goals.size(), 3u);
    for (const auto & goal : move_goals) {
      EXPECT_EQ(goal.request.goal_constraints[0].joint_constraints.size(), 16u);
      EXPECT_EQ(goal.request.pipeline_id, config.coordinated_pipeline);
      EXPECT_EQ(goal.request.num_planning_attempts, 3);
    }
    EXPECT_EQ(move_goals[0].request.path_constraints.joint_constraints.size(), 2u);
    ASSERT_EQ(move_goals[1].request.path_constraints.joint_constraints.size(), 1u);
    for (const auto & joint : move_goals[1].request.path_constraints.joint_constraints) {
      const auto & giver = config.arms.at(sequence.giver);
      EXPECT_EQ(joint.joint_name, giver.gripper_joint);
      EXPECT_LE(joint.tolerance_above, 0.001);
    }
    std::map<std::string, double> target;
    for (const auto & joint : move_goals[1].request.goal_constraints[0].joint_constraints) {
      target[joint.joint_name] = joint.position;
    }
    EXPECT_DOUBLE_EQ(target.at(config.arms.at(sequence.receiver).gripper_joint), 0.0);
    for (const auto & joint : move_goals[1].request.goal_constraints[0].joint_constraints) {
      const bool gripper =
          joint.joint_name == "left_gripper" || joint.joint_name == "right_gripper";
      EXPECT_DOUBLE_EQ(joint.tolerance_above, gripper ? config.gripper_goal_tolerance : 0.0005);
    }
    for (const auto & side : {"left", "right"}) {
      for (const auto & name : config.arms.at(side).joints) {
        EXPECT_DOUBLE_EQ(target.at(name), std::string(side) == "left" ? 0.2 : -0.2);
      }
    }
    EXPECT_EQ(motion.result().last_completed_step, "retreat_together");
  }
}

TEST_F(RosFixture, DualArmStepWaitsForBothArmsMeasuredArrival) {
  configure_five_step_sequences(config);
  omitted_joint = config.arms.at("right").joints.back();
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(config.motion_sequences.at("left_to_right").steps[0]));
  ASSERT_TRUE(until(
      [&]() {
        return !move_goals.empty() && moves.empty();
      },
      1.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_FALSE(until(
      [&]() {
        return executor.result().state == ExecutionState::Succeeded;
      },
      0.08,
      [&]() {
        executor.tick();
      }));
  positions.at(omitted_joint) = -0.1;
  EXPECT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Succeeded;
      },
      1.0,
      [&]() {
        executor.tick();
      }));
}

TEST_F(ContextFixture, FiveStageApproachFailureCannotCloseOrRelease) {
  configure_five_step_sequences(config);
  FakeExecutor executor;
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("right_to_left", reason));
  motion.tick();
  executor.current.state = ExecutionState::Succeeded;
  motion.tick();
  motion.tick();
  ASSERT_EQ(executor.steps.back().name, "open_approach");
  executor.current = {ExecutionState::Failed, "EXECUTION_FAILED", "coordinated plan failed"};
  motion.tick();
  motion.tick();
  EXPECT_EQ(executor.steps.size(), 2u);
  EXPECT_EQ(motion.result().last_completed_step, "face_each_other");
}

TEST_F(ContextFixture, InvalidCoordinatedTargetsAreRejectedBeforeSending) {
  configure_five_step_sequences(config);
  auto sequence = config.motion_sequences.at("left_to_right");
  sequence.steps[0].right_target.pop_back();
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
  sequence = config.motion_sequences.at("left_to_right");
  sequence.steps[1].gripper_target = {std::numeric_limits<double>::quiet_NaN()};
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
  sequence = config.motion_sequences.at("left_to_right");
  sequence.steps[1].left_target.clear();
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
  sequence = config.motion_sequences.at("left_to_right");
  sequence.steps[1].arm = "unknown";
  EXPECT_FALSE(validate_motion_sequence(sequence, config).empty());
}

TEST_F(RosFixture, OnlyTimeoutWithoutTrajectoryIsRetryablePlanningFailure) {
  move_error = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  Step step;
  step.name = "timeout_test";
  step.type = "joint";
  step.arm = "left";
  step.target = std::vector<double>(7, 0.0);
  step.timeout = 2;
  for (bool execution : {false, true}) {
    move_result_has_execution = execution;
    ASSERT_TRUE(executor.start(step));
    ASSERT_TRUE(until([&] {
      executor.tick();
      return executor.result().state != ExecutionState::Running;
    }));
    EXPECT_EQ(executor.result().code, execution ? "EXECUTION_FAILED" : "PLANNING_TIMEOUT");
  }
}

TEST_F(RosFixture, MoveItPayloadFailurePreventsGripperCommands) {
  move_error = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason));
  ASSERT_TRUE(until(
      [&]() {
        return !motion.running();
      },
      2.0,
      [&]() {
        motion.tick();
      }));
  EXPECT_EQ(motion.result().state, "FAILED");
  EXPECT_EQ(requests.size(), 1u);
}

TEST_F(RosFixture, TopicDeliveryDoesNotCountAsReachingGripperTarget) {
  config.gripper_transport = "topic";
  follow_topic = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  auto step = config.motion_sequences.at("left_to_right").steps[1];
  step.timeout = 0.12;
  config.cancel_timeout = 0.4;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Failed;
      },
      1.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_EQ(executor.result().code, "TIMEOUT");
  EXPECT_NE(executor.result().detail.find("right_gripper"), std::string::npos);
  EXPECT_GE(requests.size(), 2u); // original trajectory plus the measured-position hold
}

TEST_F(RosFixture, GripperWaitOnlySkipsClosureButWaitsForDuration) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  timer->cancel(); // Deliberately keep measured fingers short of the target.
  auto step = config.motion_sequences.at("left_to_right").steps[1];
  step.gripper_wait_only = true;
  step.duration = 0.2;
  step.timeout = 1.0;
  const auto started = Clock::now();
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return !grippers.empty(); }));
  const auto & tolerance = grippers.front()->get_goal()->goal_tolerance.at(0);
  EXPECT_EQ(tolerance.position, -1.0);
  EXPECT_EQ(tolerance.velocity, -1.0);
  EXPECT_EQ(tolerance.acceleration, -1.0);
  auto result = std::make_shared<Joint::Result>();
  result->error_code = Joint::Result::SUCCESSFUL;
  grippers.front()->succeed(result);
  ASSERT_TRUE(until([&]() { return executor.result().state == ExecutionState::Succeeded; },
      1.0, [&]() { refresh(); executor.tick(); }));
  EXPECT_GE(elapsed(started), step.duration);
  EXPECT_DOUBLE_EQ(context->snapshot().joints.at("right_gripper").position, 0.2);
}

TEST_F(RosFixture, GripperWaitOnlyStillRejectsControllerErrors) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  timer->cancel();
  auto step = config.motion_sequences.at("left_to_right").steps[1];
  step.gripper_wait_only = true;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return !grippers.empty(); }));
  auto result = std::make_shared<Joint::Result>();
  result->error_code = Joint::Result::PATH_TOLERANCE_VIOLATED;
  grippers.front()->abort(result);
  ASSERT_TRUE(until([&]() { return executor.result().state == ExecutionState::Failed; },
      1.0, [&]() { refresh(); executor.tick(); }));
  EXPECT_EQ(executor.result().code, "EXECUTION_FAILED");
}

TEST_F(RosFixture, GripperWaitOnlyCanCancelBeforeAcceptance) {
  auto_gripper = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  auto step = config.motion_sequences.at("left_to_right").steps[1];
  step.gripper_wait_only = true;
  ASSERT_TRUE(executor.start(step));
  executor.cancel();
  ASSERT_TRUE(until([&]() { return executor.result().state == ExecutionState::Cancelled; },
      2.0, [&]() { executor.tick(); }));
  EXPECT_EQ(cancels, 1);
}

TEST_F(RosFixture, ActionSuccessWithoutMeasuredTargetExplainsTimeout) {
  omitted_joint = "left_joint0";
  config.cancel_timeout = 0.4;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  auto step = config.motion_sequences.at("left_to_right").steps[0];
  step.timeout = 0.12;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Failed;
      },
      1.0,
      [&]() {
        executor.tick();
      }));
  const auto result = executor.result();
  EXPECT_EQ(result.code, "TIMEOUT");
  EXPECT_NE(result.detail.find("Joint target not reached: left_joint0"), std::string::npos);
  EXPECT_NE(result.detail.find("error=0.100000, tolerance=0.010000"), std::string::npos);
  EXPECT_NE(result.detail.find("action_terminal=true, action_success=true"), std::string::npos);
  EXPECT_EQ(requests.size(), 1u);
}

TEST_F(RosFixture, PrecisePlanningStillAcceptsMeasuredErrorWithinOneDegree) {
  omitted_joint = "left_joint0";
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  auto step = config.motion_sequences.at("left_to_right").steps[0];
  step.target.assign(step.target.size(), .01);
  step.tolerance = 0.017453292519943295;
  step.planning_tolerance = .0005;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Succeeded;
      },
      1.0,
      [&]() {
        executor.tick();
      }));
  ASSERT_EQ(move_goals.size(), 1u);
  for (const auto & constraint : move_goals[0].request.goal_constraints[0].joint_constraints) {
    EXPECT_DOUBLE_EQ(constraint.tolerance_above, .0005);
  }
}

TEST_F(RosFixture, MissingActionResultExplainsTimeout) {
  auto_complete = false;
  config.cancel_timeout = 0.4;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  auto step = config.motion_sequences.at("left_to_right").steps[0];
  step.timeout = 0.12;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Failed;
      },
      1.0,
      [&]() {
        executor.tick();
      }));
  const auto result = executor.result();
  EXPECT_EQ(result.code, "TIMEOUT");
  EXPECT_NE(result.detail.find("Waiting for controller/action result"), std::string::npos);
  EXPECT_NE(result.detail.find("action_terminal=false, action_success=false"), std::string::npos);
}

TEST_F(RosFixture, TopicModeUsesMeasuredFeedbackToFinish) {
  config.gripper_transport = "topic";
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(config.motion_sequences.at("left_to_right").steps[1]));
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Succeeded;
      },
      1.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_DOUBLE_EQ(context->snapshot().joints.at("right_gripper").position, 0.4);
}

TEST_F(RosFixture, CancellationBeforeGoalAcceptanceCancelsLateGoal) {
  auto_complete = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(config.motion_sequences.at("left_to_right").steps[0]));
  executor.cancel(); // no callbacks have yet delivered the remote goal handle
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Cancelled;
      },
      2.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_EQ(cancels, 1);
}

TEST_F(RosFixture, UnconfirmedStopStaysLatchedAfterLateCancellationResult) {
  auto_complete = false;
  config.cancel_timeout = 0.03;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  const auto step = config.motion_sequences.at("left_to_right").steps[0];
  ASSERT_TRUE(executor.start(step));
  executor.cancel();
  std::this_thread::sleep_for(40ms); // deliberately do not spin the remote server
  refresh();
  executor.tick();
  EXPECT_EQ(executor.result().state, ExecutionState::StopUnconfirmed);
  ASSERT_TRUE(until(
      [&]() {
        return cancels == 1 && moves.empty();
      },
      2.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_EQ(executor.result().state, ExecutionState::StopUnconfirmed);
  EXPECT_FALSE(executor.start(step));
}

TEST_F(RosFixture, TaskActionRejectsOverlapAndCancelsOnlyItsActiveExecution) {
  auto_complete = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  TaskManager manager(*node, *context, motion);
  auto client = rclcpp_action::create_client<demo_1f::action::Handover>(mock, "/handover");
  ASSERT_TRUE(until([&]() {
    return client->action_server_is_ready();
  }));
  demo_1f::action::Handover::Goal request;
  request.direction = "left_to_right";
  auto first = client->async_send_goal(request);
  ASSERT_TRUE(until([&]() {
    return first.wait_for(0s) == std::future_status::ready;
  }));
  const auto handle = first.get();
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(until([&]() {
    return !moves.empty();
  }));
  auto second = client->async_send_goal(request);
  ASSERT_TRUE(until([&]() {
    return second.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_EQ(second.get(), nullptr);
  auto result = client->async_get_result(handle);
  client->async_cancel_goal(handle);
  ASSERT_TRUE(until([&]() {
    return result.wait_for(0s) == std::future_status::ready;
  }));
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_FALSE(result.get().result->sequence_completed);
  EXPECT_EQ(requests.size(), 1u);
}

TEST_F(RosFixture, PoseTargetUsesPtpAndMeasuredTcpInsteadOfJointTargets) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  tf2_ros::StaticTransformBroadcaster broadcaster(mock);
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = config.planning_frame;
  tf.child_frame_id = config.arms.at("right").tcp;
  tf.header.stamp = node->now();
  tf.transform.translation.x = .4;
  tf.transform.translation.y = -.2;
  tf.transform.translation.z = .8;
  tf.transform.rotation.w = 1;
  broadcaster.sendTransform(tf);
  ASSERT_TRUE(until([&] {
    return context->pose_of(tf.child_frame_id).has_value();
  }));
  Step step;
  step.name = "cup_observe";
  step.type = "pose";
  step.arm = "right";
  step.frame = config.planning_frame;
  step.target = {.4, -.2, .8, 0, 0, 0, 1};
  step.timeout = 1;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&] {
    executor.tick();
    return executor.result().state != ExecutionState::Running;
  }));
  EXPECT_EQ(executor.result().state, ExecutionState::Succeeded) << executor.result().detail;
  ASSERT_EQ(move_goals.size(), 1u);
  EXPECT_EQ(move_goals[0].request.planner_id, "PTP");
  EXPECT_EQ(move_goals[0].request.goal_constraints[0].orientation_constraints.size(), 1u);
  EXPECT_TRUE(move_goals[0].request.goal_constraints[0].joint_constraints.empty());
  EXPECT_EQ(move_goals[0].request.goal_constraints[0].position_constraints.size(), 1u);
}

TEST_F(RosFixture, TiltBoundsConstrainOnlyGoalAndRequireMeasuredArrivalTilt) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  tf2_ros::StaticTransformBroadcaster broadcaster(mock);
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = config.planning_frame;
  tf.child_frame_id = config.arms.at("left").tcp;
  tf.transform.translation.x = .4;
  tf.transform.translation.y = .2;
  tf.transform.translation.z = .8;
  auto send_tilt = [&](double roll, double pitch, double yaw = -45) {
    tf2::Quaternion hand, tcp_axes;
    hand.setRPY(roll * M_PI / 180, pitch * M_PI / 180, yaw * M_PI / 180);
    tcp_axes.setRPY(0, -M_PI / 2, 0);
    const auto q = hand * tcp_axes;
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    tf.header.stamp = node->now();
    broadcaster.sendTransform(tf);
    ASSERT_TRUE(until([&] {
      const auto current = context->pose_of(tf.child_frame_id);
      if (!current) {
        return false;
      }
      const auto & actual = current->pose.orientation;
      // TF may return the equivalent quaternion -q.
      return std::abs(actual.x * q.x() + actual.y * q.y() + actual.z * q.z() + actual.w * q.w()) >
             1 - 1e-10;
    }));
  };
  // Starting outside the goal tilt band must still allow a planning request.
  send_tilt(60, 20);
  Step step;
  step.name = "cup_goal_tilt";
  step.type = "pose";
  step.position_only = true;
  step.arm = "left";
  step.frame = config.planning_frame;
  step.target = {.4, .2, .8, 0, 0, 0, 1};
  step.tcp_roll_limit = 10 * M_PI / 180;
  step.tcp_pitch_min = 20 * M_PI / 180;
  step.tcp_pitch_limit = 60 * M_PI / 180;
  step.tcp_pitch_downward_only = true;
  step.tcp_yaw_inward_only = true;
  step.timeout = 3;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&] {
    executor.tick();
    return !move_goals.empty() && moves.empty();
  }));
  const auto & request = move_goals[0].request;
  EXPECT_EQ(request.pipeline_id, "ompl");
  EXPECT_TRUE(request.path_constraints.orientation_constraints.empty());
  ASSERT_EQ(request.goal_constraints.size(), 1u);
  const auto & goal = request.goal_constraints[0];
  EXPECT_EQ(goal.position_constraints.size(), 2u);
  ASSERT_EQ(goal.orientation_constraints.size(), 3u);
  const auto & band = goal.orientation_constraints[0];
  EXPECT_EQ(band.header.frame_id, config.arms.at("left").tcp);
  EXPECT_EQ(band.link_name, config.planning_frame);
  EXPECT_NEAR(band.absolute_x_axis_tolerance, 9 * M_PI / 180, 1e-9);
  EXPECT_NEAR(band.absolute_y_axis_tolerance, 59 * M_PI / 180, 1e-9);
  EXPECT_NEAR(band.absolute_z_axis_tolerance, M_PI, 1e-9);
  // A successful controller result and correct XYZ cannot bypass measured tilt checks.
  for (const auto & tilt :
      std::vector<std::pair<double, double>>{{60, 40}, {0, 65}, {0, -20}, {0, 10}}) {
    send_tilt(tilt.first, tilt.second);
    EXPECT_FALSE(until(
        [&] {
          executor.tick();
          return executor.result().state != ExecutionState::Running;
        },
        .08));
  }
  for (double yaw : {20., -100.}) {
    send_tilt(0, 40, yaw);
    EXPECT_FALSE(until(
        [&] {
          executor.tick();
          return executor.result().state != ExecutionState::Running;
        },
        .08));
  }
  send_tilt(0, 40);
  ASSERT_TRUE(until([&] {
    executor.tick();
    return executor.result().state != ExecutionState::Running;
  }));
  EXPECT_EQ(executor.result().state, ExecutionState::Succeeded) << executor.result().detail;
}

TEST_F(RosFixture, FlexiblePoseUsesOmplCorridorAndAcceptsSmallRotation) {
  enable_smoothing(); // Provide the separate trajectory execution action.
  config.smooth_motion = false;
  config.ompl_planning_attempts = 4;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  tf2_ros::StaticTransformBroadcaster broadcaster(mock);
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = config.planning_frame;
  tf.child_frame_id = config.arms.at("left").tcp;
  tf.header.stamp = node->now();
  tf.transform.translation.x = .4;
  tf.transform.translation.y = .2;
  tf.transform.translation.z = .8;
  tf.transform.rotation.x = std::sin(5 * M_PI / 360);
  tf.transform.rotation.w = std::cos(5 * M_PI / 360);
  broadcaster.sendTransform(tf);
  ASSERT_TRUE(until([&] {
    return context->pose_of(tf.child_frame_id).has_value();
  }));
  Step step;
  step.name = "cup_position";
  step.type = "pose";
  step.orientation_slack = 10 * M_PI / 180;
  step.tcp_path_margin = .02;
  step.tcp_path_min_z = .795;
  step.planning_time = 3;
  step.arm = "left";
  step.frame = config.planning_frame;
  step.target = {.4, .2, .8, 0, 0, 0, 1};
  step.timeout = 5;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&] {
    executor.tick();
    return executor.result().state != ExecutionState::Running;
  }));
  EXPECT_EQ(executor.result().state, ExecutionState::Succeeded) << executor.result().detail;
  ASSERT_EQ(move_goals.size(), 1u);
  const auto & request = move_goals[0].request;
  EXPECT_EQ(request.group_name, config.arms.at("left").group);
  EXPECT_EQ(request.pipeline_id, "ompl");
  EXPECT_EQ(request.planner_id, "RRTConnectkConfigDefault");
  ASSERT_EQ(request.goal_constraints.size(), 1u);
  EXPECT_EQ(request.goal_constraints[0].position_constraints.size(), 1u);
  EXPECT_DOUBLE_EQ(request.goal_constraints[0]
                       .position_constraints[0]
                       .constraint_region.primitives[0]
                       .dimensions[0],
      config.pose_position_tolerance);
  ASSERT_EQ(request.goal_constraints[0].orientation_constraints.size(), 1u);
  const auto & orientation = request.goal_constraints[0].orientation_constraints[0];
  EXPECT_EQ(orientation.parameterization, moveit_msgs::msg::OrientationConstraint::ROTATION_VECTOR);
  EXPECT_NEAR(orientation.absolute_x_axis_tolerance, 9 * M_PI / 180, 1e-9);
  ASSERT_EQ(request.path_constraints.position_constraints.size(), 1u);
  EXPECT_NEAR(request.path_constraints.position_constraints[0]
                  .constraint_region.primitives[0]
                  .dimensions[0],
      .04,
      1e-9);
  EXPECT_DOUBLE_EQ(request.allowed_planning_time, 3);
  EXPECT_EQ(request.num_planning_attempts, 4); // One shared 3 s budget, not 4 x 3 s.
  const auto & box = request.path_constraints.position_constraints[0].constraint_region;
  EXPECT_NEAR(box.primitives[0].dimensions[1], .04, 1e-9);
  EXPECT_NEAR(box.primitives[0].dimensions[2], .025, 1e-9);
  EXPECT_NEAR(box.primitive_poses[0].position.z - box.primitives[0].dimensions[2] / 2, .795, 1e-9);
  EXPECT_TRUE(request.goal_constraints[0].joint_constraints.empty());
  EXPECT_TRUE(request.path_constraints.orientation_constraints.empty());
  step.planning_time = 0;
  step.timeout = 1;
  step.target[3] = std::sin(40 * M_PI / 360);
  step.target[6] = std::cos(40 * M_PI / 360);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until(
      [&] {
        executor.tick();
        return executor.result().state != ExecutionState::Running;
      },
      2.0));
  EXPECT_NE(executor.result().state, ExecutionState::Succeeded);
}

TEST_F(RosFixture, PositionOnlyPoseUsesOmplAndCompletesAtDifferentOrientation) {
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  tf2_ros::StaticTransformBroadcaster broadcaster(mock);
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = config.planning_frame;
  tf.child_frame_id = config.arms.at("left").tcp;
  tf.header.stamp = node->now();
  tf.transform.translation.x = .4;
  tf.transform.translation.y = .2;
  tf.transform.translation.z = .8;
  // A 180-degree difference must not block a position-only completion.
  tf.transform.rotation.x = 1;
  broadcaster.sendTransform(tf);
  ASSERT_TRUE(until([&] {
    return context->pose_of(tf.child_frame_id).has_value();
  }));
  Step step;
  step.name = "cup_position";
  step.type = "pose";
  step.position_only = true;
  step.arm = "left";
  step.frame = config.planning_frame;
  step.target = {.4, .2, .8, 0, 0, 0, 1};
  step.timeout = 1;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&] {
    executor.tick();
    return executor.result().state != ExecutionState::Running;
  }));
  EXPECT_EQ(executor.result().state, ExecutionState::Succeeded) << executor.result().detail;
  ASSERT_EQ(move_goals.size(), 1u);
  const auto & request = move_goals[0].request;
  EXPECT_EQ(request.group_name, config.arms.at("left").group);
  EXPECT_EQ(request.pipeline_id, "ompl");
  EXPECT_EQ(request.planner_id, "RRTConnectkConfigDefault");
  EXPECT_EQ(request.num_planning_attempts, 3);
  ASSERT_EQ(request.goal_constraints.size(), 1u);
  EXPECT_EQ(request.goal_constraints[0].position_constraints.size(), 1u);
  EXPECT_DOUBLE_EQ(request.goal_constraints[0]
                       .position_constraints[0]
                       .constraint_region.primitives[0]
                       .dimensions[0],
      config.pose_position_tolerance * .5);
  EXPECT_TRUE(request.goal_constraints[0].orientation_constraints.empty());
  EXPECT_TRUE(request.goal_constraints[0].joint_constraints.empty());
  EXPECT_TRUE(request.path_constraints.orientation_constraints.empty());
  // Free orientation must not mean free position: offset XYZ still fails completion.
  step.target[0] += .1;
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until(
      [&] {
        executor.tick();
        return executor.result().state != ExecutionState::Running;
      },
      2.0));
  EXPECT_NE(executor.result().state, ExecutionState::Succeeded);
  // Other step types must not silently lose their constraints.
  step.type = "linear";
  MotionSequence invalid;
  invalid.steps = {step};
  EXPECT_FALSE(validate_motion_sequence(invalid, config).empty());
}

} // namespace
} // namespace demo_1f

namespace demo_1f {
namespace {
TEST_F(RosFixture, ContinuousPlansAreReadOnlyUntilOneCombinedExecution) {
  enable_smoothing();
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason)) << reason;
  ASSERT_TRUE(until(
      [&]() {
        return !motion.running();
      },
      5.0,
      [&]() {
        motion.tick();
      }));
  EXPECT_TRUE(motion.result().sequence_completed) << motion.result().detail;
  ASSERT_EQ(move_goals.size(), 3u);
  for (const auto & goal : move_goals) {
    EXPECT_TRUE(goal.planning_options.plan_only);
    EXPECT_EQ(goal.request.num_planning_attempts, 3);
  }
  ASSERT_EQ(execution_goals.size(), 2u);
  EXPECT_EQ(execution_goals[0].trajectory.joint_trajectory.joint_names.size(), 16u);
  EXPECT_EQ(requests,
      (std::vector<std::string>{"move:handover_arms:RRTConnectkConfigDefault",
          "move:handover_arms:RRTConnectkConfigDefault",
          "execute:path",
          "gripper:right",
          "gripper:left",
          "move:handover_arms:RRTConnectkConfigDefault",
          "execute:path"}));
}

TEST_F(RosFixture, CurrentLimitedReceiverDoesNotBlockReleaseAndRetreat) {
  enable_smoothing();
  config.require_passive_gripper_stationary = false;
  for (auto & [direction, sequence] : config.motion_sequences) {
    sequence.steps[2].gripper_wait_only = true;
  }
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  for (const std::string direction : {"right_to_left", "left_to_right"}) {
    gripper_position_jitter = 0;
    const auto & sequence = config.motion_sequences.at(direction);
    blocked_gripper = config.arms.at(sequence.receiver).gripper_joint;
    jitter_joint = blocked_gripper;
    std::string reason;
    ASSERT_TRUE(motion.start(direction, reason)) << reason;
    ASSERT_TRUE(until([&]() { return !motion.running(); }, 6.0, [&]() {
      if (context->snapshot().phase == "receiver_close") gripper_position_jitter = .006;
      motion.tick();
    }));
    EXPECT_TRUE(motion.result().sequence_completed) << motion.result().detail;
    EXPECT_FALSE(motion.result().handover_confirmed);
    EXPECT_LT(positions.at(blocked_gripper), .3); // Closed goal .4 was never reached.
    ASSERT_GE(execution_goals.size(), 2u);
    EXPECT_EQ(execution_goals[execution_goals.size() - 2].trajectory.joint_trajectory.joint_names.size(), 15u);
    EXPECT_EQ(execution_goals.back().trajectory.joint_trajectory.joint_names.size(), 14u);
  }
}

TEST_F(RosFixture, CancelDuringPlanningNeverDispatchesExecution) {
  enable_smoothing();
  auto_complete = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  const auto & steps = config.motion_sequences.at("left_to_right").steps;
  ASSERT_TRUE(executor.start_sequence({steps[0], steps[1]}));
  ASSERT_TRUE(until([&]() {
    return !moves.empty();
  }));
  executor.cancel();
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Cancelled;
      },
      2.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_TRUE(execution_goals.empty());
  EXPECT_EQ(move_goals.size(), 1u);
}

TEST_F(RosFixture, CancelDuringContinuousExecutionWaitsForStop) {
  enable_smoothing();
  auto_execute = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  const auto & steps = config.motion_sequences.at("left_to_right").steps;
  ASSERT_TRUE(executor.start_sequence({steps[0], steps[1]}));
  ASSERT_TRUE(until(
      [&]() {
        return !executions.empty();
      },
      3.0,
      [&]() {
        executor.tick();
      }));
  executor.cancel();
  EXPECT_EQ(executor.result().state, ExecutionState::Cancelling);
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Cancelled;
      },
      2.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_GT(cancels, 0);
}

TEST_F(RosFixture, ContinuousExecutionFailureBlocksBothGripperActions) {
  enable_smoothing();
  execute_error = moveit_msgs::msg::MoveItErrorCodes::CONTROL_FAILED;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  HandoverMotion motion(config, *context, executor);
  std::string reason;
  ASSERT_TRUE(motion.start("left_to_right", reason));
  ASSERT_TRUE(until(
      [&]() {
        return !motion.running();
      },
      3.0,
      [&]() {
        motion.tick();
      }));
  EXPECT_FALSE(motion.result().sequence_completed);
  EXPECT_TRUE(grippers.empty());
  for (const auto & request : requests) {
    EXPECT_EQ(request.find("gripper:"), std::string::npos);
  }
}
} // namespace
} // namespace demo_1f

namespace demo_1f {
namespace {
TEST_F(RosFixture, ContinuousApiRejectsGripperStepsBeforePlanning) {
  enable_smoothing();
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  const auto & steps = config.motion_sequences.at("left_to_right").steps;
  EXPECT_FALSE(executor.start_sequence({steps[2], steps[3]}));
  EXPECT_TRUE(move_goals.empty());
  EXPECT_TRUE(execution_goals.empty());
  EXPECT_TRUE(grippers.empty());
}
} // namespace
} // namespace demo_1f

namespace demo_1f {
namespace {
TEST_F(RosFixture, CancelBeforeGripperSceneResponseCannotSendGripperGoal) {
  enable_smoothing();
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  const auto & step = config.motion_sequences.at("left_to_right").steps[2];
  ASSERT_TRUE(executor.start(step));
  executor.cancel();
  ASSERT_TRUE(until(
      [&]() {
        return executor.result().state == ExecutionState::Cancelled;
      },
      2.0,
      [&]() {
        executor.tick();
      }));
  EXPECT_TRUE(grippers.empty());
  for (const auto & request : requests) {
    EXPECT_EQ(request.find("gripper:"), std::string::npos);
  }
}

TEST_F(RosFixture, CorrectionPlansBeforeExecutingTheReturnedTrajectory) {
  const auto step = correction_step();
  auto_execute = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return !execution_goals.empty(); }, 2., [&]() { executor.tick(); }));
  ASSERT_EQ(move_goals.size(), 1u);
  EXPECT_TRUE(move_goals.front().planning_options.plan_only);
  EXPECT_EQ(move_goals.front().request.num_planning_attempts, 3);
  EXPECT_EQ(execution_goals.front().trajectory.joint_trajectory.joint_names, config.arms.at("left").joints);
  EXPECT_EQ(execution_goals.front().trajectory.joint_trajectory.points.size(), 2u);
  EXPECT_EQ(executor.result().state, ExecutionState::Running);
  auto_execute = true;
  ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Running; }, 2., [&]() { executor.tick(); }));
  EXPECT_EQ(executor.result().state, ExecutionState::Succeeded) << executor.result().detail;
}

TEST_F(RosFixture, CorrectionClassifiesEmptyPlanningFailuresWithoutExecution) {
  const auto step = correction_step();
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  for (const auto error : {moveit_msgs::msg::MoveItErrorCodes::FAILURE,
                          moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED,
                          moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT}) {
    move_error = error;
    ASSERT_TRUE(executor.start(step));
    ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Running; }, 2., [&]() { executor.tick(); }));
    EXPECT_TRUE(move_goals.back().planning_options.plan_only);
    EXPECT_EQ(executor.result().code, error == moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT ? "PLANNING_TIMEOUT" : "PLANNING_FAILED");
    EXPECT_TRUE(execution_goals.empty());
  }
}

TEST_F(RosFixture, CorrectionExecutionFailureNeverBecomesPlanningFailure) {
  const auto step = correction_step();
  execute_error = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Running; }, 2., [&]() { executor.tick(); }));
  ASSERT_EQ(execution_goals.size(), 1u);
  EXPECT_EQ(executor.result().code, "EXECUTION_FAILED");
}

TEST_F(RosFixture, CorrectionDoesNotRetryAnUnexpectedExecutionTraceInPlanResult) {
  const auto step = correction_step();
  move_error = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
  move_result_has_execution = true;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Running; }, 2., [&]() { executor.tick(); }));
  EXPECT_EQ(executor.result().code, "EXECUTION_FAILED");
  EXPECT_TRUE(execution_goals.empty());
}

TEST_F(RosFixture, CorrectionCancelledDuringPlanningDoesNotExecute) {
  const auto step = correction_step();
  auto_complete = false;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return !moves.empty(); }, 2., [&]() { executor.tick(); }));
  executor.cancel();
  ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Cancelling; }, 2., [&]() { executor.tick(); }));
  EXPECT_EQ(executor.result().state, ExecutionState::Cancelled);
  EXPECT_TRUE(execution_goals.empty());
}


TEST_F(RosFixture, CorrectionRejectsMovedStartBeforeExecutingPlan) {
  const auto step = correction_step();
  plan_start_offset = .1;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Running; }, 2., [&]() { executor.tick(); }));
  EXPECT_EQ(executor.result().code, "PATH_PREPARATION_FAILED");
  EXPECT_NE(executor.result().detail.find("Robot moved"), std::string::npos);
  EXPECT_TRUE(execution_goals.empty());
}

TEST_F(RosFixture, CorrectionRejectsUnexpectedExecutionEvenOnPlanSuccess) {
  const auto step = correction_step();
  move_result_has_execution = true;
  TrajectoryExecutor executor(*node, config, *context);
  warmup(executor);
  ASSERT_TRUE(executor.start(step));
  ASSERT_TRUE(until([&]() { return executor.result().state != ExecutionState::Running; }, 2., [&]() { executor.tick(); }));
  EXPECT_EQ(executor.result().code, "PATH_PREPARATION_FAILED");
  EXPECT_TRUE(execution_goals.empty());
}

} // namespace
} // namespace demo_1f
