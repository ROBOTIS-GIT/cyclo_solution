#include "demo_1f/tasks/task_manager.hpp"
#include "demo_1f/tasks/demo_recovery.hpp"
#include <gtest/gtest.h>
#include <thread>

using namespace std::chrono_literals;
namespace demo_1f {
namespace {
using Demo = action::RunDemo;
using Pick = ffw_vision_msgs::action::PickDynamixel;

class Environment : public ::testing::Environment {
  void SetUp() override { int argc = 0; rclcpp::init(argc, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};
auto * environment = ::testing::AddGlobalTestEnvironment(new Environment);

class Executor : public ExecutorPort {
public:
  std::vector<std::string> executed;
  ExecutionResult state;
  std::string rejected_step;
  bool stall = false;
  size_t rejected_call = 0, stalled_call = 0;
  bool ready(const MotionSequence &, std::string &) const override { return true; }
  bool start(const Step & step) override {
    executed.push_back(step.name);
    if (step.name == rejected_step || executed.size() == rejected_call) {
      state = {ExecutionState::Failed, "EXECUTION_FAILED", "test execution failure"};
      return false;
    }
    state.state = ExecutionState::Running;
    return true;
  }
  void tick() override {
    if (!stall && executed.size() != stalled_call && state.state == ExecutionState::Running)
      state.state = ExecutionState::Succeeded;
  }
  void cancel(const std::string & code, const std::string & detail) override {
    state = {ExecutionState::Cancelled, code, detail};
  }
  ExecutionResult result() const override { return state; }
};
class Scene : public ApproachScenePort {
public:
  bool ready() const override { return true; }
  bool apply(const moveit_msgs::msg::PlanningScene &) override { return true; }
  ExecutionResult poll() override { return {ExecutionState::Succeeded, {}, {}}; }
};
class Cup : public CupMotion {
public:
  using CupMotion::CupMotion;
  int runs = 0;
  int failures = 0;
  std::string failure = "CUP_NOT_FOUND";
  std::string requested_arm;
  bool start(const std::string & arm, const std::string & cup_id,
      std::string & reason, bool execute = true) override {
    requested_arm = arm;
    return CupMotion::start(arm == "both" ? "left" : arm, cup_id, reason, execute);
  }
  void tick() override {
    if (result_.state == "CANCELLING") { result_.state = "CANCELLED"; return; }
    if (!running()) return;
    if (runs <= failures) { fail(failure, "test failure"); return; }
    result_.state = "SUCCEEDED";
    result_.sequence_completed = true;
  }
protected:
  const char * initial_phase() const override { return "TEST_CUP"; }
  void reset() override { ++runs; }
};

class DemoTest : public ::testing::Test {
protected:
  rclcpp::Node::SharedPtr node;
  rclcpp::executors::SingleThreadedExecutor loop;
  Config config;
  ApproachConfig cup_config;
  Executor executor;
  Scene scene;
  std::unique_ptr<RobotContext> context;
  std::unique_ptr<HandoverMotion> handover;
  std::unique_ptr<InitialPoseMotion> initial;
  std::unique_ptr<Cup> approach, refine, withdraw;
  std::unique_ptr<TaskManager> manager;
  rclcpp_action::Server<Pick>::SharedPtr backend;
  rclcpp_action::Client<Demo>::SharedPtr client;
  std::shared_ptr<rclcpp_action::ServerGoalHandle<Pick>> pending_pick;
  int picks = 0, pick_failures = 0;
  bool stall_pick = false, moved_failure = false, uncertain_stop = false;

  void SetUp() override {
    node = std::make_shared<rclcpp::Node>("demo_task_test",
        rclcpp::NodeOptions().parameter_overrides({rclcpp::Parameter("demo.retry_delay", .01)}));
    config.require_tcp_tf = false;
    config.state_max_age = 5;
    for (const std::string name : {"left", "right"}) {
      ArmConfig arm;
      arm.name = name;
      arm.joints = {name + "_joint"};
      arm.gripper_joint = name + "_grip";
      config.arms[name] = arm;
    }
    MotionSequence sequence;
    sequence.direction = "right_to_left";
    sequence.giver = "right";
    sequence.receiver = "left";
    Step step;
    step.name = "handover_test";
    step.type = "joint";
    step.arm = "right";
    step.target = {.1};
    step.timeout = 1;
    step.duration = 1;
    sequence.steps = {step};
    config.motion_sequences[sequence.direction] = sequence;
    auto init = sequence;
    init.direction = "initial_pose";
    init.steps.front().name = "initial_test";
    config.motion_sequences[init.direction] = init;
    context = std::make_unique<RobotContext>(*node, config);
    handover = std::make_unique<HandoverMotion>(config, *context, executor);
    initial = std::make_unique<InitialPoseMotion>(config, *context, executor);
    approach = std::make_unique<Cup>(config, cup_config, *context, executor, scene);
    refine = std::make_unique<Cup>(config, cup_config, *context, executor, scene);
    withdraw = std::make_unique<Cup>(config, cup_config, *context, executor, scene);
    manager = std::make_unique<TaskManager>(*node, *context, *handover,
        std::vector<CupMotionRegistration>{{"/approach_cup", *approach}, {"/refine_cup", *refine},
            {"/withdraw_cup", *withdraw}}, initial.get());
    backend = rclcpp_action::create_server<Pick>(node, "/dynamixel_pick/execute",
        [](const auto &, auto) { return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE; },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](auto handle) {
          ++picks;
          EXPECT_FALSE(handle->get_goal()->move_to_observe);
          if (stall_pick) { pending_pick = handle; return; }
          auto result = std::make_shared<Pick::Result>();
          result->success = picks > pick_failures;
          result->holding = false; // No physical grasp verdict; motion success is sufficient.
          result->stop_confirmed = !uncertain_stop;
          result->retryable = !result->success;
          result->motion_started = moved_failure;
          if (!result->success) result->error_code = "OBJECT_NOT_FOUND";
          if (result->success) handle->succeed(result); else handle->abort(result);
        });
    client = rclcpp_action::create_client<Demo>(node, "/run_demo");
    loop.add_node(node);
    ASSERT_TRUE(client->wait_for_action_server(2s));
  }
  void pump() {
    sensor_msgs::msg::JointState joints;
    joints.header.stamp = node->now();
    joints.name = {"left_joint", "right_joint", "left_grip", "right_grip"};
    joints.position = {0, 0, 0, 0};
    joints.velocity = {0, 0, 0, 0};
    context->observe_joints(joints);
    loop.spin_some();
    if (pending_pick && pending_pick->is_canceling()) {
      auto result = std::make_shared<Pick::Result>();
      result->stop_confirmed = true;
      result->error_code = "CANCELLED";
      pending_pick->canceled(result);
      pending_pick.reset();
    }
    std::this_thread::sleep_for(2ms);
  }
  template<class Future> void wait(Future & future) {
    auto end = std::chrono::steady_clock::now() + 5s;
    while (future.wait_for(0s) != std::future_status::ready && std::chrono::steady_clock::now() < end) pump();
    ASSERT_EQ(future.wait_for(0s), std::future_status::ready);
  }
  rclcpp_action::ClientGoalHandle<Demo>::SharedPtr send(const std::string & command = "start") {
    pump();
    Demo::Goal goal;
    goal.command = command;
    goal.cup_id = "cup";
    auto accepted = client->async_send_goal(goal);
    wait(accepted);
    return accepted.get();
  }
  void TearDown() override {
    loop.remove_node(node);
    client.reset(); backend.reset(); manager.reset();
    withdraw.reset(); refine.reset(); approach.reset(); initial.reset(); handover.reset(); context.reset(); node.reset();
  }
};

TEST_F(DemoTest, StartupDoesNotMove) {
  for (int i = 0; i < 30; ++i) pump();
  EXPECT_EQ(picks, 0);
  EXPECT_TRUE(executor.executed.empty());
  EXPECT_EQ(approach->runs, 0);
  EXPECT_EQ(refine->runs, 0);
}

TEST_F(DemoTest, InitialOnlyNeedsNoPickBackendAndStopsBeforePick) {
  backend.reset();
  auto goal = send("initial"); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(result.get().result->last_completed_stage, "INITIALIZE_ROBOT");
  EXPECT_FALSE(result.get().result->can_resume);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test"}));
  EXPECT_EQ(picks, 0);
  EXPECT_EQ(approach->runs, 0);
  EXPECT_EQ(refine->runs, 0);
  EXPECT_FALSE(send("resume"));
}

TEST_F(DemoTest, InitialOnlyCanBeCancelledWithoutStartingPick) {
  executor.stall = true;
  auto goal = send("initial"); ASSERT_TRUE(goal);
  for (int i = 0; i < 50 && executor.executed.empty(); ++i) pump();
  ASSERT_FALSE(executor.executed.empty());
  auto cancel = client->async_cancel_goal(goal); wait(cancel);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(picks, 0);
}

TEST_F(DemoTest, SequencesAllMotionsAndRetriesOnlyFailedStage) {
  pick_failures = 1;
  approach->failures = 1;
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(picks, 2);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test", "handover_test", "initial_test"}));
  EXPECT_EQ(approach->runs, 2);
  EXPECT_EQ(refine->runs, 1);
  EXPECT_EQ(withdraw->runs, 1);
  EXPECT_EQ(result.get().result->holding_state, "UNVERIFIED");
  EXPECT_EQ(withdraw->requested_arm, "both");
  EXPECT_EQ(result.get().result->last_completed_stage, "RETURN_INITIAL");
}
TEST_F(DemoTest, ExhaustedObservationRetriesResumeWithoutRepicking) {
  approach->failures = 3;
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  ASSERT_TRUE(result.get().result->can_resume);
  EXPECT_EQ(picks, 1);
  auto resumed = send("resume"); ASSERT_TRUE(resumed);
  auto finished = client->async_get_result(resumed); wait(finished);
  EXPECT_TRUE(finished.get().result->success);
  EXPECT_EQ(picks, 1);
  EXPECT_EQ(executor.executed.size(), 3u);
}
TEST_F(DemoTest, PickOnlyCanContinueWithHandover) {
  const auto initial_sequence = config.motion_sequences.at("initial_pose");
  config.motion_sequences.erase("initial_pose"); // Pick must not require an initialization sequence.
  auto goal = send("pick"); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_TRUE(result.get().result->success);
  EXPECT_TRUE(executor.executed.empty());
  config.motion_sequences["initial_pose"] = initial_sequence; // Full continuation returns here.
  auto resumed = send("resume"); ASSERT_TRUE(resumed);
  auto finished = client->async_get_result(resumed); wait(finished);
  EXPECT_TRUE(finished.get().result->success);
  EXPECT_EQ(picks, 1);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"handover_test", "initial_test"}));
}

TEST_F(DemoTest, FailedRefinementNeverStartsReturnToInitial) {
  refine->failures = 1;
  refine->failure = "EXECUTION_FAILED";
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_FALSE(result.get().result->success);
  EXPECT_EQ(result.get().result->last_completed_stage, "APPROACH_CUP");
  EXPECT_EQ(withdraw->runs, 0);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test", "handover_test"}));
}

TEST_F(DemoTest, WithdrawalFailureNeverStartsReturnOrRepeatsRelease) {
  withdraw->failures = 1;
  withdraw->failure = "EXECUTION_FAILED";
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_FALSE(result.get().result->success);
  EXPECT_EQ(result.get().result->last_completed_stage, "REFINE_CUP");
  EXPECT_FALSE(result.get().result->can_resume);
  EXPECT_EQ(refine->runs, 1);
  EXPECT_EQ(withdraw->runs, 1);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test", "handover_test"}));
}

TEST_F(DemoTest, StandaloneWithdrawalDoesNotStartReleaseOrInitialPose) {
  backend.reset();
  pump();
  auto manual = rclcpp_action::create_client<action::ApproachCup>(node, "/withdraw_cup");
  ASSERT_TRUE(manual->wait_for_action_server(2s));
  action::ApproachCup::Goal request;
  request.arm = "left";
  request.cup_id = "cup";
  request.execute = true;
  auto accepted = manual->async_send_goal(request); wait(accepted);
  ASSERT_TRUE(accepted.get());
  auto result = manual->async_get_result(accepted.get()); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(withdraw->runs, 1);
  EXPECT_EQ(refine->runs, 0);
  EXPECT_EQ(picks, 0);
  EXPECT_TRUE(executor.executed.empty());
}

TEST_F(DemoTest, ReturnFailureDoesNotRepeatPickOrRelease) {
  executor.rejected_call = 3;
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_FALSE(result.get().result->success);
  EXPECT_EQ(result.get().result->last_completed_stage, "WITHDRAW_CUP");
  EXPECT_FALSE(result.get().result->can_resume);
  EXPECT_EQ(picks, 1);
  EXPECT_EQ(refine->runs, 1);
  EXPECT_EQ(executor.executed.size(), 3u);
}

TEST_F(DemoTest, ReturnToInitialCanBeCancelledWithoutRepeatingRelease) {
  executor.stalled_call = 3;
  auto goal = send(); ASSERT_TRUE(goal);
  for (int i = 0; i < 250 && executor.executed.size() < 3; ++i) pump();
  ASSERT_EQ(executor.executed.size(), 3u);
  auto cancel = client->async_cancel_goal(goal); wait(cancel);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(picks, 1);
  EXPECT_EQ(refine->runs, 1);
  EXPECT_TRUE(manager->can_shutdown());
}
TEST_F(DemoTest, ExecutionFailureDoesNotReplayPickOrRelease) {
  pick_failures = 5; moved_failure = true;
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_FALSE(result.get().result->can_resume);
  EXPECT_EQ(picks, 1);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test"}));
  EXPECT_EQ(refine->runs, 0);
}
TEST_F(DemoTest, CancellationWaitsForBackendAndBlocksOtherGoals) {
  stall_pick = true;
  auto goal = send(); ASSERT_TRUE(goal);
  for (int i = 0; i < 50 && !pending_pick; ++i) pump();
  ASSERT_TRUE(pending_pick);
  EXPECT_FALSE(send());
  auto manual = rclcpp_action::create_client<action::Handover>(node, "/handover");
  action::Handover::Goal manual_goal;
  manual_goal.direction = "right_to_left";
  auto rejected = manual->async_send_goal(manual_goal); wait(rejected);
  EXPECT_FALSE(rejected.get());
  auto cup_client = rclcpp_action::create_client<action::ApproachCup>(node, "/approach_cup");
  action::ApproachCup::Goal cup_goal;
  cup_goal.arm = "left"; cup_goal.cup_id = "cup"; cup_goal.execute = true;
  auto cup_rejected = cup_client->async_send_goal(cup_goal); wait(cup_rejected);
  EXPECT_FALSE(cup_rejected.get());
  auto cancel = client->async_cancel_goal(goal); wait(cancel);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test"}));
  EXPECT_TRUE(manager->can_shutdown());
}
TEST_F(DemoTest, UnknownStopLatchesFault) {
  uncertain_stop = true;
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().result->error_code, "STOP_UNCONFIRMED");
  EXPECT_FALSE(send());
  EXPECT_EQ(executor.executed, (std::vector<std::string>{"initial_test"}));
}
TEST_F(DemoTest, InitializationFailureNeverStartsPick) {
  executor.rejected_step = "initial_test";
  auto goal = send(); ASSERT_TRUE(goal);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_FALSE(result.get().result->success);
  EXPECT_EQ(picks, 0);
  EXPECT_EQ(approach->runs, 0);
}
TEST_F(DemoTest, InitializationCanBeCancelledBeforeAnyPick) {
  executor.stall = true;
  auto goal = send(); ASSERT_TRUE(goal);
  for (int i = 0; i < 50 && executor.executed.empty(); ++i) pump();
  ASSERT_FALSE(executor.executed.empty());
  auto cancel = client->async_cancel_goal(goal); wait(cancel);
  auto result = client->async_get_result(goal); wait(result);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(picks, 0);
  EXPECT_TRUE(manager->can_shutdown());
}
TEST(DemoRecovery, DoesNotRetryReleaseOrArbitraryExecutionFailures) {
  EXPECT_FALSE(demo_can_retry("HANDOVER", "EXECUTION_FAILED"));
  EXPECT_FALSE(demo_can_retry("REFINE_CUP", "EXECUTION_FAILED"));
  EXPECT_TRUE(demo_can_retry("REFINE_CUP", "WRIST_NOT_READY"));
}
} // namespace
} // namespace demo_1f

namespace demo_1f {
namespace {
TEST_F(DemoTest, NewStartAndInitialAllowedAfterCompletedPickFailure) {
  pick_failures = 1;
  moved_failure = true;
  auto goal = send("pick"); ASSERT_TRUE(goal);
  auto failed = client->async_get_result(goal); wait(failed);
  ASSERT_FALSE(failed.get().result->success);
  auto initial_goal = send("initial"); ASSERT_TRUE(initial_goal);
  auto initialized = client->async_get_result(initial_goal); wait(initialized);
  ASSERT_TRUE(initialized.get().result->success);
  auto start_goal = send(); ASSERT_TRUE(start_goal);
  auto finished = client->async_get_result(start_goal); wait(finished);
  EXPECT_TRUE(finished.get().result->success);
}
} // namespace
} // namespace demo_1f
