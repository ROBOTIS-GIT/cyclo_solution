#include "demo_1f/runtime/state/robot_context.hpp"
#include <gtest/gtest.h>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>

using namespace std::chrono_literals;

namespace demo_1f {
namespace {
class CupTfFixture : public ::testing::Test {
protected:
  void SetUp() override {
    static int index = 0;
    const auto name = "cup_tf_" + std::to_string(++index);
    const auto topic = "/" + name + "/cup";
    node = std::make_shared<rclcpp::Node>(
        name, rclcpp::NodeOptions().arguments({"--ros-args", "-r", "/perception/cup:=" + topic}));
    config.state_max_age = 1;
    config.tf_max_age = 1;
    context = std::make_unique<RobotContext>(*node, config);
    publisher = node->create_publisher<msg::CupObservation>(topic, 5);
    broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(node);
    frame = name + "_camera";
    executor.add_node(node);
    ASSERT_TRUE(until([&] {
      return publisher->get_subscription_count() > 0;
    }));
  }

  bool until(const std::function<bool()> & condition, double seconds = 1) {
    const auto start = Clock::now();
    while (elapsed(start) < seconds) {
      executor.spin_some();
      if (condition()) {
        return true;
      }
      std::this_thread::sleep_for(2ms);
    }
    return condition();
  }

  void spin_for(double seconds) {
    const auto start = Clock::now();
    until(
        [&] {
          return elapsed(start) >= seconds;
        },
        seconds + .1);
  }

  void transform(const rclcpp::Time & stamp, double x = 1) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.frame_id = config.planning_frame;
    tf.child_frame_id = frame;
    tf.header.stamp = stamp;
    tf.transform.translation.x = x;
    tf.transform.rotation.w = 1;
    broadcaster->sendTransform(tf);
  }

  void seed_tf(const rclcpp::Time & stamp, double x = 1) {
    ASSERT_TRUE(until([&] {
      transform(stamp, x);
      return context->pose_of(frame).has_value();
    }));
  }

  msg::CupObservation observation(const rclcpp::Time & stamp) {
    msg::CupObservation input;
    input.header.frame_id = frame;
    input.header.stamp = stamp;
    input.source = "head";
    input.cup_id = "cup";
    input.observation_type = msg::CupObservation::BODY;
    input.body_position.x = .2;
    input.body_position.z = .8;
    input.inner_radius = .045;
    input.position_error = .09;
    input.confidence = .9;
    input.valid = true;
    return input;
  }

  std::optional<CupState> cup() {
    return context->cup_observation("cup", "head");
  }

  rclcpp::Node::SharedPtr node;
  Config config;
  std::unique_ptr<RobotContext> context;
  rclcpp::Publisher<msg::CupObservation>::SharedPtr publisher;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster;
  rclcpp::executors::SingleThreadedExecutor executor;
  std::string frame;
};

TEST_F(CupTfFixture, DelayedTfUsesImageTimestampInsteadOfLatestPose) {
  const auto image_time = node->now();
  seed_tf(image_time - rclcpp::Duration::from_seconds(.1), 1);
  publisher->publish(observation(image_time));
  spin_for(.03);
  EXPECT_FALSE(cup()); // Latest TF exists, but does not cover the image timestamp.
  const auto phase_started = Clock::now();
  transform(image_time + rclcpp::Duration::from_seconds(.1), 3);
  ASSERT_TRUE(until([&] {
    return cup().has_value();
  }));
  const auto result = *cup();
  EXPECT_LT(result.received, phase_started); // Waiting for TF must not make an old input "new".
  EXPECT_EQ(
      result.observation.header.stamp, static_cast<builtin_interfaces::msg::Time>(image_time));
  EXPECT_NEAR(result.observation.body_position.x, 2.2, 1e-6); // Interpolated TF, not latest 3.2.
}

TEST_F(CupTfFixture, ContinuousNewFramesDoNotStarveOlderFramesWaitingForTf) {
  const auto first = node->now() - rclcpp::Duration::from_seconds(.15);
  seed_tf(first - rclcpp::Duration::from_seconds(.05));
  for (int i = 0; i < 4; ++i) {
    publisher->publish(observation(first + rclcpp::Duration::from_seconds(.02 * i)));
    spin_for(.005);
  }
  EXPECT_FALSE(cup());
  // TF now covers the first three images; the newest image must remain pending.
  transform(first + rclcpp::Duration::from_seconds(.05));
  ASSERT_TRUE(until([&] {
    auto c = cup();
    return c && c->stable_frames >= 3;
  }));
  EXPECT_EQ(cup()->observation.header.stamp,
      static_cast<builtin_interfaces::msg::Time>(first + rclcpp::Duration::from_seconds(.04)));
  transform(first + rclcpp::Duration::from_seconds(.08));
  ASSERT_TRUE(until([&] {
    auto c = cup();
    return c && c->stable_frames == 4;
  }));
}

TEST_F(CupTfFixture, DetectionLossImmediatelyInvalidatesPendingOlderDetection) {
  const auto stamp = node->now();
  seed_tf(stamp - rclcpp::Duration::from_seconds(.1));
  publisher->publish(observation(stamp));
  spin_for(.02);
  auto lost = observation(stamp + rclcpp::Duration::from_seconds(.01));
  lost.valid = false;
  publisher->publish(lost);
  ASSERT_TRUE(until([&] {
    auto c = cup();
    return c && !c->observation.valid;
  }));
  transform(stamp + rclcpp::Duration::from_seconds(.1));
  spin_for(.03);
  ASSERT_TRUE(cup());
  EXPECT_FALSE(cup()->observation.valid);
  EXPECT_EQ(cup()->observation.header.stamp, lost.header.stamp);
}

TEST_F(CupTfFixture, ExpiredPendingFramesCannotReappearAfterTfArrives) {
  config.state_max_age = .1;
  const auto stamp = node->now();
  seed_tf(stamp - rclcpp::Duration::from_seconds(.05));
  publisher->publish(observation(stamp));
  spin_for(.16);
  // Widening the policy must not resurrect a frame that was already discarded.
  config.state_max_age = 1;
  transform(stamp + rclcpp::Duration::from_seconds(.1));
  spin_for(.03);
  EXPECT_FALSE(cup());
}
} // namespace
} // namespace demo_1f
