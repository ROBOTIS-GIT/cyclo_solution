#include "demo_1f/motions/pick/pick_cup_motion.hpp"
#include "demo_1f/motions/initial/initial_pose_motion.hpp"
#include "demo_1f/motions/place/approach_cup_motion.hpp"
#include "demo_1f/motions/place/refine_cup_motion.hpp"
#include "demo_1f/motions/place/withdraw_cup_motion.hpp"
#include "demo_1f/runtime/config.hpp"
#include "demo_1f/runtime/execution/trajectory_executor.hpp"
#include "demo_1f/runtime/execution/trajectory_smoother.hpp"
#include "demo_1f/tasks/task_manager.hpp"
#include <csignal>
#include <thread>

namespace {
volatile std::sig_atomic_t stop_requested = 0;

void signal_handler(int) {
  stop_requested = 1;
}
} // namespace

int main(int argc, char ** argv) {
  demo_1f::initialize_moveit_logging();
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  int exit_code = 0;
  try {
    auto node = std::make_shared<rclcpp::Node>("handover_demo");
    const auto config = demo_1f::load_config(*node);
    for (const auto & [name, sequence] : config.motion_sequences) {
      if (!sequence.validation_error.empty()) {
        RCLCPP_WARN(node->get_logger(),
            "%s not configured: %s",
            name.c_str(),
            sequence.validation_error.c_str());
      }
    }
    const auto approach_config = demo_1f::load_approach_config(*node);
    demo_1f::RobotContext robot_context(*node, config,
        {approach_config.inner_radius, approach_config.wall_thickness, approach_config.max_position_error});
    demo_1f::TrajectoryExecutor trajectory_executor(*node, config, robot_context);
    demo_1f::HandoverMotion handover_motion(config, robot_context, trajectory_executor);
    demo_1f::InitialPoseMotion initial_motion(config, robot_context, trajectory_executor);
    demo_1f::ApproachScene approach_scene(*node);
    demo_1f::ApproachCupMotion approach_motion(
        config, approach_config, robot_context, trajectory_executor, approach_scene);
    demo_1f::RefineCupMotion refine_motion(
        config, approach_config, robot_context, trajectory_executor, approach_scene);
    demo_1f::WithdrawCupMotion withdraw_motion(
        config, approach_config, robot_context, trajectory_executor, approach_scene);
    const auto pick_config = demo_1f::load_pick_config(*node);
    demo_1f::PickCupMotion pick_motion(
        config, approach_config, pick_config, robot_context, trajectory_executor, approach_scene);
    demo_1f::TaskManager task_manager(*node,
        robot_context,
        handover_motion,
        {{"/approach_cup", approach_motion},
            {"/refine_cup", refine_motion},
            {"/withdraw_cup", withdraw_motion},
            {"/pick_cup", pick_motion}}, &initial_motion);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    RCLCPP_INFO(
        node->get_logger(), "Handover demo ready for requests; startup performs no robot motion");
    while (rclcpp::ok()) {
      executor.spin_some();
      if (stop_requested) {
        task_manager.request_shutdown();
        if (task_manager.can_shutdown()) {
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (task_manager.demo_faulted() ||
        trajectory_executor.result().state == demo_1f::ExecutionState::StopUnconfirmed) {
      RCLCPP_ERROR(
          node->get_logger(), "Shutdown with STOP_UNCONFIRMED; inspect robot before restarting");
      exit_code = 2;
    }
    executor.remove_node(node);
  } catch (const std::exception & error) {
    std::fprintf(stderr, "handover_demo: %s\n", error.what());
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}
