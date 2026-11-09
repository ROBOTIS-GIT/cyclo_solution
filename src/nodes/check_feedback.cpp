// Read-only diagnostic: no action clients, action servers or motion publishers.
#include "demo_1f/runtime/config.hpp"
#include "demo_1f/runtime/state/robot_context.hpp"
#include <iomanip>
#include <iostream>
#include <thread>

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  int code = 1;
  try {
    auto node = std::make_shared<rclcpp::Node>("handover_feedback_check");
    const auto config = demo_1f::load_config(*node);
    demo_1f::RobotContext context(*node, config);
    const auto started = demo_1f::Clock::now();
    while (rclcpp::ok() && demo_1f::elapsed(started) < 5.0) {
      rclcpp::spin_some(node);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::string reason;
    bool ready = context.ready(reason);
    if (!ready) {
      std::cout << reason << '\n';
    }
    std::cout << "Gripper stop mode=" << config.gripper_stop_mode
              << ", window=" << config.gripper_stop_window
              << " s, position range limit=" << config.gripper_stop_position_tolerance << " rad\n";
    const auto state = context.snapshot();
    for (const auto & [side, arm] : config.arms) {
      (void)side;
      auto joints = arm.joints;
      joints.push_back(arm.gripper_joint);
      for (const auto & name : joints) {
        const auto it = state.joints.find(name);
        if (it == state.joints.end()) {
          ready = false;
          continue;
        }
        const auto & joint = it->second;
        const bool stopped = context.joint_stationary(name, joint, reason);
        ready = ready && stopped;
        if (name == arm.gripper_joint || !stopped) {
          std::cout << std::setprecision(9) << name << ": position=" << joint.position
                    << ", reported_velocity="
                    << (joint.velocity ? std::to_string(*joint.velocity) : "unknown")
                    << ", position_window_span="
                    << (joint.position_window_span ? std::to_string(*joint.position_window_span)
                                                   : "incomplete")
                    << ", stationary=" << (stopped ? "true" : "false") << " " << reason << '\n';
        }
      }
    }
    std::cout << (ready ? "PASS" : "FAIL") << ": feedback check only; no motion commands sent\n";
    code = ready ? 0 : 1;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
  }
  rclcpp::shutdown();
  return code;
}
