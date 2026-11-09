#pragma once
#include "demo_1f/core/types.hpp"
#include <rclcpp/rclcpp.hpp>

namespace demo_1f {
Config load_config(rclcpp::Node & node);
std::string validate_motion_sequence(const MotionSequence & sequence, const Config & config);
} // namespace demo_1f
