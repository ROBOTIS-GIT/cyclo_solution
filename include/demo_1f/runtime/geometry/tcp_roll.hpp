#pragma once
#include <geometry_msgs/msg/quaternion.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <optional>
#include <string>

namespace demo_1f {
// SG2 TCP: forward=-Z, left=+Y, up(camera side)=+X.
// Body-frame ZYX roll about the forward direction; undefined at vertical forward.
std::optional<double> tcp_roll(const geometry_msgs::msg::Quaternion & orientation);
std::optional<double> tcp_yaw(const geometry_msgs::msg::Quaternion & orientation);
// Heading relative to robot +X; the sign of inward heading depends on the arm.
bool tcp_yaw_is_inward(double yaw, const std::string & arm);
moveit_msgs::msg::OrientationConstraint tcp_inward_yaw_constraint(
    const std::string & tcp, const std::string & body_frame, const std::string & arm);
moveit_msgs::msg::OrientationConstraint tcp_yaw_constraint(
    const std::string & tcp, const std::string & body_frame, double center, double limit);
std::optional<double> tcp_pitch(const geometry_msgs::msg::Quaternion & orientation);
geometry_msgs::msg::Quaternion leveled_tcp(const geometry_msgs::msg::Quaternion & orientation,
    bool horizontal = false,
    double downward_pitch = 0);
moveit_msgs::msg::OrientationConstraint tcp_roll_constraint(
    const std::string & tcp, const std::string & body_frame, double limit, double pitch_limit = 0);
// Bound the height of a virtual point 1 m along hand-forward. Together with the
// TCP position sphere this enforces minimum downward pitch for any yaw/roll.
moveit_msgs::msg::PositionConstraint tcp_min_pitch_constraint(const std::string & tcp,
    const std::string & body_frame,
    const geometry_msgs::msg::Point & goal,
    double position_radius,
    double minimum_pitch);
// Intersect with the symmetric pitch band to allow downward forward direction only.
moveit_msgs::msg::OrientationConstraint tcp_downward_constraint(
    const std::string & tcp, const std::string & body_frame);
} // namespace demo_1f
