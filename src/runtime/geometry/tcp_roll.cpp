#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tf2/LinearMath/Matrix3x3.h>

namespace demo_1f {
std::optional<double> tcp_roll(const geometry_msgs::msg::Quaternion & q) {
  const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(norm) || std::abs(norm - 1) > .001) {
    return {};
  }
  tf2::Matrix3x3 r(tf2::Quaternion(q.x, q.y, q.z, q.w));
  // Avoid the Euler singularity: roll is not identifiable with a vertical forward axis.
  if (std::hypot(r[0][2], r[1][2]) < std::sin(M_PI / 180)) {
    return {};
  }
  return std::atan2(r[2][1], r[2][0]);
}

std::optional<double> tcp_yaw(const geometry_msgs::msg::Quaternion & q) {
  if (!tcp_roll(q)) {
    return {};
  }
  const tf2::Matrix3x3 r(tf2::Quaternion(q.x, q.y, q.z, q.w));
  return std::atan2(-r[1][2], -r[0][2]); // Hand forward=-TCP Z.
}

bool tcp_yaw_is_inward(double yaw, const std::string & arm) {
  if (!std::isfinite(yaw) || (arm != "left" && arm != "right")) {
    return false;
  }
  const double inward = arm == "left" ? -yaw : yaw;
  return inward >= -1e-9 && inward <= M_PI / 2 + 1e-9;
}

moveit_msgs::msg::OrientationConstraint tcp_inward_yaw_constraint(
    const std::string & tcp, const std::string & body_frame, const std::string & arm) {
  if (tcp.empty() || body_frame.empty() || (arm != "left" && arm != "right")) {
    throw std::invalid_argument("Invalid TCP inward yaw constraint");
  }
  const double center = arm == "left" ? -M_PI / 4 : M_PI / 4;
  return tcp_yaw_constraint(tcp, body_frame, center, M_PI / 4 - M_PI / 180);
}

moveit_msgs::msg::OrientationConstraint tcp_yaw_constraint(
    const std::string & tcp, const std::string & body_frame, double center, double limit) {
  if (tcp.empty() || body_frame.empty() || !std::isfinite(center) || !std::isfinite(limit) ||
      limit <= 0 || limit > M_PI) {
    throw std::invalid_argument("Invalid TCP heading constraint");
  }
  tf2::Quaternion heading, axes;
  heading.setRPY(0, 0, center);
  axes.setRPY(0, -M_PI / 2, 0);
  const auto q = heading * axes;
  moveit_msgs::msg::OrientationConstraint c;
  c.header.frame_id = body_frame;
  c.link_name = tcp;
  c.orientation.x = q.x();
  c.orientation.y = q.y();
  c.orientation.z = q.z();
  c.orientation.w = q.w();
  // In SG2 TCP axes, the XYZ error is Rx(yaw-center)*Ry(pitch)*Rz(-roll).
  // Restrict X (heading), not the raw TCP quaternion's Z Euler angle.
  c.parameterization = c.XYZ_EULER_ANGLES;
  c.absolute_x_axis_tolerance = limit;
  c.absolute_y_axis_tolerance = M_PI;
  c.absolute_z_axis_tolerance = M_PI;
  c.weight = 1;
  return c;
}

std::optional<double> tcp_pitch(const geometry_msgs::msg::Quaternion & q) {
  const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(norm) || std::abs(norm - 1) > .001) {
    return {};
  }
  const tf2::Matrix3x3 r(tf2::Quaternion(q.x, q.y, q.z, q.w));
  // Hand forward=-TCP Z, so body ZYX pitch is asin(-forward.z).
  return std::asin(std::clamp(r[2][2], -1.0, 1.0));
}

moveit_msgs::msg::OrientationConstraint tcp_roll_constraint(
    const std::string & tcp, const std::string & body_frame, double limit, double pitch_limit) {
  if (tcp.empty() || body_frame.empty() || !std::isfinite(limit) || limit <= 0 || limit > M_PI ||
      !std::isfinite(pitch_limit) || pitch_limit < 0 || pitch_limit >= M_PI / 2) {
    throw std::invalid_argument("Invalid TCP roll constraint");
  }
  moveit_msgs::msg::OrientationConstraint c;
  // Reverse the frames deliberately. With hand axes C=Ry(pi/2), MoveIt's
  // intrinsic XYZ error is (R_body_tcp*C)^T, whose X/Y angles are -body ZYX roll/pitch.
  // A normal world-frame OrientationConstraint on TCP does NOT implement this.
  c.header.frame_id = tcp;
  c.link_name = body_frame;
  c.orientation.y = std::sin(M_PI / 4);
  c.orientation.w = std::cos(M_PI / 4);
  c.parameterization = c.XYZ_EULER_ANGLES;
  c.absolute_x_axis_tolerance = limit;
  c.absolute_y_axis_tolerance = pitch_limit > 0 ? pitch_limit : 89 * M_PI / 180;
  c.absolute_z_axis_tolerance = M_PI;
  c.weight = 1;
  return c;
}

moveit_msgs::msg::PositionConstraint tcp_min_pitch_constraint(const std::string & tcp,
    const std::string & body_frame,
    const geometry_msgs::msg::Point & goal,
    double position_radius,
    double minimum_pitch) {
  if (tcp.empty() || body_frame.empty() || !std::isfinite(position_radius) ||
      position_radius <= 0 || !std::isfinite(minimum_pitch) || minimum_pitch <= 0 ||
      minimum_pitch >= M_PI / 2 || !std::isfinite(goal.x) || !std::isfinite(goal.y) ||
      !std::isfinite(goal.z)) {
    throw std::invalid_argument("Invalid TCP minimum pitch constraint");
  }
  constexpr double forward_distance = 1.0;
  moveit_msgs::msg::PositionConstraint c;
  c.header.frame_id = body_frame;
  c.link_name = tcp;
  c.target_point_offset.z = -forward_distance; // Virtual point; no geometry is added to the robot.
  // p lies in the TCP goal sphere, q=p+forward. q.z <= goal.z-radius-sin(min)
  // implies forward.z <= -sin(min), including at the lowest allowed TCP position.
  const double lower = goal.z - forward_distance - position_radius;
  const double upper = goal.z - forward_distance * std::sin(minimum_pitch) - position_radius;
  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions = {2 * (forward_distance + position_radius),
      2 * (forward_distance + position_radius),
      upper - lower};
  geometry_msgs::msg::Pose pose;
  pose.position = goal;
  pose.position.z = (lower + upper) / 2;
  pose.orientation.w = 1;
  c.constraint_region.primitives.push_back(box);
  c.constraint_region.primitive_poses.push_back(pose);
  c.weight = 1;
  return c;
}

moveit_msgs::msg::OrientationConstraint tcp_downward_constraint(
    const std::string & tcp, const std::string & body_frame) {
  auto c = tcp_roll_constraint(tcp, body_frame, M_PI / 2);
  c.orientation.y = 0;
  c.orientation.w = 1;
  c.absolute_y_axis_tolerance = M_PI;
  // Reverse-frame error is R_body_tcp^T. Its XYZ X angle lies within +/-pi/2
  // iff error(2,2) >= 0, i.e. TCP Z points up and hand forward=-TCP Z points down.
  // Combined with the existing roll/pitch band, yaw remains unrestricted.
  return c;
}

geometry_msgs::msg::Quaternion leveled_tcp(
    const geometry_msgs::msg::Quaternion & orientation, bool horizontal, double downward_pitch) {
  if (!tcp_roll(orientation)) {
    throw std::invalid_argument("Cannot level a vertical/invalid TCP orientation");
  }
  const tf2::Matrix3x3 r(
      tf2::Quaternion(orientation.x, orientation.y, orientation.z, orientation.w));
  auto forward = -r.getColumn(2);
  if (horizontal) {
    forward.setZ(0);
    forward.normalize();
    forward *= std::cos(downward_pitch);
    forward.setZ(-std::sin(downward_pitch));
  }
  const auto left = tf2::Vector3(0, 0, 1).cross(forward).normalized();
  const auto up = forward.cross(left);
  tf2::Quaternion q;
  tf2::Matrix3x3(up.x(),
      left.x(),
      -forward.x(),
      up.y(),
      left.y(),
      -forward.y(),
      up.z(),
      left.z(),
      -forward.z())
      .getRotation(q);
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}
} // namespace demo_1f
