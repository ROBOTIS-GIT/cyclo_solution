#include "demo_1f/runtime/execution/trajectory_smoother.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <moveit/utils/logger.hpp>
#include <rclcpp/duration.hpp>
#include <rcutils/logging.h>
#include <ruckig/ruckig.hpp>
#include <set>
#include <stdexcept>

namespace demo_1f {
void initialize_moveit_logging() {
  const int level = rcutils_logging_get_default_logger_level();
  rcutils_logging_set_default_logger_level(RCUTILS_LOG_SEVERITY_ERROR);
  // No ROS context exists yet, so the logger retains no hidden node. Restore
  // normal logging immediately; only the expected fallback warnings are muted.
  (void)moveit::getLogger("demo_1f");
  rcutils_logging_set_default_logger_level(level);
}

namespace {
using Positions = std::vector<double>;
using Point = trajectory_msgs::msg::JointTrajectoryPoint;

double seconds(const builtin_interfaces::msg::Duration & time) {
  return rclcpp::Duration(time).seconds();
}

double distance(const Positions & first, const Positions & second) {
  double result = 0;
  for (size_t j = 0; j < first.size(); ++j) {
    result = std::max(result, std::abs(first[j] - second[j]));
  }
  return result;
}

std::vector<size_t> simplify(const std::vector<Positions> & positions, double tolerance) {
  std::vector<size_t> indices;
  std::function<void(size_t, size_t)> visit = [&](size_t first, size_t last) {
    double norm = 0;
    for (size_t j = 0; j < positions[first].size(); ++j) {
      norm += std::pow(positions[last][j] - positions[first][j], 2);
    }
    double largest = tolerance;
    size_t split = first;
    for (size_t i = first + 1; i < last; ++i) {
      double projection = 0;
      for (size_t j = 0; j < positions[first].size(); ++j) {
        projection +=
            (positions[i][j] - positions[first][j]) * (positions[last][j] - positions[first][j]);
      }
      projection = norm > 0 ? std::clamp(projection / norm, 0.0, 1.0) : 0;
      double error = 0;
      for (size_t j = 0; j < positions[first].size(); ++j) {
        error = std::max(error,
            std::abs(positions[i][j] - positions[first][j] -
                     projection * (positions[last][j] - positions[first][j])));
      }
      if (error > largest) {
        largest = error;
        split = i;
      }
    }
    if (split != first) {
      visit(first, split);
      visit(split, last);
    } else {
      indices.push_back(last);
    }
  };
  indices.push_back(0);
  if (positions.size() > 1) {
    visit(0, positions.size() - 1);
  }
  return indices;
}

struct Sample {
  double position, velocity, acceleration, jerk;
};

Sample quintic(const Point & first, const Point & last, size_t joint, double u) {
  const double dt = seconds(last.time_from_start) - seconds(first.time_from_start);
  const double c0 = first.positions[joint];
  const double c1 = first.velocities[joint] * dt;
  const double c2 = 0.5 * first.accelerations[joint] * dt * dt;
  const double d = last.positions[joint] - c0 - c1 - c2;
  const double v = last.velocities[joint] * dt - c1 - 2 * c2;
  const double a = last.accelerations[joint] * dt * dt - 2 * c2;
  const double c3 = 10 * d - 4 * v + a / 2;
  const double c4 = -15 * d + 7 * v - a;
  const double c5 = 6 * d - 3 * v + a / 2;
  return {c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5)))),
      (c1 + u * (2 * c2 + u * (3 * c3 + u * (4 * c4 + u * 5 * c5)))) / dt,
      (2 * c2 + u * (6 * c3 + u * (12 * c4 + u * 20 * c5))) / (dt * dt),
      (6 * c3 + u * (24 * c4 + u * 60 * c5)) / (dt * dt * dt)};
}
} // namespace

MotionLimits effective_limits(const Config & config,
    const std::string & name,
    const Step & step,
    const moveit::core::RobotModelConstPtr & model) {
  MotionLimits limits;
  if (config.motion_limits.count(name)) {
    limits = config.motion_limits.at(name);
  }
  if (model) {
    const auto & names = model->getVariableNames();
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      throw std::runtime_error("Unknown trajectory joint: " + name);
    }
    const auto & bounds = model->getVariableBounds(name);
    if (bounds.velocity_bounded_) {
      limits.velocity = std::min(limits.velocity, bounds.max_velocity_);
    }
  }
  limits.velocity *= step.velocity_scale > 0 ? step.velocity_scale : config.velocity_scale;
  limits.acceleration *=
      step.acceleration_scale > 0 ? step.acceleration_scale : config.acceleration_scale;
  if (!std::isfinite(limits.velocity) || !std::isfinite(limits.acceleration) ||
      !std::isfinite(limits.jerk) || limits.velocity <= 0 || limits.acceleration <= 0 ||
      limits.jerk <= 0) {
    throw std::runtime_error("Invalid motion limits: " + name);
  }
  return limits;
}

SmoothPath smooth_segments(const std::vector<moveit_msgs::msg::RobotTrajectory> & segments,
    const std::vector<Step> & steps,
    const Config & config,
    const moveit::core::RobotModelConstPtr & model) {
  if (segments.empty() || segments.size() != steps.size()) {
    throw std::runtime_error("Missing planned motion segments");
  }
  SmoothPath output;
  auto & trajectory = output.trajectory.joint_trajectory;
  trajectory.joint_names = segments.front().joint_trajectory.joint_names;
  const size_t dofs = trajectory.joint_names.size();
  if (std::set<std::string>(trajectory.joint_names.begin(), trajectory.joint_names.end()).size() !=
      dofs) {
    throw std::runtime_error("Duplicate planned joint names");
  }
  if (dofs == 0) {
    throw std::runtime_error("Empty planned joint list");
  }
  std::vector<Positions> positions;
  std::vector<size_t> owners, boundaries;
  Positions planned_endpoint;
  std::set<std::string> previous_held_joints;
  for (size_t segment = 0; segment < segments.size(); ++segment) {
    const auto & source = segments[segment].joint_trajectory;
    if (source.points.empty() || source.joint_names.size() != dofs ||
        !segments[segment].multi_dof_joint_trajectory.points.empty()) {
      throw std::runtime_error("Invalid coordinated trajectory");
    }
    std::vector<Positions> reordered;
    for (const auto & point : source.points) {
      if (point.positions.size() != dofs) {
        throw std::runtime_error("Malformed planned point");
      }
      Positions values;
      for (const auto & name : trajectory.joint_names) {
        const auto found = std::find(source.joint_names.begin(), source.joint_names.end(), name);
        if (found == source.joint_names.end()) {
          throw std::runtime_error("Joint lists differ between segments");
        }
        const double value = point.positions[found - source.joint_names.begin()];
        if (!std::isfinite(value)) {
          throw std::runtime_error("Nonfinite planned position");
        }
        values.push_back(value);
      }
      reordered.push_back(std::move(values));
    }
    if (!planned_endpoint.empty() && distance(planned_endpoint, reordered.front()) > 1e-6) {
      throw std::runtime_error("Discontinuous planning start between segments");
    }
    planned_endpoint = reordered.back();
    std::set<std::string> held_joints;
    const auto & step = steps[segment];
    for (const auto & [side, arm] : config.arms) {
      if (step.type == "joint_gripper" && side != step.arm) {
        held_joints.insert(arm.joints.begin(), arm.joints.end());
      }
      if (step.type == "dual_joint" || side != step.arm) {
        held_joints.insert(arm.gripper_joint);
      }
    }
    for (size_t joint = 0; joint < dofs; ++joint) {
      const auto & name = trajectory.joint_names[joint];
      if (held_joints.count(name)) {
        const double planned_start = reordered.front()[joint];
        const double held_position = positions.empty() ? planned_start : positions.back()[joint];
        for (auto & point : reordered) {
          if (std::abs(point[joint] - planned_start) > std::min(step.tolerance, 0.001) + 1e-8) {
            throw std::runtime_error("Planned path moves a held joint: " + name);
          }
          // Planner tolerances allow small drift. Remove it before blending so
          // held joints stay constant; the modified path is collision-checked below.
          point[joint] = held_position;
        }
      } else if (!positions.empty() && previous_held_joints.count(name)) {
        // The next segment starts at the corrected held endpoint, then may move.
        reordered.front()[joint] = positions.back()[joint];
      }
    }
    previous_held_joints = std::move(held_joints);
    for (const auto index : simplify(reordered, config.path_simplification_tolerance)) {
      if (!positions.empty() && distance(positions.back(), reordered[index]) < 1e-9) {
        continue;
      }
      positions.push_back(reordered[index]);
      owners.push_back(segment);
    }
    boundaries.push_back(positions.size() - 1);
  }
  std::vector<std::vector<MotionLimits>> limits(positions.size());
  std::vector<double> estimates(positions.size(), 0.02);
  for (size_t i = 1; i < positions.size(); ++i) {
    for (size_t j = 0; j < dofs; ++j) {
      const auto bound =
          effective_limits(config, trajectory.joint_names[j], steps[owners[i]], model);
      limits[i].push_back(bound);
      const double delta = std::abs(positions[i][j] - positions[i - 1][j]);
      estimates[i] = std::max({estimates[i],
          delta / bound.velocity,
          std::sqrt(delta / bound.acceleration),
          std::cbrt(delta / bound.jerk)});
    }
  }
  std::vector<Positions> velocities(positions.size(), Positions(dofs, 0.0));
  for (size_t i = 1; i + 1 < positions.size(); ++i) {
    for (size_t j = 0; j < dofs; ++j) {
      const double before = (positions[i][j] - positions[i - 1][j]) / estimates[i];
      const double after = (positions[i + 1][j] - positions[i][j]) / estimates[i + 1];
      if (before * after > 0) {
        const double tangent = before * after / (before + after);
        const double cap = 0.5 * std::min(limits[i][j].velocity, limits[i + 1][j].velocity);
        velocities[i][j] = std::clamp(tangent, -cap, cap);
      }
    }
  }
  Point first;
  first.positions = positions.front();
  first.velocities = first.accelerations = Positions(dofs, 0.0);
  trajectory.points.push_back(first);
  output.point_steps.push_back(0);
  std::vector<double> node_times(positions.size(), 0.0);
  double time = 0;
  ruckig::Ruckig<ruckig::DynamicDOFs> generator(dofs);
  for (size_t i = 1; i < positions.size(); ++i) {
    ruckig::InputParameter<ruckig::DynamicDOFs> input(dofs);
    input.current_position = positions[i - 1];
    input.current_velocity = velocities[i - 1];
    input.current_acceleration = Positions(dofs, 0.0);
    input.target_position = positions[i];
    input.target_velocity = velocities[i];
    input.target_acceleration = Positions(dofs, 0.0);
    for (size_t j = 0; j < dofs; ++j) {
      input.max_velocity[j] = limits[i][j].velocity * 0.98;
      input.max_acceleration[j] = limits[i][j].acceleration * 0.95;
      // Leave interpolation margin around jerk switches in the controller spline.
      input.max_jerk[j] = limits[i][j].jerk * 0.5;
    }
    if (steps[owners[i]].type == "gripper") {
      input.minimum_duration = steps[owners[i]].duration;
    }
    ruckig::Trajectory<ruckig::DynamicDOFs> curve(dofs);
    if (generator.calculate(input, curve) < ruckig::Result::Working) {
      throw std::runtime_error("Ruckig could not connect planned waypoints");
    }
    const double duration = curve.get_duration();
    const size_t samples =
        std::max<size_t>(1, std::ceil(duration / config.trajectory_sample_period));
    for (size_t sample = 1; sample <= samples; ++sample) {
      const double local_time = duration * sample / samples;
      Point point;
      point.positions.resize(dofs);
      point.velocities.resize(dofs);
      point.accelerations.resize(dofs);
      curve.at_time(local_time, point.positions, point.velocities, point.accelerations);
      point.time_from_start = rclcpp::Duration::from_seconds(time + local_time);
      trajectory.points.push_back(std::move(point));
      output.point_steps.push_back(owners[i]);
    }
    time += duration;
    node_times[i] = time;
  }
  if (trajectory.points.size() == 1) {
    trajectory.points.push_back(first);
    trajectory.points.back().time_from_start = rclcpp::Duration::from_seconds(
        steps.back().type == "gripper" ? std::max(steps.back().duration, 0.02) : 0.02);
    output.point_steps.push_back(steps.size() - 1);
  }
  for (auto boundary : boundaries) {
    output.segment_end_times.push_back(node_times[boundary]);
  }
  return output;
}

void validate_smooth_path(const SmoothPath & path,
    const std::vector<Step> & steps,
    const Config & config,
    planning_scene::PlanningScene & scene) {
  const auto & trajectory = path.trajectory.joint_trajectory;
  auto state = scene.getCurrentState();
  for (size_t i = 1; i < trajectory.points.size(); ++i) {
    const auto & first = trajectory.points[i - 1];
    const auto & last = trajectory.points[i];
    const double dt = seconds(last.time_from_start) - seconds(first.time_from_start);
    if (!(dt > 0)) {
      throw std::runtime_error("Non-increasing trajectory timestamps");
    }
    const size_t samples = std::max<size_t>(5, std::ceil(dt / 0.002));
    for (size_t sample = 0; sample <= samples; ++sample) {
      const double u = double(sample) / samples;
      for (size_t joint = 0; joint < trajectory.joint_names.size(); ++joint) {
        const auto & name = trajectory.joint_names[joint];
        const auto bound =
            effective_limits(config, name, steps[path.point_steps[i]], scene.getRobotModel());
        const auto value = quintic(first, last, joint, u);
        if (!std::isfinite(value.position) || !std::isfinite(value.velocity) ||
            !std::isfinite(value.acceleration) || !std::isfinite(value.jerk) ||
            std::abs(value.velocity) > bound.velocity + 1e-5 ||
            std::abs(value.acceleration) > bound.acceleration + 1e-5 ||
            std::abs(value.jerk) > bound.jerk + 1e-3) {
          throw std::runtime_error(
              "Smoothed trajectory exceeds a motion limit: " + name +
              " v=" + std::to_string(value.velocity) + "/" + std::to_string(bound.velocity) +
              " a=" + std::to_string(value.acceleration) + "/" +
              std::to_string(bound.acceleration) + " j=" + std::to_string(value.jerk) + "/" +
              std::to_string(bound.jerk));
        }
        state.setVariablePosition(name, value.position);
      }
      state.update();
      if (!state.satisfiesBounds(1e-8)) {
        throw std::runtime_error("Smoothed path exceeds joint bounds");
      }
      if (scene.isStateColliding(state, config.coordinated_group)) {
        throw std::runtime_error("Smoothed path is in collision");
      }
      // A giver gripper must stay fixed throughout the entire approach.
      const auto & step = steps[path.point_steps[i]];
      for (const auto & [side, arm] : config.arms) {
        if (step.type == "joint_gripper" && side != step.arm) {
          for (const auto & joint : arm.joints) {
            const auto index =
                std::find(trajectory.joint_names.begin(), trajectory.joint_names.end(), joint);
            if (index == trajectory.joint_names.end() ||
                std::abs(
                    state.getVariablePosition(joint) -
                    trajectory.points.front().positions[index - trajectory.joint_names.begin()]) >
                    std::min(step.tolerance, 0.001)) {
              throw std::runtime_error("Held arm moved during smoothing");
            }
          }
        }
        if (step.type == "dual_joint" || side != step.arm) {
          const auto index = std::find(
              trajectory.joint_names.begin(), trajectory.joint_names.end(), arm.gripper_joint);
          if (index != trajectory.joint_names.end() &&
              std::abs(
                  state.getVariablePosition(arm.gripper_joint) -
                  trajectory.points.front().positions[index - trajectory.joint_names.begin()]) >
                  0.001) {
            throw std::runtime_error("Held gripper moved during smoothing");
          }
        }
      }
    }
  }
}
} // namespace demo_1f
