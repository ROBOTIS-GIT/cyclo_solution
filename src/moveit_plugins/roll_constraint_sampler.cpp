#include <algorithm>
#include <cmath>
#include <moveit/constraint_samplers/constraint_sampler_manager.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace demo_1f {
namespace {
bool roll_constraint(const moveit_msgs::msg::OrientationConstraint & c) {
  const bool reverse_hand_frame =
      c.link_name == "base_link" &&
      (c.header.frame_id == "end_effector_l_link" || c.header.frame_id == "end_effector_r_link" ||
          c.header.frame_id == "gripper_l_tcp" || c.header.frame_id == "gripper_r_tcp") &&
      c.parameterization == c.XYZ_EULER_ANGLES && std::abs(c.orientation.x) < 1e-9 &&
      std::abs(c.orientation.z) < 1e-9;
  const bool roll_pitch_band = std::abs(c.orientation.y - std::sin(M_PI / 4)) < 1e-9 &&
                               std::abs(c.orientation.w - std::cos(M_PI / 4)) < 1e-9;
  const bool downward_half_space = std::abs(c.orientation.y) < 1e-9 &&
                                   std::abs(c.orientation.w - 1) < 1e-9 &&
                                   std::abs(c.absolute_x_axis_tolerance - M_PI / 2) < 1e-9 &&
                                   std::abs(c.absolute_y_axis_tolerance - M_PI) < 1e-9 &&
                                   std::abs(c.absolute_z_axis_tolerance - M_PI) < 1e-9;
  return reverse_hand_frame && (roll_pitch_band || downward_half_space);
}

class RollSampler final : public constraint_samplers::ConstraintSampler {
public:
  RollSampler(const planning_scene::PlanningSceneConstPtr & scene, const std::string & group)
      : ConstraintSampler(scene, group), checks_(scene->getRobotModel()) {}

  bool configure(const moveit_msgs::msg::Constraints & constraints) override {
    checks_.clear();
    if (!jmg_ || !checks_.add(constraints, scene_->getTransforms())) {
      return is_valid_ = false;
    }
    auto sampling = constraints;
    auto & orientations = sampling.orientation_constraints;
    orientations.erase(std::remove_if(orientations.begin(), orientations.end(), roll_constraint),
        orientations.end());
    // Sample the real TCP position/pose, then check ALL original constraints.
    // The stock sampler otherwise tries to solve arm IK for base_link.
    base_ = constraint_samplers::ConstraintSamplerManager::selectDefaultSampler(
        scene_, jmg_->getName(), sampling);
    return is_valid_ = true;
  }

  bool sample(moveit::core::RobotState & state,
      const moveit::core::RobotState & reference,
      unsigned int attempts) override {
    if (!is_valid_) {
      return false;
    }
    if (base_) {
      base_->setGroupStateValidityCallback(group_state_validity_callback_);
    }
    for (unsigned int i = 0; i < attempts; ++i) {
      if (base_) {
        if (!base_->sample(state, reference, 1)) {
          continue;
        }
      } else {
        state.setToRandomPositions(jmg_);
      }
      state.update();
      if (!checks_.decide(state).satisfied) {
        continue;
      }
      if (group_state_validity_callback_) {
        std::vector<double> positions;
        state.copyJointGroupPositions(jmg_, positions);
        if (!group_state_validity_callback_(&state, jmg_, positions.data())) {
          continue;
        }
      }
      return true;
    }
    return false;
  }

  const std::string & getName() const override {
    static const std::string name = "SG2TcpRollSampler";
    return name;
  }

private:
  kinematic_constraints::KinematicConstraintSet checks_;
  constraint_samplers::ConstraintSamplerPtr base_;
};
} // namespace

class RollSamplerAllocator final : public constraint_samplers::ConstraintSamplerAllocator {
public:
  bool canService(const planning_scene::PlanningSceneConstPtr &,
      const std::string & group,
      const moveit_msgs::msg::Constraints & constraints) const override {
    return (group == "arm_l" || group == "arm_r") &&
           std::any_of(constraints.orientation_constraints.begin(),
               constraints.orientation_constraints.end(),
               roll_constraint);
  }

  constraint_samplers::ConstraintSamplerPtr alloc(
      const planning_scene::PlanningSceneConstPtr & scene,
      const std::string & group,
      const moveit_msgs::msg::Constraints & constraints) override {
    auto sampler = std::make_shared<RollSampler>(scene, group);
    return sampler->configure(constraints) ? sampler : nullptr;
  }
};
} // namespace demo_1f
PLUGINLIB_EXPORT_CLASS(
    demo_1f::RollSamplerAllocator, constraint_samplers::ConstraintSamplerAllocator)
