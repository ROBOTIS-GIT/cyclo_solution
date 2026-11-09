// Offline URDF/FCL check: never starts a controller or sends robot commands.
#include "demo_1f/runtime/config.hpp"
#include <fstream>
#include <iostream>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <srdfdom/model.h>
#include <sstream>
#include <urdf_parser/urdf_parser.h>

std::string read(const char * path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error(std::string("Cannot read ") + path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

int validate(char ** argv) {
  auto node = std::make_shared<rclcpp::Node>("handover_demo");
  const auto config = demo_1f::load_config(*node);
  auto urdf = urdf::parseURDF(read(argv[1]));
  auto srdf = std::make_shared<srdf::Model>();
  if (!urdf || !srdf->initString(*urdf, read(argv[2]))) {
    return 2;
  }
  auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
  planning_scene::PlanningScene scene(model);
  const auto * group = model->getJointModelGroup(config.coordinated_group);
  if (!group || group->getMimicJointModels().size() != 6) {
    std::cerr << "Coordinated group must include all six passive gripper joints\n";
    return 1;
  }
  bool valid = true;
  for (const auto & [direction, sequence] : config.motion_sequences) {
    // Include the former closed preset, the new transfer opening and the mimic
    // limit, because an externally prepared giver need not start fully open.
    for (const double initial_gripper : {0.0, 0.8, 0.95, 1.0}) {
      std::cout << "Initial gripper position: " << initial_gripper << '\n';
      moveit::core::RobotState current(model);
      current.setToDefaultValues();
      for (const auto & [side, arm] : config.arms) {
        (void)side;
        current.setVariablePosition(arm.gripper_joint, initial_gripper);
      }
      current.update();
      for (const auto & step : sequence.steps) {
        auto goal = current;
        if (step.type == "dual_joint" || step.type == "dual_joint_gripper") {
          goal.setVariablePositions(config.arms.at("left").joints, step.left_target);
          goal.setVariablePositions(config.arms.at("right").joints, step.right_target);
          if (step.type == "dual_joint_gripper") {
            goal.setVariablePosition(
                config.arms.at(step.arm).gripper_joint, step.gripper_target[0]);
          }
        } else if (step.type == "joint_group") {
          goal.setVariablePositions(step.joint_names, step.target);
        } else if (step.type == "joint_gripper") {
          const auto & arm = config.arms.at(step.arm);
          goal.setVariablePositions(arm.joints, step.target);
          goal.setVariablePosition(arm.gripper_joint, step.gripper_target[0]);
        } else if (step.type == "gripper") {
          goal.setVariablePosition(config.arms.at(step.arm).gripper_joint, step.target[0]);
        } else {
          throw std::runtime_error("Unexpected preset step: " + step.type);
        }
        goal.update();
        // Exercise the same group-scoped setter used by planning, not just the
        // per-variable setters used to prepare the reference geometry above.
        const auto * step_group = step.type == "joint_group" ?
            model->getJointModelGroup(step.group) : group;
        if (!step_group) throw std::runtime_error("Unknown group: " + step.group);
        auto planned = current;
        std::vector<double> active_positions;
        for (const auto * joint : step_group->getActiveJointModels()) {
          for (const auto & name : joint->getVariableNames()) {
            active_positions.push_back(goal.getVariablePosition(name));
          }
        }
        planned.setJointGroupActivePositions(step_group, active_positions);
        for (const auto * mimic : step_group->getMimicJointModels()) {
          const auto * source = mimic->getMimic();
          const double expected =
              planned.getVariablePosition(source->getName()) * mimic->getMimicFactor() +
              mimic->getMimicOffset();
          if (std::abs(planned.getVariablePosition(mimic->getName()) - expected) > 1e-9) {
            std::cerr << "Planning left stale mimic state: " << mimic->getName() << '\n';
            valid = false;
          }
        }
        bool step_valid = true;
        for (int i = 0; i <= 200; ++i) {
          auto sample = current;
          current.interpolate(goal, i / 200.0, sample);
          sample.update();
          collision_detection::CollisionRequest request;
          request.contacts = true;
          request.max_contacts = 20;
          collision_detection::CollisionResult result;
          scene.checkCollision(request, result, sample);
          if (result.collision || !sample.satisfiesBounds(1e-6)) {
            std::cerr << direction << "/" << step.name << " invalid at " << i << "/200\n";
            for (const auto & [pair, contacts] : result.contacts) {
              (void)contacts;
              std::cerr << "  " << pair.first << " / " << pair.second << '\n';
            }
            valid = step_valid = false;
            break;
          }
        }
        std::cout << direction << "/" << step.name << ": " << (step_valid ? "PASS" : "FAIL")
                  << '\n';
        if (step.name == "face_each_other") {
          const auto left = goal.getGlobalLinkTransform(config.arms.at("left").tcp);
          const auto right = goal.getGlobalLinkTransform(config.arms.at("right").tcp);
          const double crossed = std::abs(left.linear().col(0).dot(right.linear().col(0)));
          std::cout << direction << " ready jaw-axis dot: " << crossed << '\n';
          if (crossed > 0.001) {
            valid = false;
          }
          for (const auto & [side, arm] : config.arms) {
            const auto transform = goal.getGlobalLinkTransform("arm_base_link").inverse() *
                                   goal.getGlobalLinkTransform(arm.tcp);
            std::cout << side << " TCP in arm_base_link: " << transform.translation().transpose()
                      << " forward: " << (-transform.rotation().col(2)).transpose() << '\n';
            const double sign = side == "left" ? 1.0 : -1.0;
            if ((transform.translation() - Eigen::Vector3d(0.33, sign * 0.14, -0.3)).norm() >
                    0.001 ||
                (-transform.rotation().col(2) - Eigen::Vector3d(0, -sign, 0)).norm() > 0.001) {
              valid = false;
            }
          }
        }
        if (step.name == "open_approach") {
          const auto left = goal.getGlobalLinkTransform(config.arms.at("left").tcp);
          const auto right = goal.getGlobalLinkTransform(config.arms.at("right").tcp);
          const double gap = (left.translation() - right.translation()).norm();
          const double facing = (-left.linear().col(2)).dot(-right.linear().col(2));
          const double crossed = std::abs(left.linear().col(0).dot(right.linear().col(0)));
          std::cout << direction << " approach TCP gap: " << gap << " m; facing dot: " << facing
                    << "; jaw-axis dot: " << crossed << '\n';
          // Receiver advances 20 mm along its local -Z, past the giver TCP.
          const auto & receiver = direction == "left_to_right" ? right : left;
          const auto & giver = direction == "left_to_right" ? left : right;
          const Eigen::Vector3d expected_offset = -0.02 * receiver.linear().col(2);
          if ((receiver.translation() - giver.translation() - expected_offset).norm() > 0.001 ||
              facing > -0.999 || crossed > 0.001) {
            valid = false;
          }
          for (const auto & [side, arm] : config.arms) {
            const auto ready_rotation = current.getGlobalLinkTransform(arm.tcp).rotation();
            const auto approach_rotation = goal.getGlobalLinkTransform(arm.tcp).rotation();
            const double angle =
                Eigen::AngleAxisd(ready_rotation.transpose() * approach_rotation).angle();
            if (angle > 0.001) {
              std::cerr << direction << "/" << side
                        << " changes TCP orientation on approach: " << angle << " rad\n";
              valid = false;
            }
          }
        }
        current = goal;
      }
    }
  }
  return valid ? 0 : 1;
}

int main(int argc, char ** argv) {
  if (argc < 3) {
    return 2;
  }
  rclcpp::init(argc, argv);
  const auto result = validate(argv);
  rclcpp::shutdown();
  return result;
}
