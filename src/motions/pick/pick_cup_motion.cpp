#include "demo_1f/motions/pick/pick_cup_motion.hpp"
#include "demo_1f/runtime/geometry/tcp_roll.hpp"
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace demo_1f {
PickCupMotion::PickCupMotion(const Config & robot,
    const ApproachConfig & config,
    const PickCupConfig & pick,
    RobotContext & context,
    ExecutorPort & executor,
    ApproachScenePort & scene)
    : CupMotion(robot, config, context, executor, scene), observer_(context, config), pick_(pick) {}

void PickCupMotion::reset() {}

bool PickCupMotion::move(const std::string & name, const geometry_msgs::msg::Pose & pose) {
  Step s;
  s.name = name;
  s.type = "pose";
  s.arm = arm_;
  s.frame = robot_.planning_frame;
  s.velocity_scale = pick_.speed;
  s.acceleration_scale = .08;
  s.timeout = 45;
  s.position_only = true; // OMPL: position plus independent physical hand angle bounds.
  s.tcp_roll_limit = 10 * M_PI / 180;
  s.tcp_pitch_limit = 10 * M_PI / 180;
  s.tcp_yaw_limit = 15 * M_PI / 180;
  const auto & p = pose.position;
  const auto & q = pose.orientation;
  s.target = {p.x, p.y, p.z, q.x, q.y, q.z, q.w};
  phase(name);
  approach_target_.header.frame_id = robot_.planning_frame;
  approach_target_.pose = pose;
  return execute_step(s);
}

bool PickCupMotion::open_gripper() {
  Step s;
  s.name = "PICK_OPEN";
  s.type = "gripper";
  s.arm = arm_;
  s.target = {0.0};
  s.duration = 1.0;
  s.timeout = 8;
  phase(s.name);
  return execute_step(s);
}

bool PickCupMotion::apply_scene() {
  moveit_msgs::msg::PlanningScene scene;
  scene.is_diff = true;
  scene.robot_state.is_diff = true;
  moveit_msgs::msg::CollisionObject cup;
  cup.id = "demo_place_cup";
  cup.header.frame_id = robot_.planning_frame;
  cup.operation = cup.ADD;
  shape_msgs::msg::SolidPrimitive shape;
  shape.type = shape.CYLINDER;
  shape.dimensions = {config_.cup_height, config_.inner_radius + config_.wall_thickness};
  cup.primitives = {shape};
  cup.primitive_poses = {body_};
  const std::string prefix = arm_ == "left" ? "gripper_l_" : "gripper_r_";
  const std::vector<std::string> fingers = {prefix + "rh_p12_rn_r2", prefix + "rh_p12_rn_l2"};
  scene.world.collision_objects.push_back(cup);
  // Model the table from the measured rim and configured height; never delete external objects.
  moveit_msgs::msg::CollisionObject table;
  table.id = "demo_pick_table";
  table.header.frame_id = robot_.planning_frame;
  table.operation = table.ADD;
  shape_msgs::msg::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions = {.4, .4, .04};
  auto pose = body_;
  pose.position.z -= config_.cup_height / 2 + .02;
  table.primitives = {box};
  table.primitive_poses = {pose};
  scene.world.collision_objects.push_back(table);
  scene.allowed_collision_matrix.entry_names = {cup.id, fingers[0], fingers[1]};
  scene.allowed_collision_matrix.entry_values.resize(3);
  for (auto & row : scene.allowed_collision_matrix.entry_values) {
    row.enabled.resize(3, false);
  }
  phase("PICK_SCENE");
  scene_pending_ = scene_.apply(scene);
  if (!scene_pending_) {
    fail("SCENE_UNAVAILABLE", "Cannot update cup grasp scene");
  }
  return scene_pending_;
}

void PickCupMotion::finish_approach() {
  const auto tcp = context_.pose_of(robot_.arms.at(arm_).tcp);
  if (!tcp || !tcp_yaw(tcp->pose.orientation)) {
    fail("TCP_TF_MISSING", "Cannot save measured pregrasp orientation");
    return;
  }
  pregrasp_ = tcp->pose;
  approach_target_.pose = pregrasp_;
  at_observation_pose_ = true;
  result_.state = "SUCCEEDED";
  result_.sequence_completed = true;
  result_.detail = "Horizontal pregrasp reached; stopped before contact. No grasp command issued";
  phase("PICK_APPROACH_READY");
}

void PickCupMotion::tick() {
  if (!running()) {
    return;
  }
  if (phase_ == "PICK_APPROACH") {
    const auto error = pick_.validate(config_.cup_height);
    if (!error.empty()) {
      fail("PICK_CONFIG_INVALID", error);
      return;
    }
    const auto initial = context_.pose_of(robot_.arms.at(arm_).tcp);
    if (!initial) {
      fail("TCP_TF_MISSING", "Cannot capture initial TCP for cup approach direction");
      return;
    }
    const auto wrist = context_.pose_of(arm_ == "left" ? "arm_l_link6" : "arm_r_link6");
    if (!wrist) {
      fail("WRIST_TF_MISSING", "Cannot capture joint6 wrist center for cup approach direction");
      return;
    }
    // Use joint6 origin for XY heading; retain TCP orientation for the near-cup fallback.
    initial_reference_ = initial->pose;
    initial_reference_.position = wrist->pose.position;
    phase("PICK_OBSERVE");
  }
  if (phase_ == "PICK_OBSERVE") {
    std::string reason;
    double position_error = 0;
    if (!observer_.poll(cup_id_, estimated_cup_pose_, position_error, reason)) {
      result_.detail = reason;
      if (elapsed(phase_started_) > config_.observation_timeout) {
        fail("CUP_NOT_FOUND", reason);
      }
      return;
    }
    body_ = estimated_cup_pose_.pose;
    const auto heading = cup_approach_yaw(initial_reference_, body_.position);
    if (!heading) {
      fail("PICK_GEOMETRY_INVALID", "Cannot determine initial wrist-to-cup heading");
      return;
    }
    const double yaw = *heading;
    auto rim = body_.position;
    rim.z += config_.cup_height / 2;
    pregrasp_ = cup_grasp_pose(rim, config_.cup_height, pick_, yaw);
    pregrasp_.position.x -= pick_.stand_off * std::cos(yaw);
    pregrasp_.position.y -= pick_.stand_off * std::sin(yaw);
    approach_target_.header = estimated_cup_pose_.header;
    approach_target_.pose = pregrasp_;
    if (!execute_) {
      result_.state = "SUCCEEDED";
      result_.sequence_completed = true;
      result_.detail = "Horizontal pregrasp preview; BODY assumed cup middle; IK/path not verified";
    } else {
      apply_scene();
    }
    return;
  }
  std::string health;
  if (!context_.ready(health)) {
    if (moving_) {
      executor_.cancel("ROBOT_STATE_INVALID", health);
    } else if (!scene_pending_) {
      fail("ROBOT_STATE_INVALID", health);
      return;
    }
  }
  executor_.tick();
  if (scene_pending_) {
    const auto r = scene_.poll();
    if (r.state == ExecutionState::Running) {
      return;
    }
    scene_pending_ = false;
    if (r.state != ExecutionState::Succeeded) {
      fail(r.code, r.detail);
      return;
    }
    if (result_.state == "CANCELLING") {
      result_.state = "CANCELLED";
      result_.code = "CANCELLED";
      return;
    }
    open_gripper();
    return;
  }
  if (moving_) {
    const auto r = executor_.result();
    if (r.state == ExecutionState::Running || r.state == ExecutionState::Cancelling) {
      return;
    }
    moving_ = false;
    if (r.state == ExecutionState::StopUnconfirmed) {
      fail("STOP_UNCONFIRMED", r.detail);
      return;
    }
    if (result_.state == "CANCELLING" || r.state == ExecutionState::Cancelled) {
      result_.state = "CANCELLED";
      result_.code = "CANCELLED";
      return;
    }
    if (r.state != ExecutionState::Succeeded) {
      fail(r.code, phase_ + ": " + r.detail);
      return;
    }
    result_.last_completed_step = phase_;
    if (phase_ == "PICK_OPEN") {
      move("PICK_PREGRASP", pregrasp_);
    } else if (phase_ == "PICK_PREGRASP") {
      finish_approach();
    }
  }
}
} // namespace demo_1f
