#pragma once
#include "demo_1f/core/executor_port.hpp"
#include "demo_1f/core/motion_result.hpp"
#include "demo_1f/motions/common/cup_scene.hpp"
#include "demo_1f/runtime/state/robot_context.hpp"

namespace demo_1f {
class CupMotion {
public:
  CupMotion(const Config & robot,
      const ApproachConfig & config,
      RobotContext & context,
      ExecutorPort & executor,
      ApproachScenePort & scene);
  virtual bool is_ready_to_start(const std::string & arm,
      const std::string & cup_id,
      std::string & reason,
      bool execute = true) const;
  virtual bool start(const std::string & arm,
      const std::string & cup_id,
      std::string & reason,
      bool execute = true);
  virtual ~CupMotion() = default;
  virtual void tick() = 0;
  virtual void cancel();
  bool running() const;

  virtual bool blocks_other_motion() const {
    return running() || result_.code == "SCENE_UNCONFIRMED" || result_.code == "STOP_UNCONFIRMED";
  }

  const MotionResult & result() const {
    return result_;
  }

  const geometry_msgs::msg::PoseStamped & approach_target() const {
    return approach_target_;
  }

  const geometry_msgs::msg::PoseStamped & estimated_cup_pose() const {
    return estimated_cup_pose_;
  }

  bool at_observation_pose() const {
    return at_observation_pose_;
  }

protected:
  virtual const char * initial_phase() const = 0;

  virtual bool supports_arm(const std::string &) const {
    return true;
  }

  virtual void reset() = 0;
  bool execute_step(const Step & step);
  void phase(const std::string & name);
  void fail(const std::string & code, const std::string & detail);
  const Config & robot_;
  const ApproachConfig & config_;
  RobotContext & context_;
  ExecutorPort & executor_;
  ApproachScenePort & scene_;
  MotionResult result_;
  std::string arm_, cup_id_, phase_;
  TimePoint phase_started_;
  bool execute_{false}, moving_{false}, scene_pending_{false};
  bool at_observation_pose_{false};
  geometry_msgs::msg::PoseStamped approach_target_, estimated_cup_pose_;
};
} // namespace demo_1f
