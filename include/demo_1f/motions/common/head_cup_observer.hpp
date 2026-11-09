#pragma once
#include "demo_1f/motions/common/cup_config.hpp"
#include "demo_1f/runtime/state/robot_context.hpp"

namespace demo_1f {
// Reads the head track captured continuously by RobotContext, including temporary occlusion.
class HeadCupObserver {
public:
  HeadCupObserver(RobotContext & context, const ApproachConfig & config)
      : context_(context), config_(config) {}

  bool poll(const std::string & cup_id,
      geometry_msgs::msg::PoseStamped & estimate,
      double & position_error,
      std::string & reason) const;

private:
  bool valid_sample(CupState & state,
      const std::string & cup_id,
      std::string & reason) const;
  RobotContext & context_;
  const ApproachConfig & config_;
};
} // namespace demo_1f
