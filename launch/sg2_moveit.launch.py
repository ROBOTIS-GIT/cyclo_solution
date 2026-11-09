"""SG2 MoveIt only. Gazebo/controller bringup is supplied by the user."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def setup(context):
    demo = Path(get_package_share_directory("demo_1f"))
    model = LaunchConfiguration("description_file").perform(context)
    config = (
        MoveItConfigsBuilder("ffw", package_name="ffw_moveit_config")
        .robot_description(file_path=model)
        .robot_description_semantic(file_path=str(demo / "config/sg2_handover.srdf"))
        .joint_limits(file_path=str(demo / "config/sg2_joint_limits.yaml"))
        .trajectory_execution(file_path=str(demo / "config/sg2_moveit_controllers.yaml"))
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl"
        )
        .to_moveit_configs()
    )
    for pipeline in ["ompl", "pilz_industrial_motion_planner"]:
        adapters = config.planning_pipelines[pipeline]["request_adapters"]
        adapters.insert(
            adapters.index("default_planning_request_adapters/CheckStartStateBounds"),
            "demo_1f/SnapStartStateBounds",
        )
    # Pilz in the deployed Jazzy version requires a chain IK solver even for
    # joint goals. A branched two-arm group uses OMPL's joint-space planner.
    # Jazzy's bundled defaults define "RRTConnect", not our requested config ID.
    # Define the ID explicitly so MoveIt does not fall back to another config.
    config.planning_pipelines["ompl"].setdefault("planner_configs", {})[
        "RRTConnectkConfigDefault"
    ] = {
        "type": "geometric::RRTConnect",
        "range": 0.0,
        # ParallelPlan must collect/compare solutions, not stop at the first one.
        # With joint-model state space, path length measures cumulative joint motion.
        "optimization_objective": "PathLengthOptimizationObjective",
        # Jazzy parses these planner strings with lexical_cast<bool>: use 1/0.
        "hybridize": 1,
        "simplify_solutions": 1,
    }
    # Position-only cup approach uses OMPL for a single arm with free wrist orientation.
    for arm in ("arm_l", "arm_r", "head", "lift", "initial_pose"):
        config.planning_pipelines["ompl"][arm] = {
            "planner_configs": ["RRTConnectkConfigDefault"],
            "default_planner_config": "RRTConnectkConfigDefault",
            "longest_valid_segment_fraction": 0.002,
            "enforce_joint_model_state_space": True,
        }
    config.planning_pipelines["ompl"]["handover_arms"] = {
        "planner_configs": ["RRTConnectkConfigDefault"],
        "default_planner_config": "RRTConnectkConfigDefault",
        "enforce_joint_model_state_space": True,
        "longest_valid_segment_fraction": 0.002,
    }
    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                config.to_dict(),
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "publish_robot_description_semantic": True,
                    "constraint_samplers": "demo_1f/RollSamplerAllocator",
                    "trajectory_execution.allowed_execution_duration_scaling": 2.0,
                    "trajectory_execution.allowed_goal_duration_margin": 5.0,
                },
            ],
        )
    ]


def generate_launch_description():
    description = (
        Path(get_package_share_directory("demo_1f"))
        / "urdf/sg2_extended_grippers.urdf"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("description_file", default_value=str(description)),
            OpaqueFunction(function=setup),
        ]
    )
