"""ffw_grasp poses -> MoveIt scene -> searched approach and lift; --run picks with them.

Scene: table and basket (floor + four sloped walls) where ffw_grasp measured them, and the
fingers' printed extensions attached to end_effector_r_link. The approach is searched until one
passes (stop only when none does; a big motion is refused only when it collides):
  grasp options (ffw_grasp: every usable candidate x free tilt) x both gripper yaws, by TCP turn
  from initial pose 3 (all within --turn-window of the least: least joint motion of those that
  pass; beyond it the first that passes); for each, the path with the least joint motion over every
  analytic IK solution (analytic_ik): joint path to the line start, straight to the grasp and
  straight up (lift); colliding solutions are left out until one passes. Backup options
  (ffw_grasp narrow openings and rolls) only when no normal one passes, in ffw_grasp's order;
  OMPL to the line start last. Every path is collision checked by move_group and shown in RViz
  (Planned Path).
  Without --run the arm is not moved.

--run: open -> approach -> close -> lift (replanned from where the arm stopped) -> hold.
--put-back then goes straight down, releases, and returns up the finger axis to the start pose.
--release-home instead opens the gripper up there and returns to HOME (initial pose 3);
--cycles N repeats plan -> pick -> release -> HOME (0 = until the basket is empty, a cycle
fails or Ctrl+C). Recovery (user 2026-11-04): a failed plan = ffw_grasp already tried the other
objects; a camera/TF failure or a still settling object is planned again; NO SOLUTION = that
object is skipped (ffw_grasp skip_last) and another one planned; empty fingers = start over
(skips cleared); no object seen twice = basket empty (exit 0). A summary line closes the run.
--step asks before every motion; Ctrl+C cancels the running motion (the arm stops there).
Needs bringup (robot_description), pick_demo (grasp planner) and move_group; movej must not run.
"""
import argparse
import math
import os
import sys
import time

from ffw_pick.analytic_ik import ArmIK, Line, line_points
from ffw_pick.config import (
    ArmConfig, COMMAND_LINE, from_file, PickConfig, require, ros_parameters)
from ffw_pick.gripper import joint_for, load_gripper
from ffw_pick.kinematics import Chain, FLIP, pose_parts, tip, urdf_links
import numpy as np

FINGER_OBJECT = 'finger_extension'  # id of the attached finger extension boxes
PILZ = 'pilz_industrial_motion_planner'
DESCRIPTION_TOPIC = '/robot_description'
# ffw_grasp task (+ calibration) keys the scene and the fingertip use
TASK_KEYS = ('basket_length', 'basket_width', 'basket_height', 'basket_wall', 'floor_height',
             'basket_inner_top', 'basket_inner_floor', 'table_margin', 'table_thick',
             'tip_below_end_effector', 'finger_tip', 'finger_thick', 'taper_height',
             'finger_max', 'finger_width', 'finger_overlap', 'finger_layers', 'hover_height')
LINE_STEP = 0.005  # m between IK points on a straight line
JOINT_STEP = 0.02  # rad between states of a joint-space path
LINE_JUMP = 0.15  # rad: a bigger joint change between line points = wrist flip
SAMPLE_DT = 0.05  # s between trajectory points
CORNER = 6  # path points on each side of a corner that are rounded
START_TRIES = 6  # line starts whose joint path collides, per grasp option, before OMPL
HOME_EXTRA_LIFTS = 2  # --release-home: times it goes home_lift higher when HOME has no path
BAN_TRIES = 30  # colliding solutions left out of one line before it is given up
BOUND_SLACK = 0.02  # rad: a grasp option is skipped once its least possible joint motion is
# this much above the best found (the search grid may undercut it by about that)


def pick_result_code(status, hooks=None):
    """The demo requires a completed plan/motion sequence, never an empty basket."""
    if hooks is not None:
        if status == 'ok':
            hooks.error_code = ''
            hooks.message = ''
            return 0
        if status == 'basket empty':
            hooks.error_code = 'OBJECT_NOT_FOUND'
            hooks.message = 'Basket empty in consecutive observations'
        elif status == 'stuck':
            hooks.error_code = 'PICK_PLAN_FAILED'
            hooks.message = hooks.message or 'No remaining graspable DYNAMIXEL'
        return 1
    return 0 if status in ('ok', 'basket empty') else 1


def basket_boxes(task):
    """[(id, dims, center, rotation)] in the basket frame (x long side, z up, origin = outer
    bottom center on the table). Walls are slabs sloping from the inner floor size (on the floor)
    to the inner rim size (at the rim), continued on the same slope up to basket_wall_height."""
    length, width, height = task['basket_length'], task['basket_width'], task['basket_height']
    wall, floor = task['basket_wall'], task['floor_height']
    wall_height = task.get('basket_wall_height', height)
    margin = task.get('basket_wall_margin', 0.0)  # walls grow inward by this much
    wall += margin
    top_l, top_w = (v - 2 * margin for v in task['basket_inner_top'])
    floor_l, floor_w = (v - 2 * margin for v in task['basket_inner_floor'])
    boxes = [('floor', (length, width, floor), np.array([0.0, 0.0, floor / 2]), np.eye(3))]
    up = np.array([0.0, 0.0, 1.0])
    for name, out, top_half, floor_half, span in (
            ('wall_x+', np.array([1.0, 0, 0]), top_l / 2, floor_l / 2, width),
            ('wall_x-', np.array([-1.0, 0, 0]), top_l / 2, floor_l / 2, width),
            ('wall_y+', np.array([0, 1.0, 0]), top_w / 2, floor_w / 2, length),
            ('wall_y-', np.array([0, -1.0, 0]), top_w / 2, floor_w / 2, length)):
        bottom = out * floor_half + up * floor
        rise = (wall_height - floor) / (height - floor)
        top = out * (floor_half + (top_half - floor_half) * rise) + up * wall_height
        z = (top - bottom) / np.linalg.norm(top - bottom)
        x = out - (out @ z) * z
        x /= np.linalg.norm(x)
        rotation = np.column_stack((x, np.cross(z, x), z))
        boxes.append((name, (wall, span, float(np.linalg.norm(top - bottom))),
                      (bottom + top) / 2 + x * wall / 2, rotation))
    margin, thick = task['table_margin'], task['table_thick']
    table = (length + 2 * margin, width + 2 * margin, thick)
    boxes.append(('table', table, np.array([0.0, 0.0, -thick / 2]), np.eye(3)))
    return boxes


def finger_boxes(task, opening):
    """[(dims, center)] of the two fingers' printed extensions in the end effector frame
    (y closes, z up the fingers), from the real tip up into the URDF fingers. Layers follow the
    taper ffw_grasp checks (finger_tip -> finger_thick over taper_height, finger_width ->
    finger_width_root over finger_length); each layer is as thick and wide as its top."""
    below = task['tip_below_end_effector']
    tip, thick, taper = task['finger_tip'], task['finger_thick'], task['taper_height']
    width, root = task['finger_width'], task.get('finger_width_root') or task['finger_width']
    length = task.get('finger_length') or 1.0
    top = below + task['finger_overlap']  # into the URDF fingers
    edges = [h for h in task['finger_layers'] if h < top] + [top]
    boxes = []
    for low, high in zip(edges[:-1], edges[1:]):
        t = min(tip + (thick - tip) * high / taper, task['finger_max'])
        w = width + (root - width) * min(high / length, 1.0)
        z = -below + (low + high) / 2
        boxes += [((w, t, high - low), np.array([0.0, side * (opening / 2 + t / 2), z]))
                  for side in (1, -1)]
    return boxes


def trajectory_values(traj, now):
    """[(t, {joint: value})] of a RobotTrajectory, other joints from now."""
    jt = traj.joint_trajectory
    out = []
    for p in jt.points:
        values = dict(now)
        values.update(zip(jt.joint_names, p.positions))
        out.append((p.time_from_start.sec + p.time_from_start.nanosec * 1e-9, values))
    return out


def summary_lines(name, samples, chain, joints, tip_below, floor_z):
    """Duration, joint travel, fastest joint, lowest fingertip and the slowest interior point."""
    times = np.array([t for t, _ in samples])
    q = np.array([[v[n] for n in joints] for _, v in samples])
    travel = np.abs(q[-1] - q[0])
    rates = np.abs(np.diff(q, axis=0)) / np.maximum(np.diff(times), 1e-9)[:, None]
    speed = np.linalg.norm(rates, axis=1)  # whole arm, for finding a stop
    fastest = int(np.argmax(rates.max(axis=0)))
    tips = np.array([tip(chain, v, tip_below)[2] for _, v in samples])
    lines = [f'{name}: {times[-1]:.1f} s, {len(samples)} points, joint travel (rad) '
             + ', '.join(f'{n[-1]}:{d:.2f}' for n, d in zip(joints, travel)),
             f'  fastest joint {joints[fastest][-1]} {rates[:, fastest].max():.2f} rad/s,'
             f' lowest fingertip {(tips.min() - floor_z) * 1000:.0f} mm above the basket floor']
    inner = speed[len(speed) // 10:len(speed) - len(speed) // 10]
    if len(inner):
        lines.append(f'  slowest in the middle {inner.min():.3f} rad/s'
                     f' ({"stops" if inner.min() < 0.01 else "no stop"})')
    return lines


def turn_between(a, b):
    """Rotation angle from rotation a to rotation b, rad."""
    return math.acos(max(-1.0, min(1.0, (np.trace(a.T @ b) - 1.0) / 2.0)))


def search_tiers(options, home_rot):
    """Normal then backup (turn, option index, flip) entries: normal by TCP turn from home_rot,
    backups (narrow openings, rolls) in the given order (ffw_grasp's)."""
    entries = [(turn_between(home_rot, o['rotation'] @ f), oi, flip)
               for oi, o in enumerate(options)
               for flip, f in ((False, np.eye(3)), (True, FLIP))]
    normal = sorted((e for e in entries if not options[e[1]].get('backup')),
                    key=lambda e: (round(e[0], 3), e[1], e[2]))
    return normal, [e for e in entries if options[e[1]].get('backup')]


PLAN_KINDS = ('empty', 'unsure', 'objects', 'skipped', 'view')  # of a failed ffw_grasp plan


def plan_kind(message):
    """What a failed ffw_grasp plan says to do (PLAN_KINDS); an older message = view."""
    word = message.split(':', 1)[0].strip()
    return word if word in PLAN_KINDS else 'view'


def summary_line(tally, ended):
    """One line for the end of a run: tally counts (picked, empty_hand, switch, replan)."""
    return (f'summary: {tally.get("cycle", 0)} cycles, {tally.get("picked", 0)} picked,'
            f' {tally.get("empty_hand", 0)} empty fingers, {tally.get("switch", 0)} other'
            f' objects after NO SOLUTION, {tally.get("replan", 0)} plans again; ended: {ended}')


def limit_margin(chain, joints, values):
    """Smallest distance of an arm joint to its URDF limit, rad."""
    return min(min(values[n] - lo, hi - values[n])
               for n, (lo, hi) in ((n, chain.limits(n)) for n in joints))


def joint_path(joints, start, goal, step=JOINT_STEP):
    """Straight joint-space states start -> goal (start excluded)."""
    count = max(1, int(math.ceil(max(abs(goal[n] - start[n]) for n in joints) / step)))
    return [{**start, **{n: start[n] + (goal[n] - start[n]) * i / count for n in joints}}
            for i in range(1, count + 1)]


def round_corners(joints, states, corners, half=CORNER):
    """Joint states with each corner index replaced by a quadratic Bezier over +-half points
    (tangent at both ends, so the motion need not stop there). Returns (states, changed idx)."""
    q = np.array([[s[n] for n in joints] for s in states])
    changed = []
    for c in corners:
        a, b = c - half, c + half
        if a < 0 or b >= len(q):
            continue
        p0, p1, p2 = q[a].copy(), q[c].copy(), q[b].copy()
        for i in range(a, b + 1):
            t = (i - a) / (b - a)
            q[i] = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            changed.append(i)
    out = [{**s, **dict(zip(joints, row))} for s, row in zip(states, q)]
    return out, changed


def time_path(chain, joints, states, tip_below, speed, accel, limits, dt=SAMPLE_DT):
    """[(t, values, velocities)] along states from rest to rest: each step at the joint and
    fingertip speed limits, then a quintic time scaling so no joint passes them (approx.).
    limits = (joint rad/s, joint rad/s^2, fingertip m/s), scaled by speed and accel."""
    q = np.array([[s[n] for n in joints] for s in states])
    tips = np.array([tip(chain, s, tip_below) for s in states])
    v, a, vc = limits[0] * speed, limits[1] * accel, limits[2] * speed
    seg = np.maximum(np.abs(np.diff(q, axis=0)).max(axis=1) / v,
                     np.linalg.norm(np.diff(tips, axis=0), axis=1) / vc)
    u = np.concatenate(([0.0], np.cumsum(seg)))
    total_u = max(u[-1], 1e-6)
    # quintic: peak rate 1.875 u/T, peak acceleration 5.77 u/T^2 (dq/du <= v)
    duration = max(1.875 * total_u, math.sqrt(5.77 * total_u * v / a), dt)
    times = np.append(np.arange(0.0, duration, dt), duration)
    tau = times / duration
    s = total_u * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
    pos = np.column_stack([np.interp(s, u, q[:, j]) for j in range(q.shape[1])])
    vel = np.gradient(pos, times, axis=0)
    vel[0] = vel[-1] = 0.0
    return [(float(t), {**states[0], **dict(zip(joints, p))}, list(map(float, w)))
            for t, p, w in zip(times, pos, vel)]


def main(argv=None, hooks=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', default='',
                        help='URDF file (default: the robot_description topic)')
    parser.add_argument('--arm', default='', help='arm file (default: installed arm_r.yaml)')
    parser.add_argument('--config', default='',
                        help='pick settings (default: installed pick.yaml); options below win')
    parser.add_argument('--task', default='', help='ffw_grasp task file (default: installed)')
    parser.add_argument('--calibration', default='',
                        help='ffw_grasp robot calibration file (default: installed)')
    parser.add_argument('--ns', help='ffw_grasp grasp_planner namespace')
    parser.add_argument('--no-plan', action='store_true',
                        help='use the last published poses instead of calling plan')
    parser.add_argument('--above', type=float,
                        help='joint path target this far behind the hover along the finger axis')
    parser.add_argument('--lift', type=float, help='straight up after closing, m')
    parser.add_argument('--speed', type=float, help='velocity scaling 0-1')
    parser.add_argument('--accel', type=float, help='acceleration scaling 0-1')
    parser.add_argument('--turn-window', type=float,
                        help='deg: grasp options turned up to this much more than the least'
                             ' turned one are all solved, least joint motion wins')
    parser.add_argument('--budget', type=float,
                        help='s of approach search before giving up (stops early on success)')
    parser.add_argument('--verbose', action='store_true', help='print every tried combination')
    parser.add_argument('--allow-finger-walls', action='store_true',
                        help='fingers may touch the basket walls (default: they may not)')
    parser.add_argument('--no-finger-boxes', action='store_true',
                        help='no attached finger extension (after the URDF fingers are updated)')
    parser.add_argument('--run', action='store_true', help='move the arm (plan only without it)')
    parser.add_argument('--step', action='store_true', help='with --run: ask before every motion')
    parser.add_argument('--put-back', action='store_true',
                        help='with --run: after holding, put the object back and return')
    parser.add_argument('--release-home', action='store_true',
                        help='with --run: after holding, release up there and return to HOME')
    parser.add_argument('--cycles', type=int, default=1,
                        help='with --release-home: picks in a row (0 = until one fails or Ctrl+C)')
    parser.add_argument('--hold', type=float,
                        help='s held up before --put-back or --release-home')
    parser.add_argument('--table', default='',
                        help='gripper opening table (default: installed gripper_table.yaml)')
    parser.add_argument('--gripper-time', type=float,
                        help='s to open or close (default: the gripper table file)')
    parser.add_argument('--check-state', action='store_true',
                        help="only compare move_group's robot state with /joint_states and exit")
    parser.add_argument('--home', action='store_true',
                        help='only take the arm to HOME (up first; asks before each move)')
    parser.add_argument('--home-lift', type=float,
                        help='m straight up before --home and --release-home go to HOME'
                             ' (0 = none)')
    parser.add_argument('--gripper', type=float, default=None,
                        help='only move the gripper to this joint value (rad, 0 = open) and exit')
    parser.add_argument('--close-joint', type=float,
                        help='gripper close command, rad (default: the gripper table file)')
    args = parser.parse_args(argv)
    if args.release_home and (args.put_back or not args.run):
        parser.error('--release-home needs --run and excludes --put-back')
    if args.cycles != 1 and (not args.release_home or args.no_plan):
        parser.error('--cycles needs --release-home and a new plan each cycle (no --no-plan)')

    import rclpy
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped
    from moveit_msgs.action import ExecuteTrajectory, MoveGroup, MoveGroupSequence
    from moveit_msgs.msg import (
        AttachedCollisionObject, CollisionObject, Constraints, DisplayTrajectory,
        JointConstraint, MotionPlanRequest, MotionSequenceItem, PlanningScene, RobotState,
        RobotTrajectory)
    from moveit_msgs.msg import AllowedCollisionEntry, PlanningSceneComponents
    from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import JointState
    from shape_msgs.msg import SolidPrimitive
    from std_msgs.msg import Float64, Float64MultiArray, String
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from ffw_grasp.grasp_planner import matrix_to_quat

    from ament_index_python.packages import get_package_share_directory

    def installed(package, name):
        return os.path.join(get_package_share_directory(package), 'config', name)
    args.arm = args.arm or installed('ffw_pick', 'arm_r.yaml')
    args.config = args.config or installed('ffw_pick', 'pick.yaml')
    args.table = args.table or installed('ffw_pick', 'gripper_table.yaml')
    args.task = args.task or installed('ffw_grasp', 'dynamixel_top_grasp.yaml')
    args.calibration = args.calibration or installed('ffw_grasp', 'robot_calibration.yaml')
    try:
        arm_cfg = from_file(ArmConfig, args.arm)
        pick_cfg = from_file(PickConfig, args.config)
        gripper = load_gripper(args.table)
        # this robot's calibration wins over a task file that still has it
        task = {**ros_parameters(args.task), **ros_parameters(args.calibration)}
        require(task, TASK_KEYS, f'{args.task} + {args.calibration}')
    except (OSError, ValueError, KeyError) as error:
        if hooks:
            hooks.message = str(error)
            hooks.error_code = 'CONFIGURATION_ERROR'
        print(f'settings: {error}')
        return 1
    for name in COMMAND_LINE:
        if getattr(args, name) is None:
            setattr(args, name, getattr(pick_cfg, name))
    if args.gripper_time is None:
        args.gripper_time = gripper.move_time
    if args.close_joint is None:
        args.close_joint = gripper.close_command
    tip_below = task['tip_below_end_effector']
    arm_joints, home_pose = list(arm_cfg.arm_joints), list(arm_cfg.home)
    finger_parts = [FINGER_OBJECT] + list(arm_cfg.finger_links)

    from rclpy.context import Context
    from rclpy.signals import SignalHandlerOptions
    ros_context = Context()
    rclpy.init(context=ros_context, signal_handler_options=SignalHandlerOptions.NO if hooks else None)
    node = rclpy.create_node('pick_moveit', context=ros_context)
    from rclpy.executors import SingleThreadedExecutor
    worker_executor = SingleThreadedExecutor(context=ros_context)
    worker_executor.add_node(node)
    got, poses = {}, {}
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    if args.urdf:
        with open(args.urdf) as f:
            urdf = f.read()
    else:  # what robot_state_publisher (bringup) holds now
        node.create_subscription(String, DESCRIPTION_TOPIC,
                                 lambda m: got.update(urdf=m.data), latched)
        end = time.time() + 5.0
        while 'urdf' not in got and time.time() < end:
            worker_executor.spin_once(timeout_sec=0.1)
        if 'urdf' not in got:
            if hooks:
                hooks.error_code = 'ROBOT_DESCRIPTION_UNAVAILABLE'
                hooks.message = f'No {DESCRIPTION_TOPIC}; bringup must publish the robot model'
            print(f'no {DESCRIPTION_TOPIC}: is bringup running? (or give --urdf)')
            worker_executor.shutdown()
            node.destroy_node()
            ros_context.try_shutdown()
            return 1
        urdf = got.pop('urdf')
    chain = Chain(urdf, arm_cfg.base_frame, arm_cfg.end_effector_link)
    arm = ArmIK(chain, arm_joints, margin=arm_cfg.joint_margin)
    # joint speed from the URDF, acceleration and fingertip speed from the arm file
    limits = (min(chain.velocity(n) for n in arm_joints), arm_cfg.max_acceleration,
              arm_cfg.max_fingertip_speed)
    robot_links = urdf_links(urdf)
    wrist_part = urdf_links(urdf, arm_cfg.wrist_link) | {FINGER_OBJECT}
    home_rot = chain.fk(dict(zip(arm_joints, home_pose)))[:3, :3]
    if args.verbose:
        print(f'settings: arm {args.arm}, pick {args.config}, gripper {args.table}, task'
              f' {args.task}, calibration {args.calibration}; joint limits {limits[0]:.2f} rad/s'
              f' {limits[1]:.2f} rad/s^2, fingertip {limits[2]:.2f} m/s')
    from rclpy.qos import qos_profile_sensor_data
    def receive_joints(message):
        got.setdefault('joints', {}).update(zip(message.name, message.position))
        for name in message.name:
            got.setdefault('joint_times', {})[name] = time.monotonic()
    node.create_subscription(JointState, '/joint_states', receive_joints, qos_profile_sensor_data)
    for name in ('grasp', 'hover', 'basket'):
        node.create_subscription(PoseStamped, f'{args.ns}/{name}_pose',
                                 (lambda key: lambda m: poses.update({key: m}))(name), latched)
    node.create_subscription(Float64, f'{args.ns}/gripper_opening',
                             lambda m: got.update(opening=m.data), latched)
    node.create_subscription(PoseArray, f'{args.ns}/grasp_candidates',
                             lambda m: got.update(candidates=m), latched)
    node.create_subscription(Float64MultiArray, f'{args.ns}/grasp_candidate_info',
                             lambda m: got.update(info=list(m.data)), latched)
    plan_client = node.create_client(Trigger, f'{args.ns}/plan')
    skip_client = node.create_client(Trigger, f'{args.ns}/skip_last')
    clear_client = node.create_client(Trigger, f'{args.ns}/clear_skips')
    scene_client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
    get_scene_client = node.create_client(GetPlanningScene, '/get_planning_scene')
    valid_client = node.create_client(GetStateValidity, '/check_state_validity')
    sequence = ActionClient(node, MoveGroupSequence, '/sequence_move_group')
    move_group = ActionClient(node, MoveGroup, '/move_action')
    display = node.create_publisher(DisplayTrajectory, '/display_planned_path', 1)
    execute = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
    follow = ActionClient(node, FollowJointTrajectory, arm_cfg.gripper_action)
    active = {}  # goal handle of the motion running now, cancelled on Ctrl+C
    commands = []  # other commands on the arm controller topic (movej, teleoperation)
    node.create_subscription(JointTrajectory, arm_cfg.controller_topic,
                             lambda m: commands.append(m), 10)
    scene_opening = {}  # finger boxes currently in the scene
    spent = {'checks': 0, 'check_s': 0.0, 'scene_s': 0.0}  # collision checks and scene updates

    def spin(seconds, until=lambda: False, check_cancel=True):
        end = time.monotonic() + seconds
        while rclpy.ok(context=ros_context) and time.monotonic() < end and not until():
            if hooks and check_cancel:
                hooks.check_cancel()
            worker_executor.spin_once(timeout_sec=0.05)

    def cancel_active():
        handle = active.get('handle')
        if handle is None:
            return not (hooks and not hooks.stop_confirmed)
        handle.cancel_goal_async()
        done = active.get('result') or handle.get_result_async()
        spin(5.0, done.done, check_cancel=False)
        if not done.done() or done.result() is None:
            return False
        active.clear()
        return True

    def call(client, request, seconds=10.0):
        future = client.call_async(request)
        try:
            spin(seconds, future.done)
        finally:
            if hooks and client is scene_client and not future.done():
                hooks.error_code = 'SCENE_UNCONFIRMED'
                hooks.message = 'Planning-scene update outcome is unknown'
        return future.result()

    def current():
        stale = [n for n in chain.moving() if time.monotonic() - got.get('joint_times', {}).get(n, 0) > 1.0]
        if stale:
            raise RuntimeError(f'Stale joint feedback: {stale}')
        return {n: got['joints'][n] for n in chain.moving()}

    def gripper_now():
        return got.get('joints', {}).get(arm_cfg.gripper_joint, float('nan'))

    def pose(position, rotation):
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, position)
        o = p.orientation
        o.x, o.y, o.z, o.w = matrix_to_quat(rotation)
        return p

    def box(dims):
        return SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(d) for d in dims])

    def apply_scene(basket_position, basket_rotation, opening):
        """Basket, table and the finger boxes for opening."""
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        for name, dims, center, rotation in basket_boxes(task):
            obj = CollisionObject(id=f'basket_{name}', operation=CollisionObject.ADD)
            obj.header.frame_id = arm_cfg.base_frame
            obj.pose = pose(basket_position + basket_rotation @ center,
                            basket_rotation @ rotation)
            obj.primitives = [box(dims)]
            obj.primitive_poses = [pose(np.zeros(3), np.eye(3))]
            scene.world.collision_objects.append(obj)
        fingers = AttachedCollisionObject(link_name=arm_cfg.end_effector_link,
                                          touch_links=list(arm_cfg.touch_links))
        fingers.object.id = FINGER_OBJECT
        fingers.object.header.frame_id = arm_cfg.end_effector_link
        if args.no_finger_boxes:  # the URDF fingers already have the real length
            fingers.object.operation = CollisionObject.REMOVE
        else:
            fingers.object.operation = CollisionObject.ADD
            for dims, center in finger_boxes(task, opening):
                fingers.object.primitives.append(box(dims))
                fingers.object.primitive_poses.append(pose(center, np.eye(3)))
        scene.robot_state.attached_collision_objects = [fingers]
        result = call(scene_client, ApplyPlanningScene.Request(scene=scene))
        if result is None or not result.success:
            return False
        return finger_walls(args.allow_finger_walls)

    def state_mismatch():
        """'' if move_group's robot state matches /joint_states on the lift and arm joints
        (collision checks use its state, our IK uses /joint_states), else what differs."""
        if not get_scene_client.wait_for_service(timeout_sec=5.0):
            return 'no /get_planning_scene: is move_group running?'
        spin(2.0, lambda: 'joints' in got)
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ROBOT_STATE
        result = call(get_scene_client, request)
        if result is None:
            return 'no answer from /get_planning_scene'
        js = result.scene.robot_state.joint_state
        theirs = dict(zip(js.name, js.position))
        ours = got.get('joints', {})
        rows = []
        for n in chain.moving():
            if n not in theirs or n not in ours:
                rows.append(f'{n}: move_group {theirs.get(n, "missing")}, joint_states'
                            f' {ours.get(n, "missing")}')
            elif abs(theirs[n] - ours[n]) > pick_cfg.state_tolerance:
                rows.append(f'{n}: move_group {theirs[n]:+.3f}, joint_states {ours[n]:+.3f}')
        return '; '.join(rows)

    def finger_walls(allowed):
        """Finger-wall contacts allowed or not in move_group's collision matrix, written both
        ways each time (move_group keeps what an earlier run set). Checked by default again
        (user 2026-11-02, after fingers hit the basket). The whole matrix is sent back (a diff
        replaces it)."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        result = call(get_scene_client, request)
        if result is None:
            return False
        acm = result.scene.allowed_collision_matrix
        walls = [f'basket_{n}' for n, *_ in basket_boxes(task) if n.startswith('wall')]
        for name in finger_parts + walls:
            if name not in acm.entry_names:
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_names.append(name)
                acm.entry_values.append(AllowedCollisionEntry(
                    enabled=[False] * len(acm.entry_names)))
        index = {n: i for i, n in enumerate(acm.entry_names)}
        for f in finger_parts:
            for w in walls:
                acm.entry_values[index[f]].enabled[index[w]] = allowed
                acm.entry_values[index[w]].enabled[index[f]] = allowed
        scene = PlanningScene(is_diff=True, allowed_collision_matrix=acm)
        scene.robot_state.is_diff = True
        result = call(scene_client, ApplyPlanningScene.Request(scene=scene))
        return result is not None and result.success

    def fingers_for(opening, force=False):
        """Finger boxes in the scene match this opening (the scene is re-applied if not)."""
        if force or scene_opening.get('value') != round(opening, 4):
            started = time.time()
            if not apply_scene(scene_opening['basket'][0], scene_opening['basket'][1], opening):
                raise RuntimeError('planning scene not applied')
            scene_opening['value'] = round(opening, 4)
            spent['scene_s'] += time.time() - started

    def joint_goal(values):
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=n, position=float(values[n]),
                                               tolerance_above=1e-3, tolerance_below=1e-3,
                                               weight=1.0) for n in arm_joints]
        return c

    def state_of(values):
        s = RobotState(is_diff=True)
        s.joint_state.name = list(arm_joints)
        s.joint_state.position = [float(values[n]) for n in arm_joints]
        return s

    def request(planner, goal, start=None, pipeline=PILZ):
        r = MotionPlanRequest(group_name=arm_cfg.group, pipeline_id=pipeline, planner_id=planner,
                              num_planning_attempts=1, allowed_planning_time=3.0,
                              max_velocity_scaling_factor=args.speed,
                              max_acceleration_scaling_factor=args.accel)
        r.goal_constraints = [goal]
        if start is not None:
            r.start_state = start
        return r

    def action_result(client, goal, seconds, label):
        if hooks:
            hooks.check_cancel()
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'{label}: no action server (is move_group running?)')
        moving = client is execute or client is follow
        if hooks and moving:
            hooks.motion_started = True
            hooks.stop_confirmed = False
            hooks.phase(label.upper().replace(' ', '_'), label)
        send = client.send_goal_async(goal)
        # A pending acceptance must be resolved before cancellation can target its handle.
        spin(5.0, send.done, check_cancel=False)
        if not send.done():
            raise RuntimeError(f'{label}: goal acceptance unknown')
        handle = send.result()
        if handle is None or not handle.accepted:
            if hooks and moving:
                hooks.stop_confirmed = True
            raise RuntimeError(f'{label}: goal rejected')
        active['handle'] = handle
        done = handle.get_result_async()
        active['result'] = done
        spin(seconds, done.done)
        if not done.done() or done.result() is None:
            confirmed = cancel_active()
            if hooks:
                hooks.stop_confirmed = confirmed
            raise RuntimeError(f'{label}: timed out; stop confirmed={confirmed}')
        active.clear()
        if hooks and moving:
            hooks.stop_confirmed = True
        return done.result().result

    def plan_sequence(items, start=None):
        """Pilz [(request, blend radius)] -> (RobotTrajectory, None) or (None, reason)."""
        goal = MoveGroupSequence.Goal()
        goal.request.items = [MotionSequenceItem(req=r, blend_radius=float(b)) for r, b in items]
        goal.request.items[0].req.start_state = start or RobotState(is_diff=True)
        goal.planning_options.plan_only = True
        result = action_result(sequence, goal, 30.0, 'Pilz')
        response = result.response
        if response.error_code.val != 1 or not response.planned_trajectories:
            return None, f'Pilz error {response.error_code.val} (see move_group log)'
        return response.planned_trajectories[0], None

    def plan_ompl(goal_values, start=None, seconds=3.0):
        goal = MoveGroup.Goal()
        goal.request = request('', joint_goal(goal_values), start, pipeline='ompl')
        goal.request.allowed_planning_time = float(seconds)
        goal.request.start_state = start or RobotState(is_diff=True)
        goal.planning_options.plan_only = True
        result = action_result(move_group, goal, 30.0, 'OMPL')
        if result.error_code.val != 1:
            return None, f'OMPL error {result.error_code.val}'
        return result.planned_trajectory, None

    def first_bad(states, coarse=4):
        """(index, 'pct%: link - object', contact pairs) of the first state move_group finds in
        collision, or (None, '', []). Every coarse-th state first (a collision usually shows
        there), then the rest."""
        order = list(range(0, len(states), coarse)) + [i for i in range(len(states)) if i % coarse]
        for i in order:
            started = time.time()
            result = call(valid_client, GetStateValidity.Request(robot_state=state_of(states[i]),
                                                                 group_name=arm_cfg.group))
            spent['checks'] += 1
            spent['check_s'] += time.time() - started
            if result is None:
                return i, 'no answer from /check_state_validity', []
            if not result.valid:
                pairs = sorted({(c.contact_body_1, c.contact_body_2) for c in result.contacts})
                names = ', '.join(f'{a} - {b}' for a, b in pairs) or 'invalid'
                return i, f'{i * 100 // max(len(states) - 1, 1)}%: {names}', pairs
        return None, '', []

    def bad_states(states, coarse=4):
        return first_bad(states, coarse)[1]

    def pose_only(pairs):
        """True when a contact is between the wrist part (link 7 and below, finger boxes) and
        the world: it is there for every IK solution of that fingertip pose."""
        return any(
            (a in wrist_part or a not in robot_links) and (b in wrist_part or b not in robot_links)
            and (a in wrist_part or b in wrist_part) for a, b in pairs)

    def joints_of(values):
        return np.array([values[n] for n in arm_joints])

    def with_arm(base, q):
        return {**base, **dict(zip(arm_joints, map(float, q)))}

    def traj_states(traj, base):
        return [v for _, v in trajectory_values(traj, base)]

    def own_trajectory(states):
        return sampled_trajectory(time_path(chain, arm_joints, states, tip_below, args.speed,
                                            args.accel, limits))

    def sampled_trajectory(samples):
        out = RobotTrajectory()
        out.joint_trajectory.joint_names = list(arm_joints)
        for t, values, vel in samples:
            p = JointTrajectoryPoint(positions=[float(values[n]) for n in arm_joints],
                                     velocities=vel)
            p.time_from_start.sec = int(t)
            p.time_from_start.nanosec = int((t % 1) * 1e9)
            out.joint_trajectory.points.append(p)
        return out

    def own_path(parts, start):
        """One rest-to-rest trajectory through [states...] parts, corners between parts rounded;
        the rounded states are collision checked here (the parts were checked by the caller)."""
        states, corners = [start], []
        for part in parts:
            states += part
            corners.append(len(states) - len(part) - 1)
        states, changed = round_corners(arm_joints, states, corners[1:])
        bad = bad_states([states[i] for i in sorted(set(changed))], coarse=1)
        if bad:
            return None, f'rounded corner collides {bad}'
        return own_trajectory(states), None

    def free_path(line, base, fixed, deadline, bound=math.inf):
        """Least-motion path of line (analytic_ik.Line) from base, colliding solutions left out
        until one passes: (states, '') or (None, why). Given up (not checked) once the line's
        searched cost cannot beat bound."""
        start = joints_of(base)
        for _ in range(BAN_TRIES):
            path, why = line.path(start, fixed, LINE_JUMP, bound + BOUND_SLACK)
            if why == 'over bound':
                return None, f'skipped: joint motion at least {line.cost:.2f} rad'
            if path is None:
                return None, why
            states = [with_arm(base, q) for q in path]
            i, text, pairs = first_bad(states)
            if i is None:
                return states, ''
            if pose_only(pairs):  # another arm solution puts the fingers in the same place
                return None, f'collides {text} (the gripper itself: every arm solution)'
            if time.time() > deadline:
                return None, f'collides {text} (search budget used up)'
            line.ban(i, lambda q: bool(bad_states([with_arm(base, q)], coarse=1)))
        return None, f'collides {text} ({BAN_TRIES} colliding solutions left out)'

    def plan_line(start_values, end_position, rotation, label):
        """Straight fingertip line from start_values, least joint motion (analytic IK). Checked."""
        points = line_points(chain.fk(start_values)[:3, 3], end_position, LINE_STEP)[1:]
        states, why = free_path(Line(arm, start_values, points, rotation), start_values, True,
                                time.time() + args.budget)
        if states is None:
            raise RuntimeError(f'{label}: {why}')
        return own_trajectory([start_values] + states), 'analytic IK line'

    def plan_home(here, outbound=None):
        """Segments from here to HOME: Pilz PTP, own joint path, back the way the approach came
        (outbound = its joint states HOME -> above, ending at above), else OMPL; all checked.
        RuntimeError names why each one failed."""
        home = {**here, **dict(zip(arm_joints, home_pose))}
        reasons = []
        traj, why = plan_sequence([(request('PTP', joint_goal(home)), 0.0)])
        bad = bad_states(traj_states(traj, here)) if traj is not None else ''
        if traj is not None and not bad:
            return [traj]
        reasons.append(f'PTP: {why or "collides " + bad}')
        joints = joint_path(arm_joints, here, home)
        bad = bad_states(joints)
        if not bad:
            return [own_trajectory([here] + joints)]
        reasons.append(f'joint path: collides {bad}')
        if outbound:
            # to the approach's line start, then its joint path backwards: passed once already
            back = [{**here, **{n: v[n] for n in arm_joints}}
                    for v in list(reversed(outbound[:-1]))] + [home]
            to_above = joint_path(arm_joints, here, {**here, **{n: outbound[-1][n]
                                                               for n in arm_joints}})
            bad = bad_states(to_above + back)
            traj, why = (None, f'collides {bad}') if bad else own_path([to_above, back], here)
            if traj is not None:
                return [traj]
            reasons.append(f'back the way it came: {why}')
        for _ in range(2):
            traj, why = plan_ompl(home, state_of(here), seconds=10.0)
            bad = bad_states(traj_states(traj, here)) if traj is not None else ''
            if traj is not None and not bad:
                return [traj]
            reasons.append(f'OMPL: {why or "collides " + bad}')
        raise RuntimeError('no path back to HOME (' + '; '.join(reasons) + ')')

    def ask(text, always=False):
        if hooks:
            hooks.check_cancel()
            return
        if (args.step or always) and input(
                f'{text} [Enter = go, q = stop] ').strip().lower() == 'q':
            raise KeyboardInterrupt

    def go_home():
        """--home: straight up, then HOME (both checked in the current planning scene)."""
        spin(2.0, lambda: 'joints' in got)
        here = current()
        off = max(abs(here[n] - h) for n, h in zip(arm_joints, home_pose))
        if off <= arm_cfg.home_tolerance:
            print(f'already at HOME ({off:.3f} rad)')
            return 0
        if not valid_client.wait_for_service(timeout_sec=5.0):
            print('no /check_state_validity: is move_group running?')
            return 1
        commands.clear()
        spin(1.0)
        if commands:
            print(f'{len(commands)} commands/s arrive on {arm_cfg.controller_topic}: stop it first')
            return 1
        print(f'the arm is {off:.2f} rad from HOME; the scene is whatever move_group holds now')
        if args.home_lift > 0:
            pose = chain.fk(here)
            up, how = plan_line(here, pose[:3, 3] + np.array([0.0, 0.0, args.home_lift]),
                                pose[:3, :3], 'up')
            display.publish(DisplayTrajectory(trajectory=[up], trajectory_start=state_of(here)))
            ask(f'straight up {args.home_lift * 100:.0f} cm ({how})?', always=True)
            run_trajectory(up, 'up')
        segments = plan_home(current())
        display.publish(DisplayTrajectory(trajectory=segments,
                                          trajectory_start=state_of(current())))
        for i, seg in enumerate(segments, 1):
            ask(f'to HOME {i}/{len(segments)}?', always=True)
            run_trajectory(seg, f'to HOME {i}/{len(segments)}')
        print('at HOME')
        return 0

    def run_trajectory(traj, label):
        end = traj.joint_trajectory.points[-1].time_from_start
        seconds = end.sec + end.nanosec * 1e-9
        if hooks:
            hooks.phase(label.upper().replace(' ', '_'), f'{seconds:.1f} s')
        print(f'{label} ({seconds:.1f} s)')
        result = action_result(execute, ExecuteTrajectory.Goal(trajectory=traj), seconds + 10.0,
                               label)
        if result.error_code.val != 1:
            raise RuntimeError(f'{label}: MoveIt execution error code {result.error_code.val}')

    def run_gripper(value, label):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [arm_cfg.gripper_joint]
        point = JointTrajectoryPoint(positions=[float(value)])
        point.time_from_start.sec = int(args.gripper_time)
        point.time_from_start.nanosec = int((args.gripper_time % 1) * 1e9)
        goal.trajectory.points = [point]
        print(f'{label}: gripper -> {value:.3f} rad ({args.gripper_time:.1f} s)')
        result = action_result(follow, goal, args.gripper_time + 5.0, label)
        if result.error_code != 0:  # a held object can stop the fingers early: report only
            print(f'  note: gripper result {result.error_code} {result.error_string}')
        spin(0.3)

    def read_options():
        """Grasp options from grasp_candidates (+ info rows), else the single grasp_pose."""
        cands, info = got.get('candidates'), got.get('info', [])
        out = []
        if cands is not None and cands.header.stamp == poses['grasp'].header.stamp \
                and len(info) in (n * len(cands.poses) for n in (4, 5, 6)):
            width = len(info) // max(len(cands.poses), 1)  # 4 = before rolls, 5 = before backups
            if width < 6:
                print('note: grasp_candidate_info is an older format: restart pick_demo')
            for i, p in enumerate(cands.poses):
                position, rotation = pose_parts(PoseStamped(pose=p))
                row = list(info[width * i:width * i + width]) + [0.0] * (6 - width)
                opening, tilt, name, turn, roll, backup = row
                backup = bool(backup) or bool(roll)  # older rows: rolls were the backups
                out.append({'position': position, 'rotation': rotation, 'opening': opening,
                            'label': f'{"AB"[int(name)]} tilt {math.degrees(tilt):.0f}'
                            + (f' roll {math.degrees(roll):+.0f}' if roll else '')
                            + (f' open {opening * 1000:.0f}' if backup and not roll else ''),
                            'turn': turn, 'roll': roll, 'backup': backup})
        if not out:
            position, rotation = pose_parts(poses['grasp'])
            tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1, 1)))
            out.append({'position': position, 'rotation': rotation, 'opening': got['opening'],
                        'label': f'grasp_pose tilt {tilt:.0f}', 'turn': 0.0})
        for o in out:
            o['hover'] = o['position'] + task['hover_height'] * o['rotation'][:, 2]
        return out

    def gripper_blocked(line, base):
        """(point index, contacts) of the first searched point where the gripper itself hits
        the scene (one IK solution tells: every other one puts it in the same place), or None."""
        for n, (q, _) in enumerate(line.layers):
            if not len(q):
                break
            i, text, pairs = first_bad([with_arm(base, q[0])], coarse=1)
            if i is not None and pose_only(pairs):
                return line.keys[n], text.split(': ', 1)[-1]
        return None

    def try_approach(option, rotation, now, lift, deadline, bound=math.inf):
        """(found, why, ompl) for one grasp option at one gripper yaw. found = (segments, how,
        grasp values, lift trajectory or None, joint motion): joint path to the line start,
        then straight to the grasp (and lift m straight up), least joint motion first; line
        starts whose joint path collides are left out (START_TRIES of them). ompl = (target,
        states, grasp index) of the first such line (OMPL may go around). Skipped when it
        cannot beat bound (rad of joint motion). why and how end with where the time went."""
        started, before, lines = time.time(), dict(spent), []

        def timing():
            ik = sum(ln.times['ik'] for ln in lines)
            paths = sum(ln.times['search'] + ln.times['fill'] + ln.times['smooth']
                        for ln in lines)
            return (f' [{time.time() - started:.1f} s: IK {ik:.1f}, paths {paths:.1f},'
                    f' {spent["checks"] - before["checks"]} checks'
                    f' {spent["check_s"] - before["check_s"]:.1f}, scene'
                    f' {spent["scene_s"] - before["scene_s"]:.1f}]')
        found, why, ompl = attempt(option, rotation, now, lift, deadline, bound, lines)
        if found:
            return (found[0], found[1] + timing()) + found[2:], '', ompl
        return None, why + timing(), ompl

    def attempt(option, rotation, now, lift, deadline, bound, lines):
        """try_approach without the timing."""
        fingers_for(option['opening'])
        grasp, z = option['position'], rotation[:, 2]
        why, ompl = '', None
        start = joints_of(now)

        def start_collides(q):
            return first_bad(joint_path(arm_joints, now, with_arm(now, q)))[0] is not None
        for target, begin in (('above', option['hover'] + args.above * z),
                              ('hover', option['hover'])):
            approach = line_points(begin, grasp, LINE_STEP)
            up = line_points(grasp, grasp + np.array([0.0, 0.0, lift]), LINE_STEP)[1:] \
                if lift else []
            g = len(approach) - 1
            line = Line(arm, now, approach + up, rotation)
            lines.append(line)
            if not line.why:  # the first point has solutions
                least = line.start_bound(start)
                if least - BOUND_SLACK >= bound:
                    return None, f'skipped: move to the {target} at least {least:.2f} rad', ompl
                line.solve()
            if line.why:
                why = line.why
                continue  # the hover is only for a line with no IK solution from the above
            blocked = gripper_blocked(line, now)
            if blocked is not None:
                point, contacts = blocked
                why = (f'the gripper hits {contacts} at'
                       f' {point * 100 // max(len(line.points) - 1, 1)}% of the line from the'
                       f' {target} (every arm solution)')
                if point < len(line_points(begin, option['hover'], LINE_STEP)) - 1:
                    continue  # only above the hover: from the hover may pass
                return None, why, ompl
            for _ in range(START_TRIES):
                states, why = free_path(line, now, False, deadline, bound)
                if states is None:
                    return None, why, ompl
                joints = joint_path(arm_joints, now, states[0])
                traj, text = None, first_bad(joints)[1]
                if not text:
                    traj, text = own_path([joints, states[1:g + 1]], now)
                if traj is not None:
                    move = max(abs(states[0][n] - now[n]) for n in arm_joints)
                    path = [start] + [joints_of(v) for v in states]
                    motion = sum(float(np.abs(b - a).max()) for a, b in zip(path, path[1:]))
                    how = (f'joint path to {target} (move {move:.2f} rad, limit margin'
                           f' {limit_margin(chain, arm_joints, states[0]):.3f}) + analytic IK line,'
                           f' joint motion {motion:.2f} rad')
                    lifted = own_trajectory(states[g:]) if len(states) > g + 1 else None
                    return ([traj], how, states[g], lifted, motion, joints), '', ompl
                ompl = ompl or (target, states, g)
                why = f'joint path to {target} collides {text}'
                if time.time() > deadline:
                    return None, f'{why} (search budget used up)', ompl
                line.ban(0, start_collides)
            return None, f'{why} ({START_TRIES} line starts tried)', ompl
        return None, why, ompl

    def approach_ompl(option, rotation, now, ompl):
        """try_approach's found for OMPL to the line start of ompl, or (None, why)."""
        target, states, g = ompl
        fingers_for(option['opening'])
        traj, why = plan_ompl(states[0])
        if traj is None:
            return None, why
        bad = bad_states(traj_states(traj, now))
        if bad:
            return None, f'OMPL path collides {bad}'
        line, why = own_path([states[1:g + 1]], states[0])
        if line is None:
            return None, why
        lifted = own_trajectory(states[g:]) if len(states) > g + 1 else None
        return ([traj, line], f'OMPL to {target} (stops there) + analytic IK line', states[g],
                lifted, math.inf, traj_states(traj, now)), ''

    def search_approach(options, now, lift=0.0):
        """(option, rotation, segments, how, grasp values, lift trajectory) or None. Options x
        gripper yaws in order of TCP turn from the initial pose; every one within --turn-window
        of the least turn is solved and the least joint motion of those that pass wins (user
        2026-11-02); past the window the first that passes. Within one, the least joint motion
        (try_approach). Backups (narrow openings, rolls) only when no normal one passes: first
        that passes, in ffw_grasp's order (2026-11-03). OMPL only after every one failed so."""
        deadline = time.time() + args.budget
        normal, backups = search_tiers(options, home_rot)
        retry, best = [], None
        for tier, entries in (('normal', normal), ('backup', backups)):
            if best is not None or time.time() > deadline:
                break
            if tier == 'backup' and entries:
                log.append(f'backups: {len(entries) // 2} options')
            window = entries[0][0] + math.radians(args.turn_window) if entries else 0.0
            for turn, oi, flip in entries:
                if hooks:
                    hooks.check_cancel()
                if best is not None and (tier == 'backup' or turn > window):
                    break
                if time.time() > deadline:
                    log.append(f'search budget {args.budget:.0f} s used up')
                    break
                o = options[oi]
                rotation = o['rotation'] @ FLIP if flip else o['rotation']
                name = (f'{o["label"]}{" flip" if flip else ""}'
                        f' (TCP turn {math.degrees(turn):.0f} deg)')
                found, why, ompl = try_approach(o, rotation, now, lift, deadline,
                                                best[2][4] if best else math.inf)
                log.append(f'{name}: ' + (f'OK - {found[1]}' if found else why))
                if found and (best is None or found[4] < best[2][4]):
                    best = (o, rotation, found)
                if ompl is not None:
                    retry.append((name, o, rotation, ompl))
        if best is not None:
            return (best[0], best[1]) + best[2][:4] + best[2][5:6]
        if time.time() > deadline:
            return None
        for name, o, rotation, ompl in retry:
            if time.time() > deadline:
                log.append(f'search budget {args.budget:.0f} s used up')
                return None
            found, why = approach_ompl(o, rotation, now, ompl)
            log.append(f'{name} OMPL: ' + (f'OK - {found[1]}' if found else why))
            if found:
                return (o, rotation) + found[:4] + found[5:6]
        return None

    log = []
    tally = {}  # counts for the summary line

    def count(name):
        tally[name] = tally.get(name, 0) + 1

    def plan_until_usable():
        """Plan calls until one has a grasp and its basket: '' or why it stopped (basket
        empty, stuck = no object graspable, stop = no usable view)."""
        empty = 0
        for attempt in range(1, pick_cfg.plan_tries + 1):
            if attempt > 1:
                count('replan')
                print(f'  plan again in {pick_cfg.retry_wait:.0f} s'
                      f' ({attempt - 1}/{pick_cfg.plan_tries - 1})')
                spin(pick_cfg.retry_wait)
            poses.clear()
            for key in ('opening', 'candidates', 'info'):
                got.pop(key, None)
            asked = time.time()
            result = call(plan_client, Trigger.Request(), 20.0)
            message = result.message if result else 'view: no answer'
            print(f'plan: {message} ({time.time() - asked:.1f} s)')
            if hooks:
                hooks.message = message
                hooks.error_code = 'OBJECT_NOT_FOUND'
            if result is None or not result.success:
                kind = plan_kind(message)
                empty = empty + 1 if kind == 'empty' else 0
                if empty >= 2:  # twice in a row: a dropped object has settled by then
                    return 'basket empty'
                if kind == 'skipped':  # every object left out after NO SOLUTION: same again
                    return 'stuck'
                if attempt == pick_cfg.plan_tries:
                    return 'stuck' if kind in ('objects', 'unsure') else 'stop'
                continue
            spin(3.0, lambda: all(k in poses for k in ('grasp', 'hover', 'basket'))
                 and 'opening' in got and 'joints' in got and 'candidates' in got
                 and 'info' in got)
            if 'grasp' not in poses or 'hover' not in poses or 'opening' not in got:
                print('  no grasp poses arrived after the plan')
            elif ('basket' not in poses
                  or poses['basket'].header.stamp != poses['grasp'].header.stamp):
                print('  the basket was not measured in this plan (see the plan message): not'
                      ' planning without the walls')
            else:
                if hooks:
                    hooks.message = ''
                    hooks.error_code = ''
                return ''
            if attempt == pick_cfg.plan_tries:
                return 'stop'
        return 'stop'

    def pick_once():
        """One plan -> search -> (with --run) pick. 'ok' = planned (and with --run held),
        'empty hand', 'no path' (NO SOLUTION), 'basket empty', 'stuck' or 'stop'."""
        log.clear()
        if hooks:
            hooks.phase('PICK_OBSERVE', 'Collecting fresh DYNAMIXEL observations')
        if args.release_home:
            spin(2.0, lambda: 'joints' in got)
            off = max(abs(got.get('joints', {}).get(n, math.inf) - h)
                      for n, h in zip(arm_joints, home_pose))
            if off > arm_cfg.home_tolerance:
                print(f'the arm is {off:.2f} rad from HOME (initial pose 3): move it there first')
                return 'stop'
        if not args.no_plan:
            if not plan_client.wait_for_service(timeout_sec=5.0):
                print(f'no {args.ns}/plan: is pick_demo running?')
                return 'stop'
            stopped = plan_until_usable()
            if stopped:
                return stopped
        else:
            spin(3.0, lambda: all(k in poses for k in ('grasp', 'hover', 'basket'))
                 and 'opening' in got and 'joints' in got and 'candidates' in got
                 and 'info' in got)
            if 'grasp' not in poses or 'hover' not in poses or 'opening' not in got:
                print('no grasp poses: call the plan service')
                return 'stop'
            if ('basket' not in poses
                    or poses['basket'].header.stamp != poses['grasp'].header.stamp):
                print('the basket was not measured in this plan (see the plan message): not'
                      ' planning without the walls')
                return 'stop'
        missing = [n for n in chain.moving() if n not in got.get('joints', {})]
        if missing:
            print(f'joints missing in /joint_states: {missing}')
            return 'stop'
        now = current()
        basket_position, basket_rotation = pose_parts(poses['basket'])
        floor_z = basket_position[2] + task['floor_height']
        if hooks:
            hooks.phase('PICK_PLAN', 'Selecting a collision-free analytic IK path')
        options = read_options()

        for client, name in ((scene_client, '/apply_planning_scene'),
                             (valid_client, '/check_state_validity'),
                             (get_scene_client, '/get_planning_scene')):
            if not client.wait_for_service(timeout_sec=5.0):
                print(f'no {name}: is move_group running?')
                return 'stop'
        scene_opening['basket'] = (basket_position, basket_rotation)
        fingers_for(options[0]['opening'])
        yaw = math.degrees(math.atan2(basket_rotation[1, 0], basket_rotation[0, 0]))
        print(f'scene: basket at ({basket_position[0]:.3f}, {basket_position[1]:.3f},'
              f' {basket_position[2]:.3f}) yaw {yaw:.0f} deg, '
              + ('URDF fingers' if args.no_finger_boxes else 'finger extension boxes')
              + f'; {len(options)} grasp options: ' + ', '.join(o['label'] for o in options))
        start_bad = bad_states([now], coarse=1)
        if start_bad:
            if hooks:
                hooks.error_code = 'START_STATE_COLLISION'
                hooks.message = start_bad
            print(f'the start pose is in collision with the scene: {start_bad}')
            return 'stop'

        started = time.time()
        found = search_approach(options, now, args.lift)
        print(f'search: {len(log)} tried in {time.time() - started:.1f} s')
        if args.verbose or found is None:
            print('\n'.join(f'  {line}' for line in log))
        if found is None:
            if hooks:
                hooks.error_code = 'PICK_PLAN_FAILED'
                hooks.message = log[-1] if log else 'No collision-free grasp path'
            print('NO SOLUTION: every option above failed; the arm is not moved')
            return 'no path'
        option, rotation, approach, how, grasp_values, lift, outbound = found
        print(f'chosen: {option["label"]} - {how}')
        lift_end = option['position'] + np.array([0.0, 0.0, args.lift])
        lift_how = 'analytic IK line'

        start = state_of(now)
        display.publish(DisplayTrajectory(trajectory=approach + [lift], trajectory_start=start))
        spin(0.5)
        samples, t0 = [], 0.0
        for seg in approach:
            part = trajectory_values(seg, now)
            samples += [(t0 + t, v) for t, v in part]
            t0 += part[-1][0]
        print('\n'.join(summary_lines('approach', samples, chain, arm_joints, tip_below, floor_z)))
        print('\n'.join(summary_lines(f'lift ({lift_how})', trajectory_values(lift, grasp_values),
                                      chain, arm_joints, tip_below, floor_z)))
        if not args.run:
            print('plan ok: shown in RViz (Planned Path); the arm was not moved (--run to move)')
            return 'ok'

        # bringup's cyclo_teleoperation publishes there only while teleoperating: count messages
        commands.clear()
        spin(1.0)
        if commands:
            names = ', '.join(i.node_name for i in
                              node.get_publishers_info_by_topic(arm_cfg.controller_topic))
            print(f'{len(commands)} commands/s arrive on {arm_cfg.controller_topic} (from {names};'
                  ' movej or teleoperation?): stop it first')
            return 'stop'
        table = gripper.table
        opening = option['opening']
        open_joint = joint_for(table, opening)
        close_joint = args.close_joint  # the object stops the fingers; empty closes past it
        verification = (f'holding = stopped below {gripper.hold_below:.2f}' if hooks is None
                        else 'physical grasp not verified')
        print(f'gripper open {opening * 1000:.0f} mm ({open_joint:.3f} rad), close command'
              f' {close_joint:.3f} rad ({verification}), speed scaling {args.speed:.2f}')

        ask('open the gripper?')
        run_gripper(open_joint, 'open')
        for i, seg in enumerate(approach, 1):
            ask(f'approach {i}/{len(approach)}?')
            run_trajectory(seg, f'approach {i}/{len(approach)}')
        ask('close the gripper?')
        run_gripper(close_joint, 'close')

        def holding(label):
            # a box between the fingers stops them short of the command; empty = closed fully
            held = got['joints'].get(arm_cfg.gripper_joint, float('nan'))
            fresh = time.monotonic() - got.get('joint_times', {}).get(arm_cfg.gripper_joint, 0) < 1.0
            yes = (fresh and math.isfinite(held) and open_joint + 0.02 < held
                   < min(gripper.hold_below, close_joint - 0.02))
            print(f'  {label}: gripper at {held:.3f} rad'
                  f' (holding below {gripper.hold_below:.2f}) -> '
                  + ('holding' if yes else 'EMPTY'))
            return yes
        held = holding('closed') if hooks is None else False
        # from where the arm really stopped: MoveIt refuses a start 0.01 rad off the plan
        lift, lift_how = plan_line(current(), lift_end, rotation, 'lift')
        ask(f'lift {args.lift * 100:.0f} cm ({lift_how})?')
        run_trajectory(lift, 'lift')
        if hooks is not None:
            # Demo completion is controller completion, not a joint-position grasp estimate.
            # Physical grasp verification will be provided by perception separately.
            return 'ok'
        if held:
            spin(0.3)
            held = holding('lifted')  # slipped out while lifting = closes the rest of the way
        if not held and not args.release_home:
            print(f'stopped: nothing in the gripper. Open it (--gripper {table[0][0]:.1f})'
                  ' and go back (--home)')
            return 'stop'
        if args.release_home:
            print(f'holding {args.hold:.0f} s')
            spin(args.hold)
            # fully open as at HOME: the wrist camera sees the fingers the same way every cycle
            ask('release up here?')
            run_gripper(table[0][0], 'release')
            fingers_for(table[0][1])
            # up first, as --home: straight to HOME from here can sweep the open fingers
            # through a wall
            if args.home_lift > 0:
                here = current()
                pose = chain.fk(here)
                try:
                    up, up_how = plan_line(here, pose[:3, 3] + np.array([0.0, 0.0, args.home_lift]),
                                           pose[:3, :3], 'up before HOME')
                except RuntimeError as error:
                    up = None
                    print(f'  note: {error}; to HOME from here')
                if up is not None:
                    ask(f'up {args.home_lift * 100:.0f} cm ({up_how})?')
                    run_trajectory(up, 'up before HOME')
            for extra in range(HOME_EXTRA_LIFTS + 1):
                try:
                    home_segments = plan_home(current(), outbound)
                    break
                except RuntimeError as error:
                    if extra == HOME_EXTRA_LIFTS or args.home_lift <= 0:
                        raise
                    print(f'  note: {error}')
                    here = current()
                    pose = chain.fk(here)
                    higher = pose[:3, 3] + np.array([0.0, 0.0, args.home_lift])
                    up, up_how = plan_line(here, higher, pose[:3, :3], 'more up before HOME')
                    ask(f'up {args.home_lift * 100:.0f} cm more ({up_how})?')
                    run_trajectory(up, 'more up before HOME')
            for i, seg in enumerate(home_segments, 1):
                ask(f'home {i}/{len(home_segments)}?')
                run_trajectory(seg, f'back to HOME {i}/{len(home_segments)}')
            print('done' if held else 'back at HOME with nothing picked')
            return 'ok' if held else 'empty hand'
        if not args.put_back:
            print(f'holding. Put it down by hand: pick_run.py --gripper {open_joint:.3f},'
                  ' then pick_run.py --home (movej needed) or RViz init')
            return 'ok'
        print(f'holding {args.hold:.0f} s')
        spin(args.hold)
        down, down_how = plan_line(current(), option['position'], rotation, 'down')
        ask(f'put it back down ({down_how})?')
        run_trajectory(down, 'down')
        ask('release?')
        run_gripper(open_joint, 'release')
        # back up the finger axis, then to the start pose: own joint path, else OMPL
        here = current()
        up_end = option['hover'] + args.above * rotation[:, 2]
        up, up_how = plan_line(here, up_end, rotation, 'up')
        up_end_values = {**here, **dict(zip(up.joint_trajectory.joint_names,
                                            up.joint_trajectory.points[-1].positions))}
        joints = joint_path(arm_joints, up_end_values, now)
        if not bad_states(joints):
            home = own_trajectory([up_end_values] + joints)
        else:
            home, why = plan_ompl(now, state_of(up_end_values))
            if home is None or bad_states(traj_states(home, up_end_values)):
                raise RuntimeError(f'no path back to the start pose ({why})')
        back_segments = [up, home]
        for i, seg in enumerate(back_segments, 1):
            ask(f'back {i}/{len(back_segments)}?')
            run_trajectory(seg, f'back to the start pose {i}/{len(back_segments)}')
        print('done')
        return 'ok'

    try:
        if args.gripper is not None:  # no movej needed: the arm controller's action
            spin(1.0, lambda: 'joints' in got)
            print(f'gripper now {gripper_now():.3f} rad')
            run_gripper(args.gripper, 'gripper')
            print(f'gripper now {gripper_now():.3f} rad')
            return 0
        mismatch = state_mismatch()
        if args.check_state:
            print(mismatch or 'move_group robot state matches /joint_states (lift and arm)')
            return 1 if mismatch else 0
        if mismatch:
            if hooks:
                hooks.error_code = 'ROBOT_STATE_INVALID'
                hooks.message = mismatch
            print(f'stopped: move_group sees the robot elsewhere ({mismatch}); its collision'
                  ' checks would be wrong. Fix its joint states first (--check-state)')
            return 1
        if args.home:
            return go_home()
        if hooks and hooks.move_to_observe:
            hooks.phase('PICK_OBSERVATION_POSE', 'Moving right wrist to basket observation pose')
            if go_home():
                return 1
            # Observation-pose motion is complete; retrying perception here is safe.
            hooks.motion_started = False
            spin(0.5)
        spin(1.0)  # old latched messages arrive first; only the new plan counts

        def clear_skips():
            if not args.no_plan and clear_client.wait_for_service(timeout_sec=2.0):
                call(clear_client, Trigger.Request())

        # ffw_grasp keeps its skips between runs; an older ffw_grasp has no skip services
        if not args.no_plan and not clear_client.wait_for_service(timeout_sec=5.0):
            print(f'no {args.ns}/clear_skips: restart pick_demo (older ffw_grasp)')
            return 1
        clear_skips()
        cycles = args.cycles if args.release_home else 1
        empty, status = 0, ''
        while cycles == 0 or tally.get('cycle', 0) < cycles:
            count('cycle')
            if cycles != 1:
                print(f'=== cycle {tally["cycle"]}' + (f'/{cycles}' if cycles else '') + ' ===')
            status = pick_once()
            switches = 0
            while status == 'no path' and not args.no_plan:
                if hooks:
                    hooks.check_cancel()
                    if switches >= pick_cfg.plan_tries:
                        hooks.message = 'No collision-free pick after bounded object switches'
                        hooks.error_code = 'PICK_PLAN_FAILED'
                        break
                    hooks.phase('PICK_RESELECT', 'Trying another DYNAMIXEL after no collision-free path')
                switches += 1
                skipped = call(skip_client, Trigger.Request())
                if skipped is None or not skipped.success:
                    print('stopping: could not skip the object (skip_last failed)')
                    status = 'stop'
                    break
                count('switch')
                print(f'other object ({skipped.message})')
                status = pick_once()
            if status == 'ok':
                if args.run:
                    count('picked')
                empty = 0
                clear_skips()  # a neighbor is gone: a skipped object may be free now
            elif status == 'empty hand':
                count('empty_hand')
                empty += 1
                clear_skips()
                if empty >= pick_cfg.empty_tries:
                    status = f'{empty} picks in a row came up empty'
                    break
                print('empty fingers: start over (skips cleared)')
            else:
                break
        ended = {'ok': 'done', 'basket empty': 'basket empty',
                 'stuck': 'no object graspable (needs a person, or a push later)',
                 'no path': 'NO SOLUTION', 'stop': 'stopped (see above)'}.get(status, status)
        if status == 'basket empty':
            print('basket empty')
        print(summary_line(tally, ended))
        return pick_result_code(status, hooks)
    except RuntimeError as error:
        if hooks:
            hooks.message = str(error)
            hooks.stop_confirmed = cancel_active()
        print(f'stopped: {error}')
        if tally:
            print(summary_line(tally, 'stopped (motion or HOME path)'))
        return 1
    except KeyboardInterrupt:
        confirmed = cancel_active()
        if hooks:
            hooks.stop_confirmed = confirmed
            if hooks.error_code != 'SCENE_UNCONFIRMED':
                hooks.error_code = 'CANCELLED' if confirmed else 'STOP_UNCONFIRMED'
            hooks.message = 'Pick cancelled; no release was commanded'
        print(f'interrupted: stop confirmed={confirmed}')
        return 1
    finally:
        worker_executor.shutdown()
        node.destroy_node()
        ros_context.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
