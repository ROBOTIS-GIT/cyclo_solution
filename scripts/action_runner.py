#!/usr/bin/env python3
"""Interactive ROS action client; nothing moves until a menu action is selected."""

import argparse
import os
import select
import signal
import sys
import time

import rclpy
from demo_1f.action import ApproachCup, Handover, RunDemo
from demo_1f.msg import DemoStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions

MENU = """
  d  전체 데모: 시작 자세 → 오른손 셀 픽 → 오른손→왼손 핸드오버
               → 왼손 컵 접근 → 보정·놓기 → 양손 손 빼기 → 시작 자세

  1  시작 자세 (양손 열기, 양팔·리프트·헤드 이동)
  2  현재 자세에서 오른손 셀 픽
  3  오른손 → 왼손 핸드오버
  4  왼손 컵 접근 (저장된 헤드캠 위치 사용)
  5  왼손 손목 보정·셀 놓기
  6  양손 손 빼기 (왼손 → 오른손 순차 상승·후퇴)

  r  실패 단계 재개
  c  현재 작업 취소
  p  상태 확인
  q  종료 (진행 중이면 취소 결과를 기다림)
명령 입력 + Enter로 실행합니다. Ctrl+C는 취소 후 종료 요청입니다.
"""


def selection(command, cup_id):
    demo_commands = {"1": "initial", "2": "pick", "d": "start", "r": "resume"}
    if command in demo_commands:
        return "demo", RunDemo.Goal(command=demo_commands[command], cup_id=cup_id)
    if command == "3":
        return "handover", Handover.Goal(direction="right_to_left")
    cup_commands = {"4": ("approach", "left"), "5": ("refine", "left"), "6": ("withdraw", "both")}
    if command in cup_commands:
        action, arm = cup_commands[command]
        return action, ApproachCup.Goal(arm=arm, cup_id=cup_id, execute=True)
    return None


class Runner(Node):
    def __init__(self, args):
        super().__init__(f"demo_action_runner_{os.getpid()}")
        self.args = args
        self.action_clients = {
            "demo": ActionClient(self, RunDemo, "/run_demo"),
            "handover": ActionClient(self, Handover, "/handover"),
            "refine": ActionClient(self, ApproachCup, "/refine_cup"),
            "withdraw": ActionClient(self, ApproachCup, "/withdraw_cup"),
            "approach": ActionClient(self, ApproachCup, "/approach_cup"),
        }
        self.subscription = self.create_subscription(
            DemoStatus,
            "/handover_demo/status",
            self.on_status,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.status = None
        self.status_time = 0.0
        self.pending = False
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_sent = False
        self.quitting = False
        self.last_feedback = None

    def on_status(self, msg):
        self.status, self.status_time = msg, time.monotonic()

    def busy(self):
        return self.pending or self.goal_handle is not None

    def send(self, command):
        selected = selection(command, self.args.cup_id)
        if selected is None:
            print("지원하지 않는 명령입니다.")
            print(MENU)
            return
        if self.busy():
            print("액션 진행 중입니다. 완료를 기다리거나 c로 취소하세요.")
            return
        key, goal = selected
        client = self.action_clients[key]
        if not client.server_is_ready():
            print("액션 서버가 준비되지 않았습니다. 별도 터미널에서 full_demo.launch.py를 실행하세요.")
            return
        self.pending = True
        self.cancel_requested = self.cancel_sent = False
        self.last_feedback = None
        try:
            client.send_goal_async(goal, feedback_callback=self.feedback).add_done_callback(
                self.accepted
            )
        except Exception:
            self.pending = False
            raise
        detail = getattr(goal, "command", getattr(goal, "direction", getattr(goal, "arm", "")))
        print(f"요청 전송: {key} {detail}")

    def feedback(self, msg):
        f = msg.feedback
        state = (f.phase, getattr(f, "detail", ""), getattr(f, "execution_state", ""))
        if state != self.last_feedback:
            print("진행:", " | ".join(value for value in state if value))
            self.last_feedback = state

    def accepted(self, future):
        self.pending = False
        try:
            handle = future.result()
        except Exception as error:
            # Acceptance is unknown; prevent sending another goal or claiming cancellation.
            self.pending = True
            print(f"요청 응답 오류, 수락 여부 미확인: {error}. 서버 상태를 확인하세요.")
            return
        if not handle.accepted:
            print("서버가 요청을 거절했습니다. 동작 서버 로그에서 사유를 확인하세요.")
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.finished)
        print("액션 수락됨")
        if self.cancel_requested:
            self.cancel()

    def finished(self, future):
        try:
            response = future.result()
        except Exception as error:
            print(f"최종 결과 수신 오류: {error}. 완료/정지 여부 미확인; 서버를 확인하세요.")
            return
        result = response.result
        status = {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}.get(
            response.status, str(response.status)
        )
        print(f"결과: {status}\n코드: {result.error_code or '-'}\n설명: {result.message}")
        if hasattr(result, "can_resume"):
            print(f"마지막 단계: {result.last_completed_stage}, 재개 가능: {result.can_resume}")
        if hasattr(result, "target_pose") and result.target_pose.header.frame_id:
            p = result.target_pose.pose.position
            print(
                f"목표 [{result.target_pose.header.frame_id}]: X={p.x:.4f} Y={p.y:.4f} Z={p.z:.4f} m"
            )
        self.goal_handle = None
        self.cancel_sent = False
        print("다음 명령을 입력하세요. (1~6 / d / r / c / p / q)")

    def cancel(self):
        if not self.busy():
            print("이 실행기에서 진행 중인 액션이 없습니다.")
            return
        self.cancel_requested = True
        if self.goal_handle is None:
            print("수락 응답 대기 중: 수락되면 즉시 취소 요청합니다.")
        elif not self.cancel_sent:
            self.cancel_sent = True
            self.goal_handle.cancel_goal_async().add_done_callback(self.cancel_response)
            print("취소 요청 전송. 정지/최종 결과를 기다립니다.")

    def cancel_response(self, future):
        try:
            if not future.result().goals_canceling:
                self.cancel_sent = False
                print("취소가 수락되지 않았습니다. 액션 최종 결과를 계속 기다립니다.")
        except Exception as error:
            self.cancel_sent = False
            print(f"취소 응답 오류: {error}; 정지 확인 아님")

    def show_status(self):
        print(
            "서버:",
            {name: client.server_is_ready() for name, client in self.action_clients.items()},
        )
        if self.status is None:
            print("아직 상태 메시지가 없습니다.")
            return
        s = self.status
        print(
            f"상태 수신 {time.monotonic() - self.status_time:.1f}초 전: {s.task_state} / {s.phase}"
        )
        print(f"피드백 fresh={s.robot_state_fresh}, 실행={s.execution_state}, 코드={s.error_code}")
        print(s.detail)

    def command(self, text):
        text = text.strip().lower()
        if text == "q":
            self.quitting = True
            if self.busy():
                self.cancel()
        elif text == "c":
            self.cancel()
        elif text == "p":
            self.show_status()
        elif text:
            self.send(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cup-id", default="cup")
    args = parser.parse_args()
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = Runner(args)
    interrupted = False

    def stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"ROS_DOMAIN_ID={os.getenv('ROS_DOMAIN_ID', '0')} RMW={os.getenv('RMW_IMPLEMENTATION', 'default')}"
    )
    print("실행기 준비. 서버는 별도 터미널에서 full_demo.launch.py로 실행하세요.")
    print(MENU)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if interrupted:
                interrupted = False
                node.command("q")
            if node.quitting and not node.busy():
                break
            if not node.quitting and select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                node.command(line if line else "q")
    finally:
        if node.busy():
            print("액션 결과 미확인. 서버 상태를 확인하세요.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
