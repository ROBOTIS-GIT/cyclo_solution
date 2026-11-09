# ffw_pick

`ffw_grasp` 가 발행한 집기 후보로 **MoveIt 을 통해 오른팔을 움직여 집는** 패키지다.
경로는 이 패키지의 해석 IK 로 만들고, MoveIt 은 장면·충돌 검사·실행만 맡는다(MoveIt IK·Pilz LIN 은 쓰지 않음).

필요한 것: bringup(`/robot_description`, `/joint_states`), `ffw_grasp` 의 `pick_demo`(plan 서비스), move_group.
movej(같은 팔 컨트롤러에 명령을 보냄)는 켜지 않는다.

## 실행 `pick_moveit`

```bash
ros2 run ffw_pick pick_moveit --verbose                           # 계획만: RViz Planned Path, 팔 안 움직임
ros2 run ffw_pick pick_moveit --run --step --release-home          # 1 회, 동작마다 Enter (q = 멈춤)
ros2 run ffw_pick pick_moveit --run --release-home --cycles 0      # 실패하거나 Ctrl+C 까지 반복
ros2 run ffw_pick pick_moveit --gripper 0                          # 그리퍼만 (rad, 0 = 열림)
ros2 run ffw_pick pick_moveit --home                               # 10 cm 올린 뒤 HOME (구간마다 확인)
ros2 run ffw_pick pick_moveit --check-state                        # move_group 상태 = /joint_states 인지
```

- `--run`: 열기 → 접근(관절 보간 + 손가락 축 직선) → 닫기 → 들기 15 cm(`--lift`, 실제 자세에서 다시 계획) → 유지
- `--release-home`: 들어 올린 자리에서 완전히 열어 놓고, 수직 10 cm(`--home-lift`) 뒤 HOME(초기 자세 3). `--cycles N` 으로 반복.
  빈손(닫은 그리퍼가 `gripper_table.yaml` 의 `hold_below` 0.94 rad 보다 더 닫힘)이면 HOME 뒤 다시 plan, 연속 2 번이면 멈춤
- `--put-back`: 유지 뒤 제자리에 내려놓고 손가락 축으로 빠져 시작 자세로
- Ctrl+C: 지금 동작을 취소하고 그 자리에 멈춘다
- 장면: `ffw_grasp` 가 잰 바구니(바닥 + 기울어진 벽 4)·탁자 + 손끝에 손가락 연장 상자. 손가락-벽 충돌도 검사(`--allow-finger-walls` 로 허용)
- 로봇 모델은 `/robot_description` 에서 받는다(`--urdf 파일` 로 대신). 관절 속도 한계도 여기서(URDF 4.8 rad/s)

## 설정 파일

| 파일 | 내용 | 옵션 |
|---|---|---|
| `config/arm_r.yaml` | 팔 하나: MoveIt 그룹, 기준·손끝·손목 링크, 관절·그리퍼 이름, 손가락 링크, 그리퍼 액션, movej 토픽, HOME(초기 자세 3), 관절 한계 여유, 가속·손끝 속도 한계 | `--arm` |
| `config/pick.yaml` | 동작: `ns`, `above`, `lift`, `speed`, `accel`, `turn_window`, `budget`, `hold`, `home_lift` (같은 이름의 옵션이 이긴다), 재시도 횟수·대기, 빈손 연속 횟수, 상태 비교 허용치 | `--config` |
| `config/gripper_table.yaml` | 그리퍼 관절 ↔ 벌림 표, 잡음 기준 `hold_below`, 닫기 명령, 열고 닫는 시간 | `--table` |
| `ffw_grasp` `dynamixel_top_grasp.yaml` | 바구니·손가락 치수, MoveIt 장면 값(탁자, 손가락 상자 층) | `--task` |
| `ffw_grasp` `robot_calibration.yaml` | 손끝 길이(`tip_below_end_effector`) 등 이 로봇의 보정값 (task 파일 다음에 읽어 이긴다) | `--calibration` |

파일에 빠진 값이나 모르는 값이 있으면 `settings: ...` 한 줄을 내고 멈춘다. `--verbose` 는 처음에 읽은 파일과 한계를 출력한다.

## 탐색

1. 후보 = `grasp_candidates` × 그리퍼 180° 두 방향. 초기 자세 3 의 TCP 방향에서 덜 돌아간 순.
   가장 덜 돌아간 것보다 `--turn-window`(15°) 안은 모두 풀고 **관절 이동이 가장 작은** 것, 밖은 처음 통과한 것
2. 후보 하나: above(hover 에서 손가락 축으로 10 cm 뒤) → 집기 → 15 cm 위 직선을 1 cm 마다 **해 전부** 놓고(`analytic_ik`)
   동적 계획법으로 관절 이동 최소 경로(시작점까지 이동 + 구간마다 가장 많이 움직인 관절의 합, 1 cm 에 0.15 rad 넘게 뛰면 안 이어짐) → 5 mm 점 채움 → 다듬기
3. 그리퍼 자체(7 번 링크 아래·손가락 상자)가 장면에 닿는 점이 있으면 그 후보는 바로 실패(팔 모양과 상관없음).
   팔이 닿으면 그 해를 빼고 다음 최단 경로, 지금 자세 → above 관절 보간이 닿으면 그 시작 해를 빼고 다시. 모두 안 되면 OMPL(above 까지)
4. 이미 찾은 최선보다 관절 이동이 클 수밖에 없는 후보는 건너뜀. `--verbose` 줄 끝 `[총 s: IK, paths, 충돌 검사 횟수 s, scene s]`

## 해석 IK `analytic_ik`

어깨 축 1·2·3 이 한 점, 손목 축 5·6 이 한 점에서 만난다. 7 번 각도를 정하면 4 번은 어깨–손목 거리로, 5·6 번은 벡터 하나를 두 축으로 돌려 맞추는 식으로,
1·2·3 번은 남은 어깨 회전으로 바로 나온다(갈래 8 개). 7 번을 1° 격자로 훑고, 갈래가 없음·한계 밖·한계 안 사이로 바뀌는 곳과 이웃 해가 0.02 rad 넘게 벌어진 곳은
반씩 6 번까지 더 나눈다(0.016°). 해가 없으면 그 자세는 정말 닿지 않는다(격자 해상도 안에서). 모든 해는 FK 로 다시 확인(1e-6).

## 시험

```bash
cd ffw_pick && python3 -m unittest discover -s test -v
```

`test/arm_r.urdf` 는 ffw_sg2_follower URDF 의 base_link → end_effector_r_link 사슬(운동학만)이다.
