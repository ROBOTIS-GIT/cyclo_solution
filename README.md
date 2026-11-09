## 전체 데모: 다이나믹셀 픽 → 핸드오버 → 컵에 놓기

TaskManager가 전체 순서와 제한된 자동 재시도·취소·재개를 관리합니다.
전체 실행 순서는 초기 자세 → 셀 픽 → 오른손에서 왼손으로 핸드오버 → 컵 접근 → 보정·놓기 → 손 빼기 → 초기 자세 복귀입니다.
초기 자세는 `config/demo.yaml`에서 설정하며 헤드는 0.69rad, 리프트는 -0.1m입니다.
픽 재시도나 중간 단계 재개 때 완료한 초기 자세 이동은 반복하지 않습니다.
`ai_worker`의 인식·파지·해석 IK 코드는 `modules/dynamixel_pick/`에 포함되어 있습니다.
빌드, 인식 의존성, 실행 및 복구 방법은 [통합 데모 설명](docs/dynamixel_demo.md)을 참고하세요.

```bash
# 기존 ROS 컨테이너 안에서, 새 이미지 없이 빌드
cd /root/ros2_ws/src/demo_1f
./scripts/build_workspace.sh
source /root/ros2_ws/install/setup.bash
ros2 launch demo_1f full_demo.launch.py use_sim_time:=false
# 다른 터미널: 1(시작 자세), 2(현재 자세에서 픽), d(전체), r(재개), c(취소)
ros2 run demo_1f action_runner.py
```

`1`은 양손을 연 뒤 양팔·리프트·헤드를 하나의 MoveIt 경로로 시작 자세에 이동합니다.
팔 접기나 리프트 중간 높이 경유는 없습니다. `2`는 초기 자세 이동 없이 현재 자세에서 셀을 집습니다.
컵/셀 인식이나 픽 백엔드 없이 동작 서버만으로 실행할 수 있습니다.
직접 요청하려면 `ros2 action send_goal /run_demo demo_1f/action/RunDemo '{command: initial}' --feedback`을 사용합니다.

시작만으로는 움직이지 않습니다. 브링업·카메라는 기존 노드를 사용합니다.

# SG2 demo_1f

ROS 2 Jazzy 데모. 핸드오버, 셀 놓기, 컵 파지 전 접근을 각각 실행합니다.

- `/handover`: 왼손→오른손 또는 오른손→왼손의 고정 핸드오버.
- 헤드 RGB-D 카메라: YOLOv8n-seg 컵 몸체 검출 + depth로 표면 XYZ 계산.
- `/approach_cup`: 헤드캠으로 검출한 컵 근처까지 TCP 이동.
- `/refine_cup`: 손목캠 입구 관측 → 보정·하강 이동 → 그리퍼 열기.
- `/withdraw_cup`: 각 손의 현재 TCP에서 10cm 상승 → TCP 정면 반대로 8cm 후퇴; 그리퍼 유지.
- `/pick_cup`: 빈 그리퍼를 열고 컵 몸통 파지 전 대기 위치까지 이동 후 정지.

로봇·카메라 bringup과 Zenoh router는 기존 컨테이너에서 실행합니다.
이 컨테이너는 들어가도 자동으로 로봇을 움직이지 않습니다.
데모 컨테이너의 소스는 `/root/ros2_ws/src/demo_1f`에 있고, 빌드 결과는
`/root/ros2_ws/build`, `/root/ros2_ws/install`, `/root/ros2_ws/log`에 생성됩니다.

## 실행

로봇 호스트에서:

```bash
cd /home/robotis/demo_1f
bash docker/container.sh start
bash docker/container.sh enter
```

컨테이너 기본 통신은 `ROS_DOMAIN_ID=30`, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`입니다.
ARM 빌드에서는 필요하면 `DEMO_BASE_IMAGE=robotis/ai-worker:2.2.8`을 지정합니다.
Jetson에서 TensorRT를 사용할 때 컨테이너에 NVIDIA GPU runtime이 제공되어야 합니다.
Compose는 NVIDIA runtime, host 네트워크·IPC, TTY와 `restart: unless-stopped`를 사용합니다.
Python 추론 환경은 Dockerfile에서 이미지에 포함하므로 별도 설치나
`pick_python:=` 지정 없이 전체 데모를 실행할 수 있습니다.
첫 이미지 빌드는 Python 패키지 다운로드에 인터넷 연결이 필요합니다.

양손의 4.5cm 연장은 공통 모델 `sg2_extended_grippers.urdf`에 반영됩니다.
데모의 TCP는 `gripper_l_tcp` / `gripper_r_tcp`이며, 기존 `end_effector_[lr]_link`의
로컬 -Z 방향으로 4.5cm 떨어져 있습니다. 픽·핸드오버·컵 접근·놓기가 이 기준을
사용합니다. 기존 브링업 TF는 유지하고, 데모가 두 TCP의 고정 TF를 추가합니다.

### 핸드오버

데모 컨테이너 터미널 1:

```bash
ros2 launch demo_1f handover.launch.py \
  config:=/demo_config/handover.yaml use_sim_time:=false
```

같은 컨테이너의 다른 터미널:

```bash
ros2 action send_goal /handover demo_1f/action/Handover \
  '{direction: left_to_right}' --feedback

# 반대 방향
ros2 action send_goal /handover demo_1f/action/Handover \
  '{direction: right_to_left}' --feedback
```

양팔 접근 → 받는 손 열고 접근 → 받는 손 닫기 → 주는 손 열기 → 양팔 후퇴 순서입니다.
현재 `handover.yaml`의 속도·가속도 스케일은 0.40입니다. 연속 팔 이동 구간은
MoveIt 계획을 이어 시간 재계산하며, 그리퍼 인계 단계는 순서를 지킵니다.
시퀀스 성공은 센서로 물체 인계를 확인했다는 뜻은 아닙니다.
받는 손의 `receiver_close`는 `gripper_wait_only: true`로 닫힘 목표각 검사를 생략합니다.
속도 제한을 적용한 실제 명령 시간과 컨트롤러 성공 응답을 기다린 뒤 주는 손을 엽니다.
물체 파지는 확인하지 않으며, 다른 그리퍼 단계와 오류·취소 처리는 기존대로 유지합니다.

### 헤드캠 인식과 브라우저

카메라 bringup을 켠 상태에서 로봇 **호스트**의 별도 터미널:

```bash
bash /home/robotis/demo_1f/scripts/test_perception.sh
```

또는 데모 **컨테이너**에서:

```bash
ros2 launch demo_1f perception_test.launch.py \
  host:=0.0.0.0 profiles:=/demo_config/perception_test.yaml
```

브라우저: `http://192.168.6.2:8765/`
출력 토픽: `/perception_test_head/cup`

화면은 헤드 인식 RGB와 헤드·왼손·오른손 depth를 함께 보여줍니다. 원본 숫자 depth는
흑백으로 표시하며 가까울수록 밝습니다(표시 범위 0.1~2.0m). 카메라 colorizer는 꺼서
손목 depth의 16UC1(mm) 값을 유지합니다. 왼손 거리 필터는 기본 22~38cm만 남기고
나머지 픽셀을 0으로 만듭니다. 웹의 최소·최대(cm) 입력으로 실행 중 범위를 바꿀 수 있습니다.
범위 변경은 현재 뷰어 세션에만 적용되며 재시작 기본값은 `left_depth_min_m`/`left_depth_max_m`
ROS 파라미터로 지정합니다. 필터 후 숫자 영상은 `~/left/depth_filtered`에 같은 촬영 시각·
좌표계·인코딩으로 발행합니다. 거리 필터는 현재 관찰 자세용입니다.

웹의 **왼손 depth — 컵 후보**는 필터 결과의 연결 영역을 분리하고 depth CameraInfo로
크기를 계산해 컵 크기에 가까운 후보를 표시합니다. 이웃 픽셀의 깊이 차이가 1.5cm를 넘는
경계는 끊어서 컵과 배경이 몇 픽셀로 붙는 경우를 줄입니다. 기본 비교 지름은 10cm이며
`candidate_diameter_m`, CameraInfo 토픽은 `depth_left_info_topic`으로 설정합니다.
화면 가장자리에 잘린 영역은 제외하고, 후보 점수가 비슷하면 선택을 보류합니다.
노란 윤곽은 후보, 초록 윤곽은 3개 이상의 서로 다른 프레임에서 2cm 이내로 유지된
선택 후보입니다. 이는 의미 인식의 확정이나 정확도 보증이 아닙니다.
십자와 optical frame XYZ는 보이는 표면 점들의 중앙값이며 입구 중심·이동 목표가 아닙니다.
이 표면 중심은 모션 목표로 사용하지 않습니다. 아래의 손목 입구 추정 노드가 별도로 확정한 입구 중심만 보정 액션에서 사용합니다.
가림 때문에 나뉘거나 책상과 붙은 컵은 누락될 수 있습니다.
`/depth/left_candidate.jpg` 및 `/status`의 `candidate`에서 같은 진단 결과를 확인할 수 있습니다.

`config/perception_test.yaml`에서 RGB/depth/CameraInfo 토픽을 설정합니다.
입력은 같은 optical frame·해상도에 등록된 rectified RGB와 depth여야 합니다.
32FC1 depth는 m, 16UC1은 mm로 해석합니다. RGB/depth 시각 차이는 40ms 이내여야 합니다.

```bash
# ROI를 좁히거나 GPU 사용을 명시하는 예시 (호스트)
bash /home/robotis/demo_1f/scripts/test_perception.sh \
  ai_backend:=tensorrt ai_score:=0.35 'roi:=[0,0,672,376]'
```

YOLO는 컵 마스크를 찾고, 마스크 안의 유효 depth와 CameraInfo로 XYZ를 계산합니다.
여러 컵·잘린 후보·부족한 depth·오래된 영상은 거절합니다. 검정색 자체로 분류하지 않습니다.
BODY XYZ는 **보이는 표면점**이며 컵 중심이나 입구 실측값이 아닙니다.
모델·TensorRT 엔진에 관한 내용은 [models/README.md](models/README.md)를 참고하세요.

뷰어 없이 `/perception/cup`만 발행하려면 `config/perception.yaml`의 헤드 토픽을 채운 후:

```bash
ros2 launch demo_1f perception.launch.py config:=/demo_config/perception.yaml
```

인식 노드를 중복 실행하지 마세요. 뷰어는 이미지·상태만 표시하며 로봇 명령을 보내지 않습니다.

### 메뉴 실행기

서버는 별도 터미널에서 `ros2 launch demo_1f full_demo.launch.py use_sim_time:=false`로 실행합니다.
실행기는 서버를 시작·종료하지 않고 액션 요청과 상태 확인만 담당합니다.

```bash
# 6.2 호스트
bash /home/robotis/demo_1f/scripts/run_actions.sh
# 컨테이너 안
ros2 run demo_1f action_runner.py
```

| 입력 | 동작 |
|---|---|
| `d` | 전체 데모: 시작 자세 → 오른손 셀 픽 → 오른손→왼손 핸드오버 → 왼손 컵 접근·보정·놓기 → 양손 손 빼기 → 시작 자세 |
| `1` | 시작 자세; 양손 열기 및 양팔·리프트·헤드 이동 |
| `2` | 현재 자세에서 오른손 셀 픽; 초기 자세 이동 없음 |
| `3` | 오른손→왼손 핸드오버 |
| `4` | 저장된 헤드캠 컵 위치로 왼손 접근 |
| `5` | 왼손 손목 보정·셀 놓기 |
| `6` | 양손 손 빼기; 왼손→오른손 순차 상승·후퇴 |
| `r` | 저장된 실패 단계 재개 |
| `c` | 이 실행기에서 보낸 액션 취소; 최종 정지 결과 대기 |
| `p` | 서버·로봇 상태 확인 |
| `q` / Ctrl+C | 진행 중 액션 취소 후 실행기 종료 |

번호가 이전 메뉴와 다릅니다. 예전 `i/k/g`, `0/7/8/9/10/11`, 방향 선택,
준비 자세, 미리보기 및 서버 시작 `s`는 실행기에서 제거했습니다.
명령 + Enter로 요청하며, 최종 결과가 오기 전에는 다음 모션을 보내지 않습니다.
실행기를 종료해도 별도 실행한 서버·카메라는 유지됩니다.

컵 잡기는 파지 전 접근까지만 구현되어 있습니다. `/pick_cup` 완료 후 전진·닫기·들어올리기는
수행하지 않습니다. 이후 ACT 등 별도 모듈을 연결할 수 있으며 현재 학습 정책은 포함하지 않습니다.

### 컵 근처로 손 이동

핸드오버 launch를 종료한 뒤 데모 컨테이너에서 실행합니다. 카메라는 그대로 둡니다.
두 launch를 동시에 실행하면 같은 동작 노드·MoveIt이 중복되므로 하나만 사용합니다.

```bash
ros2 launch demo_1f approach_cup.launch.py \
  config:=/demo_config/handover.yaml \
  approach_config:=/demo_config/approach.yaml \
  cup_topic:=/perception_test_head/cup \
  use_sim_time:=false
```

다른 데모 컨테이너 터미널:

```bash
# 목표 좌표 미리보기. 계획·로봇 이동 없음.
ros2 action send_goal /approach_cup demo_1f/action/ApproachCup \
  '{arm: left, cup_id: cup, execute: false}' --feedback

# 실제 접근. 도착하면 멈추며 그리퍼는 그대로 유지.
ros2 action send_goal /approach_cup demo_1f/action/ApproachCup \
  '{arm: left, cup_id: cup, execute: true}' --feedback
```

오른손은 `arm: right`로 선택합니다. 미리보기 성공은 IK·충돌 없는 경로의 성공을 보장하지 않습니다.
`~/approach_target`은 TCP 목표, `~/cup_estimate`는 TF 변환된 BODY 표면점입니다.

`config/approach.yaml`의 현재 설정:

| 항목 | 동작 |
|---|---|
| 목표 XYZ | BODY X/Y + `offset_xy`, BODY Z + `cup_height` + `offset_z` (현재 0.165 − 0.05 = +0.115m) |
| 컵 높이 | 0.165m; 목표 Z에 다른 여유·불확실성 높이를 추가하지 않음 |
| 컵 충돌 모델 | BODY Z를 중간 높이로 가정, 모델 입구는 Z+8.25cm |
| 수평 충돌 범위 | 컵 반지름 + 벽 두께 + 관측 불확실성 |
| TCP 롤·피치 | 몸체 기준 roll ±10°, pitch 아래쪽 20~60°, yaw 정면~안쪽 90° |
| 속도 | 접근 0.3 |

중간 높이 가정은 사용자가 선택한 테스트 가정이며 실제 입구 측정이 아닙니다.
TF는 촬영 시각의 좌표계를 변환할 뿐 표면점을 컵 중심으로 바꾸지 않습니다.
컵 몸체 인식의 크기 설정과 접근의 `inner_radius`를 일치시켜야 합니다.

SG2 TCP -Z는 손의 전방, +X는 손등/카메라 방향입니다. 롤 제한은 손목 관절 하나의 제한이 아닙니다.
roll·pitch 제한은 도착 목표에만 적용합니다. 시작 자세가 범위 밖이어도 사전 정렬 없이 컵으로 접근합니다.
이동 중 TCP 방향은 제한하지 않으므로 손이 기울거나 회전할 수 있습니다. 충돌·관절 제한 검사는 유지됩니다.
기본 설정에서 도착 목표는 추종 여유 1°를 두고 roll ±9°, pitch 아래쪽 21~59°로 계획하며,
완료 시 실제 TCP가 설정 범위(roll ±10°, pitch 아래쪽 20~60°) 안인지 확인합니다.
`tcp_pitch_downward_only: true`는 도착 시 위쪽을 보는 자세를 제외합니다. 코드의 ZYX pitch는 양수가 아래쪽이며,
`tcp_pitch_min_deg: 20`, `tcp_pitch_limit_deg: 60`은 사람이 표현하는 고도각으로 -60~-20°입니다. `tcp_yaw_inward_only: true`이면 도착 yaw를 왼손 -90~0°, 오른손 0~90°로 제한합니다.
정면은 로봇 +X 기준이며, 계획에는 경계에서 1° 여유를 둡니다. 바깥쪽·뒤쪽 목표는 허용하지 않습니다.
`false`로 설정하면 yaw가 다시 자유로워집니다. 이동 중에는 yaw도 제한하지 않습니다.
기존 `align_scale` 설정은 더 이상 사용하지 않습니다.
`tcp_roll_limit_deg: 0`은 roll 제한을 해제합니다. Pitch를 해제하려면
`tcp_pitch_downward_only: false`, `tcp_pitch_min_deg: 0`, `tcp_pitch_limit_deg: 0`을 함께 설정합니다.
책상이 로봇 기준 수평이라는 가정이며, 검출된 책상 평면을 기준으로 삼는 것은 아닙니다.

MoveIt은 URDF와 등록된 충돌물을 검사합니다. 잡힌 셀의 형상을 새로 추정·부착하지 않으며,
기존 부착물과 외부 장애물은 유지합니다. 책상·다른 장애물은 필요 시 PlanningScene에 등록해야 합니다.
RobotContext의 `HeadCupTracker`가 액션 실행 여부와 관계없이 헤드 BODY 관측을 계속 저장합니다.
촬영 시점 TF로 planning frame(`base_link`)에 변환된 유효 관측 3개의 평균 위치와 불확실성을
확정합니다. 관측 사이 간격은 1초 미만, 위치 변화는 2cm 미만이어야 합니다. 갑작스러운
좌표 변화는 새로운 위치의 3개 관측이 모인 뒤 반영하며, 손목 RIM 관측은 여기에 섞지 않습니다.
컵이 가려지거나 검출이 실패해도 마지막 확정 위치와 원래 영상·수신 시각은 유지합니다.
접근 요청 이전에 저장한 위치도 `approach.max_age: 30.0`초 안이면 사용할 수 있습니다.
이 시간이 지나면 새 관측을 요구하며, 재조회나 잘못된 관측으로 저장 시각을 갱신하지 않습니다.
입력 영상 자체의 최신성·TF 대기 제한(`state_max_age`)은 기존대로 유지합니다.
접근 시작 시 저장 위치와 충돌 모델을 함께 고정합니다. 백그라운드 저장은 계속되지만
그 접근의 목표나 MoveIt 장면을 매 프레임 바꾸지 않습니다. 다음 요청은 최신 확정 위치를 사용합니다.
추적 상태는 동작 노드 메모리에 있고 재시작하면 초기화됩니다. 로봇 본체가 고정된 데모를
전제로 하며, 본체 이동 중 과거 `base_link` 위치를 유지하는 전역 SLAM 지도는 아닙니다.
시작 시 관측 유효성·최신성 검사, 충돌 모델 적용 확인, 관절 피드백·정지 확인과 취소 처리는 유지됩니다.
기존 `max_cup_shift`, `reobserve_timeout`은 더 이상 사용하지 않습니다.
이 동작은 컵 근처 접근까지입니다. 도착 후 손목 보정은 `/refine_cup` 액션으로 별도 실행합니다.
실패 설명에는 검출 실패 원문, 안정 프레임 수, 관측 지연, depth 유효 비율, 크기·불확실성 문제를 구분해 표시합니다.
TF가 영상보다 늦게 도착하면 최대 32개 메시지를 보관해 정확한 촬영 시각으로 재시도하며,
최신 TF로 대체하지 않습니다. 오래된 관측은 폐기합니다.

## 코드 구조와 읽는 순서

동작 노드는 `handover_demo` 하나이며 내부 모션은 C++ 객체입니다. 인식과 웹 뷰어는
별도 프로세스입니다. 먼저 `src/nodes/main.cpp`에서 객체 생성과 모션 등록을 읽고,
`src/tasks/task_manager.cpp`, 원하는 모션의 `tick()`, `runtime/execution` 순서로 보면 됩니다.

```text
include/demo_1f/              공개 헤더; 아래 소스와 같은 역할별 경로
  core/                      공통 데이터, 결과, ExecutorPort 실행 인터페이스
src/
  nodes/                     main.cpp(객체 구성), check_feedback.cpp(진단)
  tasks/                     액션 접수·실행 소유권·취소·결과
  motions/
    common/                  CupMotion, HeadCupObserver, 컵 설정·장면
    handover/                HandoverMotion
    place/                   ApproachCupMotion, RefineCupMotion, WithdrawCupMotion
    pick/                    PickCupMotion, 파지 전 목표 기하·설정
  runtime/
    config.cpp               로봇/핸드오버 설정 로딩·검증
    state/                   관절 피드백, 컵 관측·헤드 위치 추적, TF, 상태 공유
    execution/               실행 수명, MoveIt 목표, 경로 준비, 그리퍼, 완료 판정
    geometry/                TCP 각도 계산
  perception/                헤드 YOLO + depth 인식
  moveit_plugins/            관절 경계 보정, TCP 자세 제약 샘플러
scripts/
  action_runner.py           사람이 입력하는 액션 실행기
  wrist_perception.py        손목 depth 구독·TF·RIM 관측 발행
  wrist_cup_geometry.py      ROS와 독립적인 후보/입구 계산
  perception_viewer.py       뷰어 ROS 노드와 영상 상태
  demo_perception/           숫자 depth 처리, HTTP API, 웹 화면
config/                      동작·인식 설정
launch/                      노드 실행 구성
test/                        기하, 상태 전이, 취소, ROS/웹 회귀 테스트
```

| 모듈 | 입력 → 처리 → 결과 |
|---|---|
| `TaskManager` | 액션 요청 → 실행 중복 차단·모션 tick → 피드백/결과 |
| `HandoverMotion` | 방향별 시퀀스 → 양팔 접근·인계·후퇴 |
| `ApproachCupMotion` | 헤드 BODY 관측 → 충돌 장면 적용 → 컵 근처 접근 |
| `RefineCupMotion` | 손목 RIM 관측·필요 시 pitch 조정 → 낮아진 중심으로 이동 → 그리퍼 열기 |
| `WithdrawCupMotion` | 지정한 손 또는 양손 순차 실행 → 방향 유지·10cm 상승 → TCP 뒤로 8cm 후퇴 |
| `PickCupMotion` | 헤드 BODY 관측 → joint6 기준 접근 방향 계산 → 그리퍼 열기·파지 전 위치에서 정지 |
| `HeadCupTracker` | 헤드 BODY 상시 수집 → 유효 3프레임 평균 갱신 → 가려져도 마지막 위치 유지 |
| `HeadCupObserver` | 저장된 헤드 위치의 수명·치수·오차 확인 → 접근용 스냅샷 |
| `CupMotion` | 컵 모션 공통 시작·취소·실패·명령 전달 |
| `RobotContext` | 관절 상태·컵 관측·TF 공유, 로봇에 명령하지 않음 |
| `TrajectoryExecutor` | Step → MoveIt/그리퍼 요청 → 완료·취소·정지 확인 |
| `ApproachScene` | 데모 충돌 모델 적용과 응답 확인 |

컵 잡기 접근은 셀 놓기 모션을 내부에서 실행하지 않습니다. 둘 다 `HeadCupObserver`를
사용하며, `RobotContext`의 상시 추적 결과를 읽습니다. 요청 후 다시 3프레임을 모으지 않습니다.
손목 입구의 안정성 판정은 별도로 유지합니다.

## 모듈 구성과 재사용

빌드 라이브러리는 `demo_runtime`(피드백·실행), `demo_motions`(모션),
`demo_tasks`(액션 서버), `cup_detector`(헤드 인식)로 나뉩니다.
ROS 패키지·액션 이름·메뉴 번호는 유지합니다. 공개 C++ 헤더는 역할별 하위 경로로 이동했습니다.

`main.cpp`에서 컵 모션을 등록합니다. TaskManager에는 구체적인 컵 모션 클래스가 들어가지 않습니다.
새 컵 모션은 `CupMotion`을 구현하고 아래 목록에 추가합니다.

```cpp
#include "demo_1f/tasks/task_manager.hpp"
#include "demo_1f/motions/place/approach_cup_motion.hpp"
#include "demo_1f/motions/place/refine_cup_motion.hpp"
#include "demo_1f/motions/place/withdraw_cup_motion.hpp"
#include "demo_1f/motions/pick/pick_cup_motion.hpp"

// context/executor/scene → 모션 → TaskManager 순서로 생성합니다.
// 등록한 모션 객체는 TaskManager보다 오래 살아 있어야 합니다.
demo_1f::TaskManager tasks(node, context, handover, {
    {"/approach_cup", approach},
    {"/refine_cup", refine},
    {"/withdraw_cup", withdraw},
    {"/pick_cup", pick},
});
```

핸드오버와 준비 자세는 같은 `HandoverMotion`을 사용하고 각각 `/handover`,
`/prepare_handover`로 받습니다. 컵 모션은 공통 `ApproachCup.action`을 사용합니다.
추후 학습 모듈이 다른 입력/결과를 필요로 하면 해당 액션 어댑터를 별도로 추가하면 됩니다.

모션의 공통 흐름은 `start()` → 주기적 `tick()` → `result()`입니다. 모션은 `ExecutorPort`에
명령하며 실제 ROS/MoveIt 구현은 `TrajectoryExecutor`에 있습니다. 테스트는 이 인터페이스에
가짜 실행기를 넣어 로봇 없이 단계 전이와 명령 순서를 확인합니다.

`TaskManager`는 20ms 타이머로 진행 중인 모션을 갱신하며 하나의 실행 소유권을 유지합니다.
미리보기는 명령을 발행하지 않습니다. 취소 시 `cancel()` 후 `running()`이 끝날 때까지
콜백과 tick을 처리합니다. 정지·장면 적용이 미확정이면 다음 모션을 차단합니다.
대기 중에는 새 trajectory 명령을 발행하지 않습니다.

제거한 항목은 사용되지 않는 파지 접촉 범위 판정, 손목 BODY 전용 입력 분기,
`refine_lower_speed`, 중복된 초기 안정화 검사, 이전 `handover_core` 링크 별칭입니다.
컵 모션의 명령 전 준비 실패 코드는 `ROBOT_NOT_READY`로 통일했습니다.
취소·정지 확인, 관측 최신성, 장면 적용 및 충돌 검사는 동작에 필요한 조건으로 유지합니다.
공개 헤더를 직접 포함하던 외부 C++ 코드는 새 경로에 맞춰 다시 빌드해야 합니다.

`build/`, `install/`, `log/`는 생성물입니다. 소스 백업은 프로젝트 밖에 보관합니다.

## 검증과 개발

```bash
# 로컬 린트 환경에 requirements-lint.txt 설치 후
bash scripts/lint.sh

# 실제 로봇과 통신하지 않는 --network none 테스트 컨테이너
bash test/run_isolated.sh /home/gyu/ws/ai_worker
```

테스트는 핸드오버 양방향, 시작 시 무동작, 취소·피드백 검증, BODY+지연 TF 접근,
롤 범위, 목표 높이, 그리퍼 미변경, AI 마스크/depth 처리를 검사합니다.
컨트롤러 통합 테스트는 GenericSystem 가상 하드웨어이며 물체 접촉·실제 grasp를 검증하지 않습니다.

평면 추정·셀 투하 액션은 포함하지 않습니다. 왼손 depth 입구 추정과 TCP 중심 보정은 아래 별도 액션으로 제공합니다.
이전 실행의 `placement_config`는 이제 `approach_config`이며 파일은 `approach.yaml`입니다.
헤드 인식 테스트는 `camera`/`detector` 선택 없이 위 명령으로 실행합니다.


## 왼손 손목캠 입구 보정

소스 변경을 빌드한 뒤 기존 동작 launch와 실행기를 재시작하면 적용됩니다.
기존 컨테이너에서 colcon으로 빌드할 수 있으며 소스 변경마다 이미지 재빌드는 필요하지 않습니다.
시작 자체는 로봇을 움직이지 않습니다.

1. `test_perception.sh`로 인식/웹을 실행합니다. `wrist_perception.py`가 함께 시작됩니다.
2. `run_actions.sh`의 **4**로 왼손 접근을 실행합니다.
3. **5**로 손목 보정·셀 놓기를 실행합니다. **6**은 양손 손 빼기, **c**는 현재 액션 취소입니다.
   미리보기가 필요하면 아래 ROS 액션을 직접 호출합니다.

```bash
ros2 action send_goal /refine_cup demo_1f/action/ApproachCup \
  '{arm: left, cup_id: cup, execute: false}' --feedback
# 실제 보정 (입구 미확정 시 pitch 조정 포함)
ros2 action send_goal /refine_cup demo_1f/action/ApproachCup \
  '{arm: left, cup_id: cup, execute: true}' --feedback
```

현재 왼손만 지원합니다. 오른손 요청은 거절됩니다. 기존 `/approach_cup`, 핸드오버와
하나의 태스크 매니저·executor를 공유하므로 동시에 실행할 수 없습니다.

- 뷰어의 `left_exclude_rect: [0.0, 0.32, 0.23, 0.72]`는 **원본 depth 영상 기준**
  정규화 `[x0,y0,x1,y1]`입니다. 영상 왼쪽 중앙의 그리퍼 사이 영역을 제외하며, 마스크 뒤의
  컵도 함께 제외합니다. 그리퍼나 카메라 장착이 바뀌면 재조정해야 합니다.
  실행 옵션 예: `bash scripts/test_perception.sh 'left_exclude_rect:=[0.0,0.32,0.23,0.72]'`.
- 거리 필터와 마스크를 통과한 숫자 depth를 `wrist_perception.py`가 받습니다.
  촬영 시점 TF로 `base_link`에 변환하고, 후보 상단 8mm의 점에 수평 원을 맞춥니다.
  컵이 똑바로 서 있다는 가정입니다. 반지름·잔차·호 범위(측정 반지름 120°, 알려진 반지름 75° 이상)·분산을 검사한 뒤
  3개 프레임의 중심 변화가 8mm 이내일 때만 입구로 발행합니다.
- `/perception/wrist_cup`은 기존 `CupObservation`의 `observation_type=2`,
  `source=wrist_left`를 사용하며 `body_position` 필드에 **입구 중심**을 담습니다.
  `valid=false` 메시지의 위치는 사용하지 않습니다. 헤드 BODY 메시지는 기존 type=1입니다.
- 새 영상이 계속 들어오지만 입구를 확정하지 못하면 TCP 위치를 유지한 채 hand pitch를
  5°씩 증가시킵니다. 최소 3개의 서로 다른 영상과 1초(`refine_view_wait`)의 관찰이 필요합니다.
  `RIM_UNSTABLE: 1/3` → `2/3` → 유효 관측 안정 프레임처럼 안정화 단계가 진행될 때는
  현재 각도를 유지하고 1초의 관찰 시간을 다시 확보합니다. 같은 단계가 반복되거나
  후보가 계속 모호하면 다음 각도에서 재관측합니다. hand yaw/roll은 유지합니다.
  두 번 제한을 제거했으며, 최종 목표 pitch는 `refine_pitch_ceiling_deg: 75.0`를 넘지 않습니다.
  마지막 조정은 상한까지 남은 각도만 움직입니다. 전체 탐색은 `refine_search_timeout: 15.0`초로
  제한하며 관찰·계획·피치 실행을 모두 포함합니다. 시간이 지나면 이동 중인 피치를 취소하고
  정지 확인 후 종료합니다. 입구가 확정되면 탐색 제한을 끝내고 기존 중심 보정으로 넘어갑니다.
  후보가 여러 개인 경우 임의로 하나를 선택하지 않습니다. 입력 단절·TF 오류에는 움직이지 않으며,
  새 입력 대기는 기존 `refine_wait: 3.0`초 제한을 유지합니다. 미리보기는 회전하지 않습니다.
  충돌 검사되는 Pilz LIN의 제자리 회전을 사용하며, 각도 상한이나 계획 실패 시 정지합니다.
- 입구 확정 후 TCP의 XY를 입구 중심에 맞춥니다. 도착 방향은 현재 방향 주위에서 회전 벡터 각 성분 ±10° 여유를 주고,
  최종 Z는 `max(현재 TCP Z, 입구 Z + 5cm) - 3cm`입니다. 12cm를 넘는 보정은 거절합니다.
  측정된 입구 높이로 컵 충돌 모델을 갱신하고, 충돌 검사되는 OMPL을 속도 비율 0.20, 가속도 비율 0.12로
  실행합니다. TCP 경로는 출발/도착 위치를 감싸는 상자에 각 방향 4cm 여유를 둡니다.
  이동 중 방향은 고정하지 않으며, 실패 시 충돌 검사를 해제하지 않습니다.
- 중심 보정 계획은 처음 3초, 이후 최대 두 번 각각 5초 재시도합니다. 마지막 시도는 경로 여유 6cm를 사용합니다.
  계획·실행 궤적이 없는 계획 시간 초과만 재시도하며, 관절 정지 및 시작 위치 유지를 다시 확인합니다.
  실제 실행 실패나 취소는 재시도하지 않습니다. 관찰 pitch 속도 비율은 `refine_speed: 0.06`입니다.
- 최종 목표를 확보한 뒤의 가림은 저장한 짧은 보정을 취소하지 않습니다.
  중심 보정과 3cm 하강을 하나의 OMPL 이동으로 실행한 뒤 그리퍼를 0.0으로 엽니다.
  이동 실패·취소 시 열기 명령을 보내지 않습니다.
  `refine_correction_speed: 0.20`, `refine_correction_acceleration: 0.12`,
  `refine_lower_distance: 0.03`, `refine_gripper_open: 0.0`으로 조정합니다.
  출발·목표 TCP는 측정 입구보다 최소 1.5cm 위여야 하며 경로 상자의 아래쪽만 이 높이로 제한하며 X/Y 및 위쪽 여유는 기본 4cm로 유지합니다. 충돌 검사를 유지합니다.
  셀 파지 오프셋·셀 충돌 형상은 아직 반영하지 않습니다.
  손 빼기는 별도 `/withdraw_cup` 액션(메뉴 `6`)으로 실행합니다. 메뉴 `6`과 전체 데모는
  `arm: both`로 왼손 → 오른손을 순차 실행합니다. 동시 이동은 아닙니다.
  `arm: left` / `right`로 한 손만 실행할 수도 있습니다. 양손 미리보기는 지원하지 않으며
  각 손을 따로 미리보기합니다. 컵 재인식 없이
  현재 TCP 방향을 유지한 채 세계 좌표 Z로 10cm 상승하고,
  TCP 정면의 반대(+TCP Z)로 8cm 후퇴합니다. 두 구간은 충돌 검사되는 Pilz LIN이며
  속도는 `refine_correction_speed`, 가속도는 그 값의 1/4을 사용합니다.
  후퇴 방향이 아래를 향하면 중단합니다. 이동 실패·취소 시 다음 구간을 실행하지 않습니다.
  `5`번은 셀 놓기까지만 수행합니다. 수동 테스트는 `5` → `6` → `1` 순서입니다.
  `6`은 그리퍼나 초기 자세 이동을 실행하지 않습니다. `d` 전체 데모는 놓기 후
  `WITHDRAW_CUP` → `RETURN_INITIAL` 순서로 손 빼기와 양팔·리프트·헤드 복귀를 실행합니다.
  한 손이 실패·취소되면 나머지 손이나 초기 자세 복귀를 시작하지 않습니다.
  컵 쪽 책상 전체 충돌 모델은 이 변경에서 추가하지 않았습니다.
  보정 액션 완료는 하강·그리퍼 열기 완료이며 실제 셀의 컵 안 착지는 확인하지 않습니다.

웹의 **왼손 입구 중심 추정**에서 사용한 점(파랑), 추정 원/중심(자홍), 상태를 확인합니다.
`RIM_UNCERTAIN`은 컵 후보는 있지만 입구를 확정할 근거가 부족하다는 뜻입니다.
표면 중심으로 대신 이동하지 않으며, 제한된 pitch 조정 후에도 확정되지 않으면 정지합니다.

정상 그리퍼 단계는 `gripper_completion_window: 0.25`와 `gripper_settle_time: 0.1`을 사용합니다.
취소·정지 확인은 `gripper_stop_window: 0.5`와 `settle_time: 0.25`를 유지합니다.
그리퍼 명령 시간은 0.6초이며 관절 제한에 따른 궤적 시간 재계산으로 더 길어질 수 있습니다.

핸드오버 완료 후 양팔 TCP는 각자의 인계 위치에서 좌우 바깥쪽으로 각각 5cm만 후퇴합니다.
손 방향과 높이는 유지하며, `d` 전체 데모는 후퇴 완료 후 바로 저장된 헤드 컵 위치로 접근합니다.
시야 확보용 추가 후퇴나 새 헤드 관측 대기는 없습니다. 저장 위치가 없거나 수명이 지난 경우에는 재관측을 기다립니다.

핸드오버 준비·인계·후퇴 목표의 전방 위치는 arm_base_link 기준 X=0.33m입니다. 기존 0.38m에서 몸쪽으로 5cm 당겼으며 높이와 손 방향은 유지합니다.

## Cup pickup approach

Cup pregrasp is a reusable API module and is not included in the demo runner menu.

`/pick_cup` uses `demo_1f/action/ApproachCup` (`arm: left`,
`cup_id: cup`, `execute: true`). Both arms are supported through the action API.
Approach infers cup bottom from head BODY midpoint minus half configured cup height.
The hand targets horizontal with roll/pitch ±10°; yaw is allowed ±15° around
the initial joint6 wrist-center-to-cup XY direction from the first planning attempt.
The wrist center (`arm_l_link6` / `arm_r_link6`) is captured once at request start.
Within 5 cm horizontal distance, the initial TCP forward yaw is retained instead;
a missing wrist/TCP TF or undefined heading aborts. The motion target remains TCP-based.
The target is 12 cm behind the grasp TCP along that direction, at bottom + 6.5 cm.
The TCP/pad offset is -3 cm (initial URDF estimate, not calibrated). No path-wide tilt
constraint is applied; the measured arrival pose is returned when the approach finishes.
OMPL samples the goal angle ranges; there is no separate ranked IK candidate search.

Approach opens the empty gripper, moves to the pregrasp pose and stops before contact.
This module performs no wrist BODY detection, correction, insertion, closing or lifting.
`/pick_cup_refine` and menu entries 12/13 have been removed. A future learned pickup
policy can take over after the approach action succeeds; no ACT policy is connected yet.
Head BODY perception, handover and wrist RIM refinement for cell placement are unchanged.

```bash
ros2 action send_goal /pick_cup demo_1f/action/ApproachCup \
  '{arm: left, cup_id: cup, execute: false}' --feedback
```


손목 림 인식은 45mm 연장 그리퍼의 현재 관찰 영상에 맞춰 왼손 depth의
22cm 미만 영역(그리퍼·셀)을 제외합니다. 그리퍼 사이 배경은 정규화 사각형
`[0.0, 0.32, 0.23, 0.72]`로 제외하며 `left_exclude_rect`로 조정할 수 있습니다.
거리 범위는 웹에서 변경할 수 있습니다. 카메라 장착·그리퍼 형상·셀 파지 위치가
달라지면 이 마스크를 다시 확인해야 합니다.
모든 크기 후보에 실제 depth 기반 림 검사를 적용한 뒤, 유효한 림이 하나일 때만
중심 안정화로 진행합니다. 유효한 림이 여러 개면 선택을 보류합니다.
가려진 부분의 depth를 채우지 않으며, 기존 호 길이·반지름·오차 조건은 유지합니다.

컵 접근 도착 pitch는 수평에서 아래쪽 **20~60°**입니다 (`config/approach.yaml`).
림의 자유 반지름 추정이 실패하면 설정된 안쪽 반지름과 벽 두께의 중간값
(현재 47.5mm)을 고정해 부분 테두리로 중심을 추정합니다. 웹 상태의
`method=known_radius`가 이 방식을 뜻합니다. 가려진 depth를 채우지는 않습니다.
서로 다른 중심이 비슷하게 설명되거나 점들이 거의 한 방향으로만 분포하면
거부합니다. 추정 오차 15mm 이하와 기존 3프레임 중심 일치 검사를 모두
통과해야 보정 목표로 사용할 수 있습니다. 반지름은 측정 결과가 아닌 모델 가정입니다.

`require_passive_gripper_stationary: true`일 때 손목 보정 시작 전 그리퍼가 움직이면
최대 2초 동안 정지 피드백을 기다립니다. 기존 정지 허용치는 유지하며, 계속 움직이면
`ROBOT_NOT_READY`로 중단합니다. 피드백이 오래되면 즉시 중단하고 취소도 가능합니다.
정지는 정상 파지를 보증하지 않습니다. 셀이 밀려 파지 위치가 달라졌다면
그리퍼·셀 마스크와 놓기 위치의 오프셋도 달라질 수 있습니다.

현재 실로봇 설정은 전류 제한 파지를 위해 `require_passive_gripper_stationary: false`입니다.
받는 그리퍼의 `gripper_wait_only` 닫기는 목표 위치·정지를 파지 성공으로 검사하지 않고,
명령 시간과 컨트롤러 완료를 기다립니다. 다음 동작 준비와 팔 동작 완료에서도
유지 중인 그리퍼의 정지 여부를 요구하지 않습니다. 팔 궤적 실행 시 유지 중인
그리퍼를 제외해 새 위치 유지 명령으로 파지 명령을 덮어쓰지 않습니다.
명시적인 열기 동작, 팔 상태 검사, 최신 피드백, 충돌 검사와 취소 시 정지 검사는
유지합니다. 경로 계산 중 실제 관절 위치가 시작 허용치를 벗어나면 여전히 중단합니다.
`handover_confirmed`는 계속 false이며 실제 파지를 확인하지 않습니다.

핸드오버·컵 접근·컵 중심 보정의 OMPL 요청은 `config/handover.yaml`의
`ompl_planning_attempts: 3`으로 후보 경로를 비교합니다. MoveIt의 ParallelPlan이
기존 `planning_time`(또는 단계별 planning_time) 안에서 최대 3개 RRTConnect 경로를
탐색하고, 관절 공간의 누적 이동 거리로 비교하며 유효한 구간을 연결해 더 짧게
다듬습니다. 계획 시간을 후보 수만큼 곱하지 않지만, 이전보다 출발 대기시간이나
CPU 사용량이 늘 수 있습니다. 1로 설정하면 단일 탐색으로 돌아갑니다(허용 범위 1~8).

이 설정은 공통 실행기를 사용하는 다른 OMPL 동작에도 적용됩니다. 핸드오버는
구간별로 비교한 뒤 기존 연결·시간 보정·충돌 검사를 수행합니다. 컵 접근의 도착
각도 조건과 보정의 TCP 이동 범위는 유지합니다. Pilz LIN/PTP와 셀 집기의 자체 IK는
변경하지 않습니다. 탐색한 후보 중 짧은 경로를 선호하며 전역 최단 경로나 TCP·팔꿈치
높이 상한을 보장하지 않습니다. 적용 후 **MoveIt과 동작 노드를 함께 재시작**해야 합니다.

### DYNAMIXEL 셀 집기 업데이트 (2026-11-06)

`modules/dynamixel_pick`는 내장 `ai_worker`의 `9202268`까지 셀 인식·파지 변경을 통합했습니다.
`dynamixel_pick.launch.py`는 `dynamixel_basket_hybrid.yaml`을 사용합니다.
분할 모델(`dynamixel_seg_n640.pt`)과 회전 박스 모델(`dynamixel_obb_n640.pt`)을 함께 사용해
OBB 중심과 분할 경계의 방향을 조합하고, 측정 높이에 맞는 셀 면 크기를 파지 계획에 전달합니다.
OBB 중심용 `camera_x_bias`는 업데이트된 보정값 `+0.008 m`입니다.

일반 파지 후보를 먼저 검사하고, 실패하면 좁은 그리퍼 벌림·기울임 후보를 검사합니다.
충돌 없는 경로가 없는 셀은 이번 요청에서 제외하고 다른 셀을 시도합니다.
데모 Action에서는 다른 셀로 전환하는 횟수도 `pick.yaml`의 `plan_tries`(현재 3)로 제한하며,
취소와 정지 확인은 기존대로 유지합니다. 바구니가 비었으면 `OBJECT_NOT_FOUND`로 보고합니다.
셀 하나를 집어 들어올리면 TaskManager에 반환하고 핸드오버로 진행합니다.
독립 실행용 반복 놓기·HOME 복귀는 데모 흐름에서 사용하지 않습니다.

연장된 `gripper_r_tcp`와 `tip_below_end_effector: 0.0`을 유지하므로 45 mm를 중복 가산하지 않습니다.
새 모델은 소스 패키지에 포함되어 있습니다. 이미지 재생성 없이 기존 컨테이너에서 패키지를 빌드하고,
셀 퍼셉션·파지 백엔드를 포함한 데모 launch를 재시작하면 적용됩니다.

### 파지 상태 판정 비활성화 (2026-11-06)

전체 데모는 파지 상태를 추정·저장하지 않으며 이전 파지 상태로 `d`/`1`/`2`를 막지 않습니다.
셀 집기 Action은 그리퍼 관절값의 `hold_below` 기준으로 성공을 판단하지 않고,
닫기와 들어올리기의 정상 완료를 보고합니다. 카메라 기반 파지 확인은 아직 없습니다.
기존 ROS 인터페이스의 `holding_state`는 호환용 `UNVERIFIED`, `PickDynamixel.holding`은
항상 false이며 실행 조건으로 사용하지 않습니다. 진행 중 실행 잠금, 취소·정지 확인,
충돌 검사, 단계별 실패 처리 및 명시적인 `resume` 체크포인트는 유지됩니다.
`d`/`1`는 양손을 여는 시작 자세를 포함하므로 실행자가 물체 유무를 확인합니다.
직접 실행하는 upstream CLI의 독립적인 파지 판정은 변경하지 않았습니다.

### 컵 중심 보정의 계획 재시도 (2026-11-06)

`CORRECT_TO_CUP_CENTER`는 `/move_action`에 `plan_only: true`로 먼저 계산을 요청하고,
성공한 궤적을 `/execute_trajectory`로 그대로 실행합니다. 실행 직전에는 최신 피드백과
계획 시작 관절값의 일치를 확인합니다. 보정 궤적에는 추가 smoothing을 적용하지 않습니다.
계획만 요청한 응답에 계획·실행 궤적이 모두 없을 때 `FAILURE(99999)` 및
`PLANNING_FAILED(-1)`를 `PLANNING_FAILED`로, `TIMED_OUT(-6)`을 `PLANNING_TIMEOUT`으로
구분합니다. 보정은 팔과 TCP가 그대로인 경우에만 최대 두 번 재계산합니다.
탐색 시간은 3초 → 5초 → 5초, TCP 경로 여유는 4cm → 4cm → 6cm입니다.
세 번 모두 실패하면 종료하며 그리퍼를 열지 않습니다. 실행 중 실패·취소·정지 미확인은
이 재시도에 포함되지 않습니다. 다른 컵 접근과 핸드오버의 계획/실행 흐름은 유지됩니다.

손목 림의 `known_radius` 보조 추정은 가려진 테두리를 활용하도록 원호 최소 범위를
75°로 설정합니다. 반지름을 함께 추정하는 기본 방식은 120°를 유지합니다.
중심 오차 15mm 이하, 높이 편차 3.5mm 이하, 중심 모호함 및 3프레임 안정화 검사는 유지합니다.

### 손목 관측 지연 및 75° 소비 조건 정합

손목 인식은 DDS 영상 큐를 1개로 유지하고, 각 timer callback에서 영상 시각의 TF가
준비된 가장 최신 프레임 하나만 계산합니다. 최신 프레임의 TF가 아직 없으면
더 오래된 준비 완료 프레임을 사용할 수 있으며 최신 TF로 대체하지 않습니다.
원호 계산 뒤에도 원본 영상이 0.8초보다 오래되면 유효 목표로 발행하지 않습니다.
기존 관측 만료 및 TF 대기 제한은 유지합니다.
보정 모듈의 원호 confidence 하한은 인식기의 75°와 같은 `75/360 ≈ 0.2083`입니다.
RobotContext의 손목 안정화 수신 간격은 기존 영상 시각 간격과 동일한 0.6초이며,
기존 헤드 원시 관측 카운터의 0.3초는 유지하지만, 접근은 별도 헤드 추적기의
3프레임·최대 1초 간격과 저장 수명을 사용합니다. 손목 안정화와 중심 오차·치수 검사는 유지합니다.
