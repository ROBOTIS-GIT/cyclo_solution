# ffw_grasp

`ffw_vision` 의 물체 관측(`ObjectObservationArray`, 픽셀)과 정렬 depth, TF 로 **위에서 수직으로 집는 자세**를 base_link 기준으로 계산해 발행하는 패키지다.
팔은 움직이지 않는다. 발행한 자세로 팔을 움직이는 것은 이 패키지 밖(사용자가 실행)이다.

## 노드 `grasp_planner`

서비스 `plan`(`std_srvs/Trigger`)을 부르면 그때부터 관측 `frames` 개를 모아 한 번 계산한다. 계속 계산하지 않는다.

처리 순서:

1. 프레임마다 `STATUS_VALID` 이고 `top_height` 가 있는 물체 중 점수가 가장 높은 것 하나 (+ `support_plane`)
2. 중심이 중앙값에서 `max_center_jitter_px` 넘게 떨어진 프레임, 다수와 자세(`pose_label`)가 다른 프레임은 버림. 남은 것이 `min_frames` 미만이면 실패(버린 이유별 개수 출력).
   정사각형 판정(`angle_period`)이 프레임마다 다르면 버리지 않고 '어느 변이 긴지 불확실'(짧은 주기)로 본다 — 줄여 잰 윗면 비율이 기준 근처면 판정이 흔들림
3. 자세·점수·높이는 남은 프레임의 중앙값, 탁자 평면 중앙값, 픽셀별 depth 중앙값
4. 첫·마지막 프레임 촬영 시각의 TF 를 비교해 카메라가 움직였으면 실패 (`max_camera_motion`, `max_camera_turn_deg`)
5. 윗면 3D: `u`, `v` 의 광선과 평면을 `top_height` 만큼 올린 면의 교점 = 중심. 크기는 양 끝 투영 거리가 `length`, `width` 가 되는 값.
   **프레임마다 3D 로 구한 뒤** 합친다(중심·크기 중앙값, 방향은 주기 고려 평균) — 정사각형에 가까운 윗면은 프레임마다 긴 변이 바뀌어 픽셀 값을 바로 합치면 크기가 틀림
6. 후보 두 개: A = 짧은 변을 집음(긴 변과 수직으로 닫힘), B = 긴 변을 집음. 벌림 = 크기 + 양쪽 `clearance`(최대 `max_open`).
   `angle_period` 가 π/2(세움, 정사각형에 가까움)면 어느 변이 긴지 모르므로 둘 다 긴 변 크기로 벌림.
   `known_short`, `known_long` 을 주면 크기 = max(측정값, 그 값) — Vision 윗면은 마스크를 줄여 재므로 실제보다 작다(눕힘 46.5 mm → 37~43 mm). 위치·방향은 측정값 그대로
7. 손가락 자리 충돌 검사: 손가락 바깥쪽에 손가락 모양(손끝 `finger_tip` → `taper_height` 위 `finger_thick`, 최대 `finger_max`, 폭 `finger_width`) 안에
   손끝보다 높은 장면 점이 `obstacle_points` 개 이상이면 막힘 (이웃 물체, 바구니 벽)
   단, depth 점프 한가운데 픽셀(양쪽 `veil_step` px 이웃과 모두 `veil_jump` 넘게 차이, 그 사이 값)은 뺀다 — 물체 가장자리에서 윗면·바닥 거리가 섞여
   시선을 따라 허공(카메라 반대쪽 손가락 자리)에 생기는 점. 이것 때문에 가운데 물체도 한 후보가 막혔음(2026-10-29)
   **기울여 접근**(`max_tilt_deg` > 0): 닫히는 축을 중심으로만 기울인다(손가락은 쥐는 면과 계속 평행). 그리퍼 윗부분을 `lean_x`, `lean_y`(로봇 쪽) 방향으로,
   닫힘 축과 수직인 방향이 그쪽과 맞는 만큼 최대 `max_tilt_deg` 까지. 막히면 `tilt_step_deg` 씩 줄여 수직까지 시도. 기울어진 손가락은 높이에 따라 기울기 쪽으로 밀려 검사한다
   몸통 검사(`finger_length` > 0): 그리퍼 몸통(`body_*`, 손끝에서 축 따라 `finger_length` 부터)과 다음 링크(`link_*`, 카메라 포함, `link_start` 부터)를 손가락 축 위쪽으로 끝까지 뻗은 상자로 보고, 그 안의 장면 점도 막힘으로 센다. 기울기별로 다시 검사
   손끝 높이: 윗면 아래 `grasp_depth`, 단 가장 낮은 손가락 모서리(기울어진 끝면 `finger_width`/2·sin, 닫힐 때 `close_drop`)가 바닥(`floor_height`, 예: 바구니 바닥) 위 `floor_clearance` 이상
   `long_side_poses` 의 자세는 측정 비율이 `square_ratio` 미만이면 어느 변이 긴지 모르므로 두 후보 모두 `known_long` 으로 벌림 (줄여 잰 윗면이 정사각형처럼 보임)
7-0. **손가락과 바구니 벽 접촉 허용**(`wall_band` > 0, 사용자 2026-11-02): 측정한 바구니 모델의 벽 안쪽 면에서 `wall_band` 안쪽부터 바깥 크기 + `wall_band` 까지의 점은 손가락 검사에서 뺀다
   (벽 가운데가 안쪽으로 볼록, 파란 벽 depth 가 프레임마다 흔들림, 바구니는 조금 밀려도 됨). 몸통·손목 검사는 벽을 그대로 본다. 이웃 물체·바구니 밖 물체는 계속 막음.
   그래서 바구니 측정을 집기 계산보다 먼저 한다. `pick_moveit.py` 도 MoveIt 에서 손가락(연장부·URDF 손가락)과 벽의 접촉을 허용한다(`--check-finger-walls` 로 끔)
7-1. **롤**(`max_roll_deg` > 0, 모든 수직 계열 자세가 막혔을 때만): 그리퍼를 가로축 중심으로 `roll_step_deg` 씩 최대 `max_roll_deg` 까지 양쪽으로 기울여 다시 검사한다 —
   손가락 위쪽 두꺼운 부분(25~30 mm)이 바구니 벽에 닿는 경우, 그 손가락 윗부분을 벽에서 떼어 냄. 대신 두 손끝 높이가 달라져서, 낮은 손끝은 바닥 여유(`tip_height_for`, 닫힐 때 내려감 포함) 위,
   높은 손끝은 윗면 `roll_top_margin` 아래여야 함(물체가 낮으면 못 기울임). 손가락·몸통 검사는 기울어진 자세 그대로(강체) 한다. 막히지 않은 수직 자세가 하나라도 있으면 롤은 쓰지 않는다
   (2026-10-30 녹화 89 장: 성공 85 그대로, 막힘 4 중 1 장(세움, 벽 옆) 이 롤 25° 로 풀림. 넓은 면이 위인 눕힘(28.5 mm)은 손끝이 들어갈 높이가 모자람)
8. 막히지 않은 후보 중 **손목을 덜 돌리는 쪽**(사용자 2026-10-29: 항상). 덜 돌리는 쪽이 로봇 쪽으로 기울이기도 좋아 팔 동작이 작다. 둘 다 막히면 실패
9. 자세: 손끝 목표 = 윗면 중심 아래 `grasp_depth`(탁자 위 최소 `min_tip_height`), z 축 = base 위쪽, y 축 = 닫히는 방향(수평).
   그리퍼가 대칭이므로 y 는 지금 `end_effector_frame` 의 y 와 가까운 쪽. `end_effector_frame` 목표 = 손끝 목표 + 위로 `tip_below_end_effector`.
   hover = 손가락 축을 따라 `hover_height` 뒤 (기울면 로봇 쪽으로 비켜 있음 — 검사한 부피 안에서 축을 따라 접근).
   높이 보정은 원인별로 따로: 손끝 목표에서 `camera_z_bias`(카메라가 높게 본 만큼) 빼고, end_effector 명령에서 `arm_z_bias`(팔이 명령보다 높이 가는 만큼) 뺌
10. 바구니(`basket_length` > 0): 같은 프레임의 파란 픽셀(`basket_*` HSV, ffw_vision 과 같은 값) 중 depth 점프 한가운데를 뺀 점 →
    탁자 위 높이 상위 3 % = 테두리 높이(`basket_height` ± 2 cm 아니면 실패) → 그 높이 ± `basket_rim_band` 점을 평면에 내려 최소 사각형으로 방향 →
    양 끝(0.5·99.5 %) 사이 = 잰 크기, 중심은 그 가운데. 한 변이 화면 끝에 잘려(끝 `basket_edge_px` 안의 점 + 잰 길이가 알려진 크기보다 1 cm 넘게 짧음)
    있으면 보이는 반대쪽 끝에서 알려진 크기만큼. 바구니 실패는 집기 실패가 아니다(응답에 이유, `basket_pose` 안 냄).
    wrist2 녹화 10장: 잰 테두리 321~332 × 239~243 mm (실측 320 × 240), 바깥 벽면은 depth 가장자리 섞임이라 빼야 함(빼지 않으면 376 × 264)

| 출력 (namespace `/ffw_grasp/<camera_id>`) | 내용 |
|---|---|
| `plan` (서비스) | 응답 `success`, `message`(자세·후보·벌림·좌표·손목 회전·탁자 z·사용 프레임 수 또는 실패 이유) |
| `grasp_pose` | `geometry_msgs/PoseStamped`, `base_frame` 기준 `end_effector_frame` 목표(손끝이 아님, 응답의 `tip z` 가 손끝). stamp = 마지막 관측 촬영 시각 |
| `hover_pose` | 같은 방향, `hover_height` 위 |
| `gripper_opening` | `std_msgs/Float64`, 벌림(m). 그리퍼 관절 값으로 바꾸는 것은 아직 없음 |
| `debug_image` | 마지막 프레임 위에 초록 = 잰 윗면, 하늘색 = 고른 손가락 자리, 빨강 네모 = 막힌 손가락 자리(손끝 높이에서 본 모양만), 빨강 점 = 그 손가락을 막는 장면 점(손가락 위쪽 두꺼운 부분까지 검사), 회색 = 안 고른 자리, 주황 = 바구니 테두리(알려진 크기). 막힘 이유는 `blocked 85 (39-61mm high)` 처럼 막는 점의 높이(탁자 위, 10~90 %)를 함께 — 물체 윗면보다 높으면 벽이나 더 높은 이웃 |
| `basket_pose` | `PoseStamped`, 바구니 바깥 바닥 중심(탁자 평면 위), x = 긴 변, z = 탁자 법선. 카메라 보정값을 뺀 값. stamp 는 `grasp_pose` 와 같음 |
| `table_pose` | `PoseStamped`, 물체 아래 탁자 평면의 점, z = 법선. 카메라 보정값을 뺀 값 |
| `object_pose` | `PoseStamped`, 물체 윗면 중심, x = 긴 변, z = 탁자 법선. 카메라 보정값을 뺀 값. **집을 수 없어도(모두 막힘) 발행** — 높이 = `table_pose` 와의 z 차이 (`pick_moveit.py --shake` 가 장면의 물체 상자로 씀) |
| `grasp_candidates` | `geometry_msgs/PoseArray`, 막히지 않은 후보 × 기울기 전부의 `end_effector_frame` 목표. 첫째 = `grasp_pose`, 그다음 같은 후보의 작은 기울기, 다른 후보 순 — 팔 쪽이 IK·충돌로 고름 |
| `grasp_candidate_info` | `std_msgs/Float64MultiArray`, `grasp_candidates` 순서대로 5 개씩: 벌림(m), 기울기(rad), 0 = A 1 = B, 손목 회전(rad), 롤(rad, + = end effector +y 쪽 손끝이 높음) |

- 출력 토픽은 모두 transient local(마지막 값 유지). `ros2 topic echo` 로 보려면 `--qos-durability transient_local`.
- `end_effector_r_link`(arm_r_link7 z −0.215) 와 손끝: URDF(표준 RH-P12-RN) 메시로는 벌림 58~70 mm 에서 2 mm 이내, 쥐며 닫히면(34 mm) 약 6 mm 더 내려감(손가락 링크 구조). **실제 손가락은 4.5 cm 더 길어서**(사용자 2026-10-30 정정, 처음엔 5 cm) task 설정 `tip_below_end_effector: 0.045`. `arm_z_bias` 는 같은 hover 자 측정을 4.5 cm 로 다시 계산한 값(+5 mm).
- 기울기를 쓰는 이유(2026-10-29): 3차 초기 자세는 팔꿈치가 접힌 모양이라 수직 접근은 관절 하나가 2~3 rad 도는 큰 동작이 필요. 50° 까지 기울이면 대부분 0.3~1.1 rad. 단, 물체 긴 변이 로봇을 향하면(A 닫힘 축이 로봇 방향) 거의 못 기울여 큰 동작이 남음.
- 기울이면 높이 오차가 수평 오차로도 나타난다(축을 따라 접근): 높이 보정을 먼저 정확히.
- 탁자 z 는 측정 평면 그대로라 손목 카메라 오차(탁자 약 +16 mm)가 포함된다. 세 보정값(`tip_below_end_effector` 그리퍼 형상, `camera_z_bias`, `arm_z_bias`)은 하나로 합치지 않는다 — 원인을 구분하기 위해 (사용자 2026-10-29). 응답 끝에 세 값이 함께 나온다.
- 보이지 않는 곳(물체 뒤 그림자)은 장애물로 보지 않는다. 지금은 바구니에 1개만 두므로 벽만 검사 대상이다.
- task 설정: `config/dynamixel_top_grasp.yaml` (오른손, 손끝 실측값, `finger_width` 는 가정, DYNAMIXEL 크기 28.5 × 34 × 46.5 mm → `known_short` 0.034, `known_long` 0.0465).
  코드 기본값을 쓰던 값도 모두 이 파일에 적혀 있다(2026-11-02). `ffw_pick` 이 읽는 MoveIt 장면 값(벽 두께·높이, 탁자 크기, 손가락 상자 층)도 이 파일 끝에 있다(노드 파라미터 아님).
- 로봇 보정: `config/robot_calibration.yaml` (`tip_below_end_effector`, `camera_x/y/z_bias`, `arm_z_bias`) — 작업 파일과 나눔. launch 인자 `calibration_file`, task 파일 다음에 읽어 이 값이 이긴다.
- 바구니 색: `ffw_vision` 의 `config/blue_basket.yaml` (launch 인자 `basket_file`, 검출기와 같은 파일).

```bash
ros2 launch ffw_grasp grasp_planner.launch.py
ros2 service call /ffw_grasp/right_wrist/plan std_srvs/srv/Trigger
```

한 번에 켜기 (bringup 다음): 손목 카메라 + 자동 검출(화면 8765, `detect_port`) + `grasp_planner` + 파지 결과 화면(8766, `grasp_port`). Ctrl+C 한 번에 카메라까지 꺼진다.
YOLO 가중치는 환경 변수 `FFW_VISION_MODEL` 또는 `model_path:=` 로 바꾼다(기본은 `ffw_vision` README).
팔 컨트롤러(cyclo movej)는 포함하지 않는다.

```bash
ros2 launch ffw_grasp pick_demo.launch.py
```

필요한 것: 손목 카메라, `ffw_vision` 자동 검출(`dynamixel_basket_seg.yaml`), 로봇 TF(bringup). TF 가 없는 bag 재생으로는 계산할 수 없다.

## 시험

```bash
cd /root/ros2_ws/src/ai_worker/ffw_grasp && python3 -m unittest discover -s test -v 2>&1 | tail -3
```

가상 장면(기울어진 카메라, 렌즈 왜곡, 벽)에서 픽셀 → 3D 왕복, 후보 선택, 자세, 프레임 합치기를 확인한다.
