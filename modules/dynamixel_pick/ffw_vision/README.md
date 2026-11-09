# ffw_vision

AI Worker 카메라 영상에서 **픽셀 관측**(원본 영상 기준 u, v와 그 영상의 header)을 만들어 발행하는 패키지다.
3차원 좌표 계산, 파지, 모션은 이 패키지의 범위가 아니다. 그 작업은 후속 좌표계 노드가 맡는다.

- `ffw_vision_msgs`: 관측 메시지 정의 (`PixelObservation`)
- `ffw_vision`: Vision 노드 (수동 선택, 자동 검출)
- 좌표·파지(픽셀 → base_link, 파지 자세)는 `ffw_grasp` 패키지

## 관측 메시지 `ffw_vision_msgs/msg/PixelObservation`

| 필드 | 의미 |
|---|---|
| `header` | 사용한 원본 영상의 header 를 그대로 복사한다. `stamp` 는 촬영 시각, `frame_id` 는 원본 영상의 광학 좌표계다. 수신·발행 시각으로 바꾸지 않는다. |
| `camera_id` | `head`, `left_wrist`, `right_wrist` 중 하나 (설정 `cameras.yaml` 의 키) |
| `image_topic` | 좌표의 기준이 된 영상 토픽. raw 인지 rectified 인지를 구분하는 데 쓴다. |
| `image_width`, `image_height` | 원본 영상 크기(픽셀) |
| `target_label` | 대상 종류 또는 요청된 대상. 수동 선택에서는 비어 있을 수 있다. |
| `point_type` | (u, v) 의 의미. 수동 선택은 `selected_point` 다. |
| `source_mode` | `SOURCE_MANUAL`(1) 수동 선택, `SOURCE_AUTO`(2) 자동 검출 |
| `status` | `STATUS_VALID`(1), `STATUS_NOT_DETECTED`(2), `STATUS_AMBIGUOUS`(3) |
| `u`, `v` | 원본 영상 기준 픽셀 좌표 |

### 좌표 규칙

- u 는 열(오른쪽이 +), v 는 행(아래쪽이 +)이다.
- 정수 (u, v) 는 그 픽셀의 **중심**이다. `sensor_msgs/CameraInfo` 의 K 행렬과 같은 기준이다.
  유효 범위는 `-0.5 <= u < image_width - 0.5`, `-0.5 <= v < image_height - 0.5` 이다.
- 화면에서 확대·축소·잘라내기를 했더라도 발행하는 값은 항상 원본 영상 좌표로 환산한 값이다.

### 수신 측 사용 규칙

1. **`status` 를 먼저 확인한다.** `STATUS_VALID` 일 때만 u, v 를 사용한다.
2. `STATUS_VALID` 가 아니면 u, v 는 NaN 이다. 미검출을 (0, 0) 으로 표현하지 않는다.
3. `status` 또는 `source_mode` 가 0(`UNSET`)이면 값을 채우지 않은 메시지이므로 버린다.
4. 3차원 변환에는 `header.stamp` 시각의 TF 와, `header.frame_id`·`image_topic`·크기가 일치하는 camera_info 를 쓴다.
5. `point_type == "selected_point"` 는 영상에서 고른 지점일 뿐이다. 물체 중심이나 파지점으로 해석하지 않는다.
6. 이 지점의 depth 는 보이는 표면까지의 거리다. 물체 중심이나 파지점이 필요하면 대상 형상에 맞는 처리를 따로 한다.

### 노드 상태

입력 끊김 같은 노드 상태는 관측 메시지에 넣지 않는다. 별도 토픽으로 발행한다(4단계에서 정의).

### 확장 예정 (5단계)

회전, 검출 점수, 다중 검출(배열 메시지)은 검출기와 실제 출력이 정해진 뒤
의미와 유효성 표현(값이 있는지 나타내는 표시)을 정의하고 추가한다.

## 물체 관측 메시지 `ffw_vision_msgs/msg/ObjectObservationArray`

자동 검출기가 한 프레임에서 찾은 물체 전체를 담는다. 검출 방식(규칙 기반, 학습 기반)과 무관하게 같은 형식이다.
발행 토픽: `/ffw_vision/<camera_id>/object_observations` (`object_detector`).

| 필드 | 의미 |
|---|---|
| `header` | 사용한 원본 영상의 header 그대로 (촬영 시각, 광학 frame_id) |
| `camera_id`, `image_topic`, `image_width`, `image_height` | `PixelObservation` 과 같은 의미 |
| `source_mode` | `SOURCE_AUTO`(2) 자동 검출, `SOURCE_MANUAL`(1) 사람이 지정 |
| `status` | `STATUS_VALID`(1) 물체 1개 이상, `STATUS_NOT_DETECTED`(2) 처리했지만 물체 없음(`objects` 비어 있음) |
| `has_support_plane`, `support_plane` | 물체가 놓인 면(탁자), `header.frame_id` 기준 `[nx, ny, nz, d]`. `n · p + d` = 면 위 높이, 단위 법선은 카메라 쪽. 측정한 검출기만 채운다 |
| `objects[]` | 물체마다 `ObjectObservation` |

`ObjectObservation`:

| 필드 | 의미 |
|---|---|
| `target_label` | 대상 종류 (예: `dynamixel`) |
| `pose_label` | task 가 정한 자세 라벨 (예: `lying`, `standing`, `unknown`) |
| `status` | `STATUS_VALID`(1), `STATUS_AMBIGUOUS`(3) 붙어 있어 분리 확신 없음 등 |
| `u`, `v` | 회전 상자 중심, 원본 픽셀 (정수 = 픽셀 중심) |
| `length`, `width` | 긴 변, 짧은 변 (픽셀) |
| `angle` | 긴 변 방향. 영상 +u 축에서 +v 쪽으로 잰 각(rad), `[-π/2, π/2)` |
| `angle_period` | 같은 모양이 되풀이되는 회전 주기. `π` 직사각형, `π/2` 정사각형에 가까움, `0` 방향 없음 |
| `has_score`, `score` | 검출 신뢰도. 위치·각도 정확도가 아니다 |
| `top_height` | 윗면이 `support_plane` 위로 얼마나 높은지(m). 측정 안 했으면 NaN |

### 수신 측(좌표계 노드) 사용 규칙

1. 배열 `status` 를 먼저 본다. `STATUS_NOT_DETECTED` 면 물체가 없다는 뜻이다(입력 끊김은 `status` 토픽에서 따로 확인).
2. 물체 `status` 가 `STATUS_AMBIGUOUS` 면 그대로 집지 말고 다시 관측하거나 사람이 판단한다.
3. 3D 위치: `header.stamp` 시각의 TF, 같은 `image_topic` 의 camera_info, 정렬된 depth 를 쓴다.
   depth 는 상자 가장자리를 조금 줄인 안쪽 영역의 중앙값을 쓰면 윗면 거리가 안정적이다.
4. 회전: `angle` 은 영상 기준이다. 그리퍼 회전으로 바꿀 때는 TF 로 영상 축을 로봇 축으로 옮기고 `angle_period` 로 가까운 쪽을 고른다.
   (오른손 관측 자세에서는 영상 +u ≈ 로봇 +x, 영상 +v ≈ 로봇 −y)

## 카메라 설정 `config/cameras.yaml`

카메라마다 영상 토픽, raw/rectified 구분, camera_info 토픽, 정렬된 depth 토픽, frame_id 를 적는다.
빈 문자열은 "아직 확인하지 않음"을 뜻한다. 로봇 출력으로 확인한 값만 채운다.

### head (ZED Mini, 2026-10-23 로봇 출력으로 확인)

| 항목 | 값 |
|---|---|
| 원본 영상 | `/zed/zed_node/left/image_rect_color` (왼쪽 rectified, 672×376, bgra8) |
| camera_info | `/zed/zed_node/left/camera_info` (plumb_bob, D = 0) |
| frame_id | `zed_left_camera_optical_frame` |
| depth | `/zed/zed_node/depth/depth_registered` (32FC1, 미터). 아래 depth 설정으로 실행했을 때만 있음 |

- raw 영상과 rectified 영상의 frame_id 가 같다. 스트림 구분은 `image_topic` 으로 한다.
- 좌표계 노드는 rectified 영상에 맞는 camera_info 의 투영 파라미터(P, D = 0)를 쓴다. raw camera_info(equidistant 왜곡)를 섞어 쓰지 않는다.
- depth 를 연결할 때는 영상 정합, 촬영 시각, 단위를 확인한 뒤 사용한다.

### 손목 D405 (2026-10-26 로봇 출력으로 확인)

| 항목 | left_wrist | right_wrist |
|---|---|---|
| 원본 영상 | `/camera_left/camera_left/color/image_rect_raw` | `/camera_right/camera_right/color/image_rect_raw` |
| 크기·인코딩 | 424×240, rgb8 | 424×240, rgb8 |
| camera_info | `…/color/camera_info` (왜곡 계수 있음 → `stream_type: raw`) | 같음 |
| 정렬된 depth | `…/aligned_depth_to_color/image_raw` (16UC1, 컬러와 같은 크기·frame_id·K) | 같음 |
| frame_id | `camera_left_color_optical_frame` | `camera_right_color_optical_frame` |

- 정렬된 depth 는 아래 launch 로 실행할 때만 있다. bringup 기본값은 colorizer 켬, align 꺼짐이라 거리값이 나오지 않는다.
  ```bash
  ros2 launch ffw_vision wrist_cameras.launch.py
  ```
- 좌우는 `ffw_bringup/config/common/rs_serial.yaml` 의 시리얼로 정해진다. camera1 = 로봇 기준 왼손, camera2 = 오른손.
- 두 카메라 모두 USB 2.1 로 연결되어 있다(해상도·프레임 상한이 낮음).
- 16UC1 값의 단위는 실제 거리와 비교해 확인한다(`depth_probe` 의 `depth_scale`).

### ZED depth 설정 `config/zed_depth_override.yaml`

bringup 의 ZED 설정(`depth_mode: NONE`)은 바꾸지 않는다. 카메라만 따로 실행할 때 이 파일로 덮어쓴다.

```bash
ros2 launch ffw_bringup camera_zed.launch.py camera_model:=zedm \
  ros_params_override_path:=$(ros2 pkg prefix ffw_vision)/share/ffw_vision/config/zed_depth_override.yaml
```

- `depth_mode: ULTRA`. NEURAL 계열은 이 컨테이너에 TensorRT 가 없어서 쓸 수 없다.
- `depth_stabilization: 0` 이라 위치 추적이 켜지지 않는다.

## 입력 확인 노드 `input_monitor`

설정한 카메라의 영상, camera_info, depth(설정된 경우)를 구독해서 주기적으로 한 줄씩 로그를 남긴다. 아무것도 발행하지 않는다.
영상을 받은 뒤 100 ms 가 지나면, 같은 stamp 의 camera_info 와 depth 가 있는지 판정한다.

```bash
ros2 run ffw_vision input_monitor --ros-args \
  --params-file $(ros2 pkg prefix ffw_vision)/share/ffw_vision/config/cameras.yaml
```

| 상태 | 의미 |
|---|---|
| `OK` | 영상이 들어오고 frame_id, 크기, camera_info 짝이 모두 맞음 |
| `NO_INPUT` | `timeout_sec` 동안 영상이 없음 |
| `NO_CAMERA_INFO` | 영상은 오지만 이번 구간에 camera_info 가 없음 |
| `NO_DEPTH` | depth_topic 이 설정됐지만 이번 구간에 depth 가 없음 |
| `MISMATCH` | frame_id·크기·depth 인코딩이 다르거나, 같은 stamp 의 camera_info·depth 가 없는 영상이 있음 |

## 수동 선택 화면 `web_select`

Jetson 에서 설정한 카메라 영상을 웹 페이지로 보여 주고, 클릭한 위치를 그 프레임의 원본 픽셀 (u, v) 로 바꾼다.
영상 표시와 수동 선택만 담당하는 노드다(핵심 처리와 분리). 화면 없이 동작해야 하는 부분은 이 노드에 두지 않는다.

```bash
ros2 run ffw_vision web_select --ros-args \
  --params-file $(ros2 pkg prefix ffw_vision)/share/ffw_vision/config/cameras.yaml
```

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `bind_address` | `127.0.0.1` | Jetson 밖으로 열지 않는다. PC 에서는 VS Code 포트 포워딩으로 접속 |
| `port` | 8765 | |
| `jpeg_quality` | 80 | 화면 전송용 압축 품질 (원본 좌표에는 영향 없음) |
| `retention_sec` | 120 | 일시정지한 프레임의 header 를 보관하는 시간 |
| `display_topic` | `''` | 화면에 띄울 토픽. 비우면 `image_topic`. `debug_image` 처럼 원본과 stamp 가 같은 영상이어야 클릭 선택이 맞는다 |

- 화면은 0.1 초마다 최신 프레임을 받는다. 각 프레임과 함께 stamp, frame_id, 원본 크기가 전달된다.
- 일시정지하면 화면의 프레임과 그 stamp 가 고정된다. 클릭하면 **화면에 보이는 프레임의 stamp** 로 요청한다.
- Jetson 은 받은 프레임의 header 를 stamp 로 보관하고 있다가 그 값으로 응답한다. 최신 프레임의 header 로 바꾸지 않는다.
- 좌표 환산: `u = floor(x · W / 표시 폭)`, `v = floor(y · H / 표시 높이)`. 클릭한 위치 아래 원본 픽셀을 고르며, 정수 값은 그 픽셀의 중심이다.
  화면이 축소돼 있으면 오차는 표시 픽셀 하나에 해당하는 원본 픽셀 수만큼이다. 잘라내기와 확대 표시는 하지 않는다.
- 보관 시간이 지난 프레임이나 다른 카메라의 stamp 는 거절한다(409).
- 계산한 (u, v) 와 프레임 stamp 를 `pixel_observer` 의 `select_pixel` 서비스로 보낸다. 서비스가 없으면 503 을 돌려준다.

## 핵심 노드 `pixel_observer`

화면 없이 동작한다. 설정한 카메라 영상의 header 를 stamp 로 보관하고, 선택 요청을 검증한 뒤 관측을 발행한다.
자동 검출을 붙일 때도 이 노드를 그대로 쓴다.

| 이름 (namespace `/ffw_vision/<camera_id>`) | 종류 | 내용 |
|---|---|---|
| `pixel_observation` | 발행 `ffw_vision_msgs/PixelObservation` | 선택한 프레임의 header 와 원본 (u, v) |
| `status` | 발행 `diagnostic_msgs/DiagnosticArray` (1 Hz) | 입력 상태. 관측과 분리 |
| `select_pixel` | 서비스 `ffw_vision_msgs/SelectPixel` | camera_id, 프레임 stamp, u, v 로 요청 |

- 거절 조건: camera_id 가 다름, 보관 중이 아닌 stamp(`retention_sec`, 기본 120 초), u·v 가 영상 밖이거나 유한하지 않음.
  거절하면 관측을 발행하지 않고, 응답 `message` 에 이유를 담는다.
- 상태: `level` OK(0) = 정상, WARN(1) = camera_info 없음·불일치, ERROR(2) = `NO_INPUT`(1 초 동안 영상 없음).
  `message` 에 상태 이름이 들어 있고, `values` 에 수신 주기, 마지막 영상 stamp, 오류 개수가 들어 있다.
- 입력이 끊겨도 보관 중인 프레임에 대한 선택은 받아들인다. 그 프레임은 실제로 촬영된 것이기 때문이다.

## 함께 실행

```bash
ros2 launch ffw_vision manual_select.launch.py            # camera_id:=head, port:=8765 기본
```

두 노드가 `/ffw_vision/<camera_id>` 아래에서 같은 설정으로 실행된다. 카메라마다 다른 namespace 를 쓰므로 이름이 겹치지 않는다.

## 자동 검출 `object_detector` (규칙 기반)

컬러와 정렬된 depth 를 stamp 가 같은 것끼리 짝지어 프레임마다 처리하고, 찾은 물체 전체를 `ObjectObservationArray` 로 발행한다.
depth_topic 이 없는 카메라는 컬러만 쓴다(`pose_label` 은 `unknown`).

| 이름 (namespace `/ffw_vision/<camera_id>`) | 종류 | 내용 |
|---|---|---|
| `object_observations` | 발행 `ffw_vision_msgs/ObjectObservationArray` | 원본 header 그대로, 물체가 없으면 `STATUS_NOT_DETECTED` |
| `debug_image` | 발행 `sensor_msgs/Image` (bgr8) | 결과를 그린 영상. 원본 header 그대로 |

처리 순서: ① `roi` 안에서 ② 어둡거나(`dark_v_max`) 채도가 낮고 밝지 않은(`gray_s_max`, `gray_v_max`) 픽셀을 물체로 본다.
③ 덩어리마다 **주변 바닥**(물체 가장자리에서 3~13 px 떨어진 고리, 다른 물체 제외)의 depth 중앙값을 그 물체의 바닥으로 쓴다.
고리가 50 px 보다 적으면 `roi` 전체 바닥을 쓴다. 카메라가 기울어 바닥 depth 가 위치마다 달라도 높이가 맞다.
④ 덩어리에서 가장 높은 20 % 의 높이(윗면 높이)를 구하고, 그보다 `top_tolerance` 이내로 낮은 픽셀만 **윗면**으로 골라낸다.
광축에서 먼 물체는 옆면도 보여 색 덩어리가 옆면 쪽으로 길어지므로, 회전 상자와 자세 높이(윗면 높이 중앙값)는 윗면으로 구한다.
윗면이 덩어리의 `min_top_ratio` 보다 작거나 depth 가 부족하면 색 덩어리 상자를 쓰고 자세는 `unknown` 이다.

| 판정 | 조건 |
|---|---|
| 버림 | 면적 < `min_area`, 또는 윗면 높이 < `min_height`(그림자·인쇄) |
| `STATUS_AMBIGUOUS` | 색 덩어리 면적 > `max_single_area`, 긴 변 > `max_single_length`(붙은 물체), 또는 `roi` 경계에 닿음(잘린 물체). 상자는 색 덩어리 전체, `pose_label` = `unknown`, `angle_period` = 0 |
| `standing` / `lying` | 높이 ≥ `standing_min_height` 이면 `standing`. 유효 depth 가 `min_depth_ratio` 보다 적으면 `unknown` |
| `angle_period` | 긴 변 / 짧은 변 < `square_ratio` 이면 π/2, 아니면 π |

- 화면 표시: 초록 = VALID, 주황 = AMBIGUOUS, 회색 = 색 덩어리 전체, 선 = 긴 변 방향, 글자 = 자세(L, S, ?) + 높이(mm), 아래 = 개수와 `roi` 전체 바닥 거리.
- 로그(2 초마다): 물체별 `h=윗면 높이 중앙값/윗면 높이(mm) f=주변 바닥(m) shift=색 덩어리 중심과의 거리(px) a=면적(px)`.
- `has_score` 는 false 다(규칙 기반이라 신뢰도 값이 없다).
- `pose_label`(`lying`, `standing`)은 **참고용 추정**이다. 광축에서 먼 쪽(영상 왼쪽)에서는 세움 높이가 낮게 나와 틀릴 수 있다.
  물체 높이가 필요하면 좌표계 노드가 중심점 depth 로 구한 3D 위치를 쓴다.
- 파지 방향(수신 측 규칙): 짧은 축 = `angle + π/2` 를 가로질러 잡는다. `angle_period` 가 π/2 면 거의 정사각형이라 두 축 모두 가능하다.
- task 설정은 `config/dynamixel_basket.yaml` 에 모은다(대상 이름, 영역, 기준값). `roi` 는 오른손 관측 자세 `observe_top_h35` + 리프트 -0.05 에서만 맞다.

```bash
ros2 launch ffw_vision auto_detect.launch.py \
  task_file:=$(ros2 pkg prefix ffw_vision)/share/ffw_vision/config/dynamixel_basket.yaml
ros2 topic echo /ffw_vision/right_wrist/object_observations --once
```

`object_detector`, `pixel_observer`, `web_select`(화면 = `debug_image`)가 함께 실행된다. 브라우저 클릭 선택도 그대로 쓸 수 있다.

## 자동 검출 `object_detector` (`detector: yolo_seg`)

같은 노드에서 `detector: yolo_seg` 로 바꾸면 학습한 분할 모델을 쓴다. ultralytics 는 이때만 불러온다(컨테이너에 `--no-deps` 설치, `YOLO_AUTOINSTALL=false`).
컬러·정렬 depth·`camera_info`(렌즈 왜곡 보정)가 필요하다. 계산은 모두 카메라 광학 좌표계라 TF 를 쓰지 않는다.

처리 순서: ① 가장 큰 파란 영역 = 바구니 ② 바구니 둘레 띠(탁자)의 depth 로 RANSAC 평면 → `support_plane`
③ 모델 마스크(`conf` 이상)를 `mask_erode_px` 만큼 줄이고 파란 픽셀·바구니 테두리를 뺀다(마스크가 벽으로 번지는 것 방지)
④ 가장 높은 `top_share` % 의 중앙값 = `top_height`, 그보다 `top_tolerance` 이내 점 = 윗면 ⑤ 윗면 점을 평면 위에서 회전 사각형으로 맞춤 → 중심·긴 변을 영상으로 투영해 `u`, `v`, `length`, `width`, `angle`
⑤ 방향(`angle_method: edges`, 기본): 마스크(+1 px) 안 컬러 경계선의 방향을 탁자 평면(윗면 높이) 위 방향으로 바꿔 90° 로 접은 히스토그램(세기 가중)의 최댓값.
상자의 수평 모서리(윗면 테두리, 옆면 아래 모서리, 직선 무늬)는 모두 상자 축과 평행하다. 긴 변 = 윗면 점을 그 두 축으로 쟀을 때 긴 쪽.
경계 픽셀 < `edge_min_px` 이거나 최댓값 ±5° 에 모인 비율 < `edge_min_share` 면 윗면 점의 회전 사각형(`min_area_rect`). 로그에 ` rect` 로 표시
(2026-10-30 녹화 75 장, 1 개 물체 41 개, 실제 YOLO 마스크: 방향 오차 중앙값 6.2° → 2.0°, 10° 넘게 틀림 15 → 3. 비율 기준을 높여 대체를 늘리면 오히려 나빠짐 → 0.15 = 거의 고르게 퍼진 경우만)

| 판정 | 조건 |
|---|---|
| `STATUS_AMBIGUOUS` | 윗면 긴 변 > `max_single_length` 또는 짧은 변 > `max_single_width`(두 개를 한 마스크로 본 것 등). `pose_label` = `unknown` |
| `standing` / `lying` | `top_height` ≥ `standing_min_height` 이면 `standing` |
| `angle_period` | 윗면(mm) 긴 변 / 짧은 변 < `square_ratio` 이면 π/2 |

- 수신 측: `u`, `v` 의 광선과 `support_plane` 을 `top_height` 만큼 올린 면의 교점 = 윗면 중심 3D. `angle` 방향의 가까운 픽셀도 같은 방법으로 → 긴 변 방향. 크기(m)는 양 끝 투영 거리가 `length`, `width` 가 되는 값 (`ffw_grasp` 의 `top_face`).
- 마스크를 줄이므로 윗면 크기(mm)는 실제보다 2~3 mm 작게 나온다. 중심은 영향이 거의 없다.
- `length`, `width` 는 윗면(mm)의 긴 변·짧은 변 **양 끝을 영상에 투영한 두 점 사이 거리**다 (기울어 보면 가까운 쪽 절반이 더 길어 보여서 한쪽만 재지 않음). 기울어 보면 `length < width` 일 수 있다. `angle` 은 실제 긴 변 방향이다.
- 화면: 회색 = 모델 마스크, 초록/주황 = 윗면, 글자 = 자세 + 높이(mm) + 점수.
- task 설정: `config/dynamixel_basket_seg.yaml` (초기 자세 3차, `auto_detect.launch.py` 의 기본 `task_file`).
  바구니 색(HSV)은 `config/blue_basket.yaml`(`basket_file`) — `ffw_grasp` 의 바구니 맞추기와 같은 파일.
- 가중치: launch 인자 `model_path`. 기본은 환경 변수 `FFW_VISION_MODEL`, 없으면 이 패키지의 `models/dynamixel_seg_n640.pt`(설치 후 share/ffw_vision/models, 2026-10-29 학습).
- 학습 도구(`tools/`, 설치 안 함, `ultralytics` 필요):
  `labelme_draft.py`(규칙 기반 초안) → labelme 로 다듬기 → `labelme_to_yolo.py`(YOLO-seg 데이터셋) → `train_seg.py`(yolo11n-seg, 448, 150 epoch) → `eval_seg.py`(라벨과 비교).
  `extract_frames.py`: 녹화(bag)에서 컬러·16비트 depth·CameraInfo·TF·관절 값을 프레임별로 저장(라벨 초안과 오프라인 분석의 입력). 데이터셋(이미지 약 50 장)은 repo 밖 `/workspace/vision_ws/datasets/dynamixel_seg`.

```bash
ros2 launch ffw_vision auto_detect.launch.py                       # camera_id:=right_wrist, port:=8765 기본
ros2 launch ffw_vision auto_detect.launch.py model_path:=/path/to/best.pt
```

## depth 확인 도구 `depth_probe`

영상과 depth 를 stamp 가 같은 것끼리 `frames` 개 모은 뒤, 영역 `roi` 의 유효 depth 비율과 거리 통계를 출력하고 PNG 를 저장한 다음 종료한다.

```bash
ros2 run ffw_vision depth_probe --ros-args \
  --params-file $(ros2 pkg prefix ffw_vision)/share/ffw_vision/config/cameras.yaml \
  -p roi:="[0, 0, 0, 0]" -p frames:=30
```

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `roi` | `[0, 0, 0, 0]` | `[u_min, v_min, u_max, v_max]` 원본 픽셀, max 는 포함 안 함. 모두 0 이면 전체 영상 |
| `frames` | 30 | 모을 짝 개수 |
| `min_depth`, `max_depth` | 0.1, 10.0 | 유효 범위(m). zedm.yaml 과 같음 |
| `depth_scale` | 0.001 | 16UC1 한 단위의 미터 값. 0 은 무효로 처리 (32FC1 은 미터 그대로) |
| `output_dir` | `/workspace/vision_ws/captures` | PNG 저장 위치 |
| `timeout_sec` | 10.0 | 이 시간 안에 다 모으지 못하면 실패로 종료 |

- 유효 depth: 유한하고 `min_depth` 이상 `max_depth` 이하인 값.
- 출력: `valid_ratio`(전체), 프레임별 최소·최대 비율, 중앙값, p10, p90, `median_std`(프레임별 중앙값의 흔들림).
- PNG: `<stamp>_<camera>_color.png`(컬러), `_depth.png`(depth 색 표시, 검정 = 무효), `_overlay.png`(둘을 반씩 겹친 것).
  모두 50 px 격자와 좌표 숫자, roi 사각형이 그려져 있다.

## 빌드 (ai_worker 컨테이너 안)

`./docker/container.sh enter` 로 컨테이너에 들어간 뒤 실행한다.
기존 underlay(`/root/ros2_ws/install`)는 컨테이너 bashrc 가 자동으로 source 한다.

```bash
cd /workspace/vision_ws
colcon build --symlink-install --base-paths \
  /root/ros2_ws/src/ai_worker/ffw_vision_msgs \
  /root/ros2_ws/src/ai_worker/ffw_vision
source /workspace/vision_ws/install/setup.bash
```

- 빌드 결과는 `/workspace/vision_ws/{build,install,log}` 에만 생긴다.
- 컨테이너 안에서 `cb`(`/root/ros2_ws` 전체 빌드)는 쓰지 않는다. 이 패키지들이 underlay 에 섞인다.

데모 통합 기본값은 `dynamixel_basket_hybrid.yaml`이며 `dynamixel_obb_n640.pt`도 함께 사용합니다.
