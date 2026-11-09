cmake_minimum_required(VERSION 3.16)
project(demo_1f LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ffw_vision_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ruckig REQUIRED)
find_package(rcutils REQUIRED)
find_package(ffw_description REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(sg2_base_model "${ffw_description_DIR}/../urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf")
set(sg2_demo_model "${CMAKE_CURRENT_BINARY_DIR}/sg2_extended_grippers.urdf")
add_custom_command(
  OUTPUT "${sg2_demo_model}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_sg2_gripper_model.py"
          --input "${sg2_base_model}" --output "${sg2_demo_model}"
  DEPENDS "${sg2_base_model}" scripts/generate_sg2_gripper_model.py
  VERBATIM
)
add_custom_target(demo_gripper_model ALL DEPENDS "${sg2_demo_model}")
install(FILES "${sg2_demo_model}" DESTINATION share/${PROJECT_NAME}/urdf)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "action/Handover.action"
  "action/RunDemo.action"
  "action/ApproachCup.action"
  "msg/DemoStatus.msg"
  "msg/CupObservation.msg"
  DEPENDENCIES
  std_msgs
  geometry_msgs
)
rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# Runtime owns ROS feedback, planning and controller execution.
add_library(
  demo_runtime SHARED
  src/runtime/config.cpp
  src/runtime/state/robot_context.cpp
  src/runtime/state/cup_observations.cpp
  src/runtime/state/head_cup_tracker.cpp
  src/runtime/state/joint_feedback.cpp
  src/runtime/state/transforms.cpp
  src/runtime/execution/trajectory_executor.cpp
  src/runtime/execution/path_preparation.cpp
  src/runtime/execution/moveit_goal.cpp
  src/runtime/execution/gripper.cpp
  src/runtime/execution/completion.cpp
  src/runtime/execution/trajectory_smoother.cpp
  src/runtime/geometry/tcp_roll.cpp
)
# Reusable motion modules have no action servers.
add_library(
  demo_motions SHARED
  src/motions/handover/handover_motion.cpp
  src/motions/common/sequence_motion.cpp
  src/motions/common/cup_motion.cpp
  src/motions/common/head_cup_observer.cpp
  src/motions/pick/pick_geometry.cpp
  src/motions/common/approach_config.cpp
  src/motions/common/approach_scene.cpp
  src/motions/place/approach_cup_motion.cpp
  src/motions/place/refine_cup_motion.cpp
  src/motions/place/withdraw_cup_motion.cpp
  src/motions/pick/pick_cup_motion.cpp
)
# TaskManager owns action routing, arbitration and status reporting.
add_library(demo_tasks SHARED src/tasks/task_manager.cpp src/tasks/cup_actions.cpp src/tasks/demo_actions.cpp)
foreach(target demo_runtime demo_motions demo_tasks)
  target_include_directories(
    ${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>
  )
endforeach()
ament_target_dependencies(
  demo_runtime
  rclcpp
  rcutils
  rclcpp_action
  control_msgs
  geometry_msgs
  moveit_msgs
  moveit_core
  sensor_msgs
  shape_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  trajectory_msgs
)
target_link_libraries(demo_runtime "${typesupport_target}" ruckig::ruckig)
target_link_libraries(demo_motions PUBLIC demo_runtime)
target_link_libraries(demo_tasks PUBLIC demo_motions)
ament_target_dependencies(demo_tasks PUBLIC ffw_vision_msgs)
add_library(demo_start_state_adapter SHARED src/moveit_plugins/snap_start_state_bounds.cpp)
target_include_directories(
  demo_start_state_adapter PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(demo_start_state_adapter moveit_core pluginlib rclcpp)
pluginlib_export_plugin_description_file(moveit_core planning_request_adapters.xml)
add_library(demo_roll_sampler SHARED src/moveit_plugins/roll_constraint_sampler.cpp)
ament_target_dependencies(demo_roll_sampler moveit_core pluginlib rclcpp)
pluginlib_export_plugin_description_file(moveit_core constraint_samplers.xml)

add_library(cup_detector src/perception/ai_cup_detector.cpp)
# Jetson uses TensorRT 10; ordinary workstations can use OpenCV CPU inference.
find_path(TENSORRT_INCLUDE NvInfer.h PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer)
find_package(CUDAToolkit QUIET)
if(TENSORRT_INCLUDE
   AND TENSORRT_LIBRARY
   AND CUDAToolkit_FOUND
)
  file(STRINGS "${TENSORRT_INCLUDE}/NvInferVersion.h" TRT_MAJOR_LINE
       REGEX "^#define NV_TENSORRT_MAJOR "
  )
  if(TRT_MAJOR_LINE MATCHES "NV_TENSORRT_MAJOR 10")
    target_compile_definitions(cup_detector PRIVATE DEMO_TENSORRT)
    target_include_directories(cup_detector PRIVATE ${TENSORRT_INCLUDE})
    target_link_libraries(cup_detector ${TENSORRT_LIBRARY} CUDA::cudart)
    # Jetson driver libraries are injected by the NVIDIA runtime, absent during
    # an ordinary Docker build. Direct executable symbols remain link-checked.
    target_link_options(cup_detector INTERFACE "-Wl,--allow-shlib-undefined")
  endif()
endif()
target_include_directories(
  cup_detector PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include> ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(cup_detector ${OpenCV_LIBS})
add_executable(perception_node src/perception/perception_node.cpp)
ament_target_dependencies(
  perception_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  tf2
  ament_index_cpp
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(perception_node cup_detector "${typesupport_target}")

add_executable(handover_demo_node src/nodes/main.cpp)
target_link_libraries(handover_demo_node demo_tasks)
add_executable(check_feedback src/nodes/check_feedback.cpp)
target_link_libraries(check_feedback demo_runtime)
install(
  TARGETS demo_runtime demo_motions demo_tasks cup_detector
  EXPORT export_demo_modules
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(
  TARGETS handover_demo_node check_feedback perception_node demo_start_state_adapter
          demo_roll_sampler
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch models DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/perception_viewer.py scripts/action_runner.py scripts/wrist_perception.py
        DESTINATION lib/${PROJECT_NAME}
)

install(FILES scripts/wrist_cup_geometry.py DESTINATION lib/${PROJECT_NAME})
install(
  DIRECTORY scripts/demo_perception
  DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING
  PATTERN "*.py"
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  add_test(NAME test_gripper_model COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test/test_gripper_model.py")
  ament_add_gtest(test_demo_task test/test_demo_task.cpp TIMEOUT 60)
  target_link_libraries(test_demo_task demo_tasks)
  set_tests_properties(test_demo_task PROPERTIES ENVIRONMENT "RMW_IMPLEMENTATION=rmw_fastrtps_cpp;ROS_DOMAIN_ID=151;ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")
  ament_add_gtest(test_ai_cup test/test_ai_cup.cpp TIMEOUT 30)
  target_link_libraries(test_ai_cup cup_detector)
  ament_add_gtest(test_cup test/test_cup.cpp test/test_cup_tf.cpp test/test_tcp_roll.cpp TIMEOUT 60)
  target_link_libraries(test_cup demo_tasks cup_detector)
  ament_target_dependencies(test_cup pluginlib)
  set_tests_properties(test_cup PROPERTIES ENVIRONMENT "RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
  ament_add_gtest(test_handover test/test_handover.cpp TIMEOUT 90)
  target_link_libraries(test_handover demo_tasks)
  ament_add_gtest(test_start_state_bounds test/test_start_state_bounds.cpp TIMEOUT 30)
  target_link_libraries(test_start_state_bounds demo_start_state_adapter)
  ament_add_gtest(test_trajectory_smoother test/test_trajectory_smoother.cpp TIMEOUT 90)
  target_link_libraries(test_trajectory_smoother demo_runtime)
  ament_target_dependencies(test_start_state_bounds moveit_core rclcpp)
  # This is an offline RobotModel test. Avoid the base image's Zenoh logger
  # teardown bug; the ROS controller integration test still uses Zenoh.
  set_tests_properties(
    test_start_state_bounds PROPERTIES ENVIRONMENT "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
  )
  set_tests_properties(
    test_trajectory_smoother PROPERTIES ENVIRONMENT "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
  )
  add_executable(validate_sg2_targets test/validate_sg2_targets.cpp)
  target_link_libraries(validate_sg2_targets demo_runtime)
  ament_target_dependencies(validate_sg2_targets moveit_core)
endif()
ament_export_include_directories(include)
ament_export_targets(export_demo_modules HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ffw_vision_msgs
  rosidl_default_runtime
  rclcpp
  rcutils
  rclcpp_action
  control_msgs
  geometry_msgs
  moveit_msgs
  moveit_core
  sensor_msgs
  shape_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  trajectory_msgs
  ruckig
  OpenCV
)
ament_package()
