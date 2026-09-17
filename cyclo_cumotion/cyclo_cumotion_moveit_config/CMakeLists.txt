################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.22)
project(cyclo_cumotion_moveit_config)

################################################################################
# Find and load build settings from external packages
################################################################################
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_kinematics REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_eigen REQUIRED)

find_library(MOVEIT_KDL_KINEMATICS_LIBRARY
  NAMES moveit_kdl_kinematics_plugin
  PATHS "${moveit_kinematics_DIR}/../../../lib"
  REQUIRED)

add_library(cyclo_goal_state_kdl_kinematics SHARED
  src/goal_state_kdl_kinematics_plugin.cpp)
target_compile_features(cyclo_goal_state_kdl_kinematics PUBLIC cxx_std_17)
ament_target_dependencies(cyclo_goal_state_kdl_kinematics
  geometry_msgs moveit_core moveit_kinematics moveit_msgs pluginlib rclcpp)
target_link_libraries(cyclo_goal_state_kdl_kinematics
  "${MOVEIT_KDL_KINEMATICS_LIBRARY}")
pluginlib_export_plugin_description_file(
  moveit_core goal_state_kdl_kinematics_plugin.xml)

add_library(cyclo_wholebody_goal_kinematics SHARED
  src/wholebody_goal_kinematics_plugin.cpp)
target_compile_features(cyclo_wholebody_goal_kinematics PUBLIC cxx_std_17)
ament_target_dependencies(cyclo_wholebody_goal_kinematics
  geometry_msgs moveit_core moveit_msgs pluginlib rclcpp tf2_eigen)
pluginlib_export_plugin_description_file(
  moveit_core wholebody_goal_kinematics_plugin.xml)

################################################################################
# Install
################################################################################
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(TARGETS cyclo_goal_state_kdl_kinematics cyclo_wholebody_goal_kinematics
  LIBRARY DESTINATION lib)
install(PROGRAMS
  scripts/cumotion_group_router.py
  scripts/joint_state_sanitizer.py
  DESTINATION lib/${PROJECT_NAME})

################################################################################
# Macro for ament package
################################################################################
ament_package()
