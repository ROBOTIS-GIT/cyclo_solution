cmake_minimum_required(VERSION 3.22.1)
project(cyclo_cumotion_robot_segmenter LANGUAGES CUDA CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Resolve the proprietary cuMotion library distributed with Isaac ROS.
ament_index_get_resource(CUMOTION_RELATIVE_PATH cumotion isaac_ros_nitros)
ament_index_get_prefix_path(AMENT_INDEX_PREFIX_PATHS)
foreach(PREFIX_PATH IN LISTS AMENT_INDEX_PREFIX_PATHS)
  set(CUMOTION_CONFIG_FILE_IN_PREFIX
    "${PREFIX_PATH}/${CUMOTION_RELATIVE_PATH}/lib/cmake/cumotion/cumotionConfig.cmake")
  if(EXISTS "${CUMOTION_CONFIG_FILE_IN_PREFIX}")
    set(CUMOTION_CONFIG_FILE "${CUMOTION_CONFIG_FILE_IN_PREFIX}")
    break()
  endif()
endforeach()

if(NOT CUMOTION_CONFIG_FILE)
  message(FATAL_ERROR "Could not find the cuMotion CMake configuration")
endif()

get_filename_component(cumotion_DIR "${CUMOTION_CONFIG_FILE}" DIRECTORY)
find_package(cumotion REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

ament_auto_add_library(cyclo_robot_segmenter_node SHARED
  src/robot_segmenter.cpp)
target_link_libraries(cyclo_robot_segmenter_node
  Eigen3::Eigen
  cumotion::cumotion
  yaml-cpp)
set_target_properties(cyclo_robot_segmenter_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

rclcpp_components_register_nodes(
  cyclo_robot_segmenter_node
  "nvidia::isaac_ros::manipulator::RobotSegmenter")

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
