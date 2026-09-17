cmake_minimum_required(VERSION 3.22)
project(cyclo_cumotion_nvblox)

find_package(ament_cmake REQUIRED)
install(PROGRAMS
  scripts/tsdf_cube_marker.py
  scripts/wait_for_esdf.py
  DESTINATION lib/${PROJECT_NAME})
install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

ament_package()
