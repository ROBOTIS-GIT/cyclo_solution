cmake_minimum_required(VERSION 3.5)

project(cyclo_cumotion_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

ament_package()
