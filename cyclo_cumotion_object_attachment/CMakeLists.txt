cmake_minimum_required(VERSION 3.22)
project(cyclo_cumotion_object_attachment)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/predefined_object_attachment.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
