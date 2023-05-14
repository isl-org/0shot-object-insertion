cmake_minimum_required(VERSION 3.0.2)
project(softgrasp_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  cv_bridge
  franka_control
  franka_msgs
  genmsg
  geometry_msgs
  gpd_ros
  image_geometry
  image_transport
  message_generation
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_visual_tools
  roscpp
  rviz_visual_tools
  softgrasp_ros_controllers
  std_msgs
  tf2
  tf2_ros
)

## System dependencies are found with CMake's conventions
find_package(Open3D REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(OpenCV REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

# Generate services in the 'srv' folder
add_service_files(
  FILES
  SetGraspState.srv
  SetGraspParams.srv
  PointCloudPerception.srv
)

# Generate actions in the 'action' folder
add_action_files(
  FILES
  MoveToGoalVicinity.action
  RaiseArm.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  gpd_ros
  std_msgs  # Or other packages containing msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

# # Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES softgrasp
  CATKIN_DEPENDS
    cv_bridge
    franka_control
    franka_msgs
    geometry_msgs
    gpd_ros
    image_geometry
    image_transport
    message_runtime
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_visual_tools
    roscpp
    softgrasp_ros_controllers
    std_msgs
    tf2
    tf2_ros
  DEPENDS
    Open3D
    Boost
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  include/${PROJECT_NAME}/geometry_utils.h
  src/geometry_utils.cpp
  include/${PROJECT_NAME}/visualization_utils.h
  src/visualization_utils.cpp
)

add_library(${PROJECT_NAME}_control
  include/${PROJECT_NAME}/arm_position_control.h
  src/arm_position_control.cpp
  include/${PROJECT_NAME}/arm_force_control.h
  src/arm_force_control.cpp
  include/${PROJECT_NAME}/hand_control.h
  src/hand_control.cpp
)

# Add cmake target dependencies of the library
# as an example, code may need to be generated before libraries
# either from message generation or dynamic reconfigure
add_dependencies(${PROJECT_NAME}_control
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_pointcloud_snapshot
  nodes/pointcloud_snapshot.cpp
)
add_executable(${PROJECT_NAME}_perception_server
  include/${PROJECT_NAME}/perception_server.h
  src/perception_server.cpp
  nodes/perception_server_node.cpp
)
add_executable(${PROJECT_NAME}_force_control_grasp
  nodes/force_control_grasp.cpp
)
add_executable(${PROJECT_NAME}_test_arm_force_control
  nodes/test_arm_force_control.cpp
)
add_executable(${PROJECT_NAME}_test_arm_position_control
  nodes/test_arm_position_control.cpp
)
add_executable(${PROJECT_NAME}_test_hand_control
  nodes/test_hand_control.cpp
)
add_executable(${PROJECT_NAME}_marker_pose
  src/marker_pose.cpp
)
add_executable(${PROJECT_NAME}_test_transforms
  src/test_transforms.cpp
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties(${PROJECT_NAME}_pointcloud_snapshot PROPERTIES
  OUTPUT_NAME pointcloud_snapshot PREFIX ""
)
set_target_properties(${PROJECT_NAME}_perception_server PROPERTIES
  OUTPUT_NAME perception_server PREFIX ""
)
set_target_properties(${PROJECT_NAME}_force_control_grasp PROPERTIES
  OUTPUT_NAME force_control_grasp PREFIX ""
)
set_target_properties(${PROJECT_NAME}_test_arm_force_control PROPERTIES
  OUTPUT_NAME test_arm_force_control PREFIX ""
)
set_target_properties(${PROJECT_NAME}_test_arm_position_control PROPERTIES
  OUTPUT_NAME test_arm_position_control PREFIX ""
)
set_target_properties(${PROJECT_NAME}_test_hand_control PROPERTIES
  OUTPUT_NAME test_hand_control PREFIX ""
)
set_target_properties(${PROJECT_NAME}_marker_pose PROPERTIES
  OUTPUT_NAME marker_pose PREFIX ""
)
set_target_properties(${PROJECT_NAME}_test_transforms PROPERTIES
  OUTPUT_NAME test_transforms PREFIX ""
)

# Add cmake target dependencies of the executable
# same as for the library above
add_dependencies(${PROJECT_NAME}_force_control_grasp
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
add_dependencies(${PROJECT_NAME}_test_arm_force_control
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
add_dependencies(${PROJECT_NAME}_test_arm_position_control
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
add_dependencies(${PROJECT_NAME}_test_hand_control
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${Open3D_LIBRARIES}
  ${catkin_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_control
  ${catkin_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_pointcloud_snapshot
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}
)
target_link_libraries(${PROJECT_NAME}_perception_server
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)
target_link_libraries(${PROJECT_NAME}_force_control_grasp
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${PROJECT_NAME}_control
)
target_link_libraries(${PROJECT_NAME}_test_arm_force_control
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_control
)
target_link_libraries(${PROJECT_NAME}_test_arm_position_control
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_control
)
target_link_libraries(${PROJECT_NAME}_test_hand_control
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_control
)
target_link_libraries(${PROJECT_NAME}_marker_pose
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
target_link_libraries(${PROJECT_NAME}_test_transforms
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_softgrasp_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
