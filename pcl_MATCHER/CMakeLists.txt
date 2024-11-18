cmake_minimum_required(VERSION 3.0.2)
project(pcl_matcher)

## 设置 C++ 标准（如果有 C++ 文件可选）
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## 找到 ROS 依赖
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  sensor_msgs
  std_msgs
  pcl_ros
  pcl_conversions
  tf
  visualization_msgs
)

## 如果需要使用 Open3D，您可以在此引入外部库
# 例如，您可以设置 Open3D 的路径（假设 Open3D 已安装）
# find_package(Open3D REQUIRED)

## 声明 catkin 包
catkin_package(
  CATKIN_DEPENDS rospy roscpp sensor_msgs std_msgs pcl_ros pcl_conversions tf visualization_msgs
)

## 包含目录路径
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Python 脚本路径配置
catkin_python_setup()

## 如果有 C++ 节点（以下为示例，可以去掉或保留）
# add_executable(pcl_matcher_node src/main.cpp src/pcl_matcher.cpp)
# target_link_libraries(pcl_matcher_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
