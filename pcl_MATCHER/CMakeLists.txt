cmake_minimum_required(VERSION 3.0.2)
project(pcl_matcher)

## 声明该包依赖哪些 ROS 包
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

## 声明 catkin 包
catkin_package()

## 指定 Python 脚本路径
catkin_python_setup()
