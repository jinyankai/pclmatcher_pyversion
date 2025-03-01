
void publishCentroids(const std::vector<Eigen::Vector3f>& centroids, const ros::Publisher& publisher)
{
    
    // 转换为 ROS 消息
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(centroids, cloud_msg);
    cloud_msg.header.frame_id = "rm_frame";  // 设定坐标系
    cloud_msg.header.stamp = ros::Time::now();

    // 发布点云
    publisher.publish(cloud_msg);
    std::cout << "publish centroids array" << std::endl;
}
