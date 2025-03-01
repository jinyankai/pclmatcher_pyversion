void publishCentroids(const std::vector<Eigen::Vector3f>& centroids, const ros::Publisher& publisher)
{
    // 创建 PCL 点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& centroid : centroids)
    {
        pcl::PointXYZ point;
        point.x = centroid.x();
        point.y = centroid.y();
        point.z = centroid.z();
        cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    // 转换为 ROS 消息
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "rm_frame";  // 设定坐标系
    cloud_msg.header.stamp = ros::Time::now();

    // 发布点云
    publisher.publish(cloud_msg);
    std::cout << "Published centroids array." << std::endl;
}
