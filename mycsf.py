
    // 转换为 ROS 消息
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";  // 设定坐标系
    cloud_msg.header.stamp = ros::Time::now();

    // 发布点云
    pub.publish(cloud_msg);
