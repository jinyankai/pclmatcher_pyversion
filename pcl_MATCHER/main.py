# utf-8
import rospy
import open3d as o3d
import time
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from sensor_msgs.msg import Point_Cloud2 as pc2
from sensor_msgs.point_cloud2 import read_points
from geometry_msgs.msg import Point
import threading
from fasteuclideancluster import FastEuclideanCluster


class PCLMatcher:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("pcl_matcher", anonymous=True)
        #fasteuclideancluster.py
        self.fec = None
        self.centroids = []
        self.isLevel = True

        # 初始化属性
        self.field_cloud = o3d.geometry.PointCloud()
        self.filtered_cloud = o3d.geometry.PointCloud()
        self.readyICP_cloud = o3d.geometry.PointCloud()
        self.ground_readyICP = o3d.geometry.PointCloud()
        self.ground_field = o3d.geometry.PointCloud()
        self.icp_cloud = o3d.geometry.PointCloud()
        self.cumulative_transform = np.eye(4)

        # 定义 Y 轴的旋转和平移变换矩阵
        theta = -np.pi / 12  # 假设传感器倾斜角度
        level_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0, theta, 0])
        self.level_transform = np.eye(4)
        self.level_transform[:3, :3] = level_rotation
        self.level_transform[:3, 3] = [0.0, 0.0, -4.1]  # Z 轴平移

        # ROS 订阅与发布
        self.cloud_sub = rospy.Subscriber("/livox/lidar", PointCloud2, self.cloud_callback, queue_size=10)
        self.field_pub = rospy.Publisher("field_cloud", PointCloud2, queue_size=1)
        self.adjusted_pub = rospy.Publisher("adjusted_cloud", PointCloud2, queue_size=10)
        self.icp_adjusted_pub = rospy.Publisher("icpadjusted_cloud", PointCloud2, queue_size=10)
        self.obstacle_cloud_pub = rospy.Publisher("obstacle_cloud", PointCloud2, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        # self.initial_pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_callback,
        #                                         queue_size=1)
        # self.clickpoint_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_point_callback,
        #                                      queue_size=1)

        # 启动线程
        self.field_pub_thread = threading.Thread(target=self.field_cloud_publisher)
        self.field_pub_thread.start()

        self.icp_thread = threading.Thread(target=self.icp_run)
        self.icp_thread.start()

        self.cumulative_transform = np.eye(4)
        self.isreadyICP = False
        self.isICPFinish = False
        self.isINITFinish = False
        self.initial_pose = np.eye(4)

        self.max_icp_iterations = 10
        self.fitness_score_threshold = 0.01

        self.cloud_buffer = []
        self.map_feature_points = []
        self.sensor_feature_points = []

        self.lock = threading.Lock()

    def __del__(self):
        """
        Destructor to ensure threads are joined safely.
        """
        rospy.loginfo("Shutting down PCLMatcher...")
        self.running = False
        if self.field_pub_thread.is_alive():
            self.field_pub_thread.join()  # 等待线程完成
        if self.icp_thread.is_alive():
            self.icp_thread.join()  # 等待线程完成
        rospy.loginfo("PCLMatcher shut down.")

    def open3d_to_ros(self,o3d_cloud, frame_id="map"):
        """
        Converts an Open3D PointCloud to a ROS PointCloud2 message.
        :param o3d_cloud: Open3D PointCloud
        :param frame_id: ROS frame ID
        :return: ROS PointCloud2 message
        """
        # Extract points from Open3D PointCloud
        points = np.asarray(o3d_cloud.points)

        # Create PointCloud2 fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Convert points to ROS PointCloud2
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        ros_cloud = pc2.create_cloud(header, fields, points)
        return ros_cloud

    def ros_to_open3d(self,ros_cloud):
        """
        Converts a ROS PointCloud2 message to an Open3D PointCloud.
        :param ros_cloud: ROS PointCloud2 message
        :return: Open3D PointCloud
        """
        # Extract points from PointCloud2 message
        points = []
        for point in read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        # Create Open3D PointCloud object
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        return o3d_cloud

    def load_pcd(self, file_path):
        """
        Loads a PCD file, applies voxel grid filtering, and converts to ROS PointCloud2.
        :param file_path: Path to the PCD file
        """
        try:
            # 加载点云
            self.field_cloud = o3d.io.read_point_cloud(file_path)
            rospy.loginfo(f"Loaded {len(self.field_cloud.points)} data points from {file_path}")

            # 应用体素滤波
            self.field_cloud = self.field_cloud.voxel_down_sample(voxel_size=0.1)

            # 转换为 ROS PointCloud2 消息
            self.ros_field_cloud = self.open3d_to_ros(self.field_cloud)
            rospy.loginfo(f"Point cloud filtered and converted to ROS PointCloud2 format.")
        except Exception as e:
            rospy.logerr(f"Couldn't read file {file_path}: {e}")



    def field_cloud_publisher(self):
        """
        Periodically publishes the point cloud to a ROS topic.
        """
        rate = rospy.Rate(10)  # 设置发布频率为10Hz
        while not rospy.is_shutdown():
            if self.ros_field_cloud is not None:
                self.ros_field_cloud.header.frame_id = "livox_frame"
                self.ros_field_cloud.header.stamp = rospy.Time.now()
                self.field_pub.publish(self.ros_field_cloud)
            rate.sleep()

    def icp_function(self, source_cloud, target_cloud, transformation_epsilon, max_correspondence_distance,
                     euclidean_fitness_epsilon, max_iterations):
        """
        Perform ICP (Iterative Closest Point) alignment between two point clouds.
        :param source_cloud: Open3D PointCloud (source)
        :param target_cloud: Open3D PointCloud (target)
        :param transformation_epsilon: Convergence criteria for transformation
        :param max_correspondence_distance: Maximum correspondence distance
        :param euclidean_fitness_epsilon: Convergence criteria for Euclidean fitness
        :param max_iterations: Maximum number of ICP iterations
        :return: aligned_cloud (Open3D PointCloud), final_transform (4x4 transformation matrix)
        """
        start_time = time.time()

        # Initialize ICP settings
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=euclidean_fitness_epsilon,
            relative_rmse=transformation_epsilon,
            max_iteration=max_iterations
        )

        # Perform ICP registration
        icp_result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            max_correspondence_distance,
            np.eye(4),  # Initial transformation
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )

        # Apply transformation to source cloud
        aligned_cloud = source_cloud.transform(icp_result.transformation)

        # Cumulative transformation matrix
        final_transform = icp_result.transformation

        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000
        print(f"ICP duration: {duration_milliseconds:.2f} ms")

        return aligned_cloud, final_transform

    def icp_run(self):
        """
        运行 ICP 的主线程
        """
        rate = rospy.Rate(10)  # 设置发布频率 10Hz
        while not rospy.is_shutdown():
            if self.isreadyICP:
                # 执行 ICP 对齐
                final_transform = self.icp_function(
                    self.readyICP_cloud,
                    self.field_cloud,
                    transformation_epsilon=1e-10,
                    max_correspondence_distance=1,
                    euclidean_fitness_epsilon=0.001,
                    max_iterations=50
                )

                # 更新累积变换矩阵
                self.initial_alignment_transform = np.dot(final_transform, self.initial_alignment_transform)
                rospy.loginfo("ICP run: Initial alignment transform matrix:")
                rospy.loginfo("\n" + str(self.initial_alignment_transform))

                self.isICPFinish = True

            if self.isICPFinish:
                self.isreadyICP = False

                # 转换为 ROS 点云消息并发布
                ros_adjusted_cloud = self.open3d_to_ros(self.readyICP_cloud, frame_id="livox_frame")
                self.icp_adjusted_pub.publish(ros_adjusted_cloud)
                rospy.loginfo("Published ICP adjusted point cloud.")

            rate.sleep()

    def publish_centroid_markers(self, centroids):
        """
        发布质心点到 RViz
        :param centroids: List of centroids, where each centroid is a [x, y, z] list or numpy array.
        """
        # 创建 Marker 消息
        points = Marker()
        points.header.frame_id = "livox_frame"
        points.header.stamp = rospy.Time.now()
        points.ns = "centroids"
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0
        points.id = 0
        points.type = Marker.POINTS

        # 设置点的尺寸
        points.scale.x = 0.2  # 点的宽度
        points.scale.y = 0.2  # 点的高度

        # 设置点的颜色为绿色
        points.color.g = 1.0
        points.color.a = 1.0  # 不透明度

        # 添加质心点
        for centroid in centroids:
            p = Point()
            p.x = centroid[0]
            p.y = centroid[1]
            p.z = centroid[2]
            points.points.append(p)

        # 发布 Marker
        self.marker_pub.publish(points)
        rospy.loginfo("Published centroid markers.")

    def upsample_voxel_grid(self,cloud, leaf_size, points_per_voxel):
        """
        对点云进行体素下采样，并在每个体素中心点周围进行随机上采样。
        :param cloud: Open3D 点云对象 (o3d.geometry.PointCloud)
        :param leaf_size: 体素大小 (float)
        :param points_per_voxel: 每个体素中生成的随机点数 (int)
        """
        # 记录开始时间
        start_time = time.time()

        # 体素下采样
        downsampled = cloud.voxel_down_sample(voxel_size=leaf_size)

        # 创建用于上采样的点云
        upsampled_points = []

        # 随机偏移的分布
        distribution = np.random.uniform(-leaf_size / 2, leaf_size / 2, (points_per_voxel, 3))

        # 对下采样后的每个点进行随机上采样
        for point in np.asarray(downsampled.points):
            if point[0] > 15.0:  # 如果 x 坐标大于 15m
                offsets = distribution  # 完整的随机偏移
            else:
                offsets = distribution[: points_per_voxel // 2]  # 偏移数减半

            # 添加随机偏移后的点
            for offset in offsets:
                upsampled_points.append(point + offset)

        # 将上采样点转换为 Open3D 点云
        upsampled_cloud = o3d.geometry.PointCloud()
        upsampled_cloud.points = o3d.utility.Vector3dVector(np.array(upsampled_points))

        # 记录结束时间
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"upsample_voxel_grid: {duration_ms:.2f} ms")

        return upsampled_cloud

    def merge_point_clouds(self,point_cloud_list):
        """
        合并多个点云
        :param point_cloud_list: 包含多个 Open3D 点云对象的列表
        :return: 合并后的 Open3D 点云对象
        """
        # 创建一个新的空点云
        merged_cloud = o3d.geometry.PointCloud()

        # 遍历所有点云并合并
        for cloud in point_cloud_list:
            merged_cloud += cloud

        return merged_cloud

    def radius_outlier_removal(self,cloud, near_radius, near_neighbor, far_radius, far_neighbor):
        """
        应用半径滤波去除点云中的离群点。
        :param cloud: 输入点云 (o3d.geometry.PointCloud)
        :param near_radius: 近处点云的搜索半径
        :param near_neighbor: 近处点云的最小邻域点数
        :param far_radius: 远处点云的搜索半径
        :param far_neighbor: 远处点云的最小邻域点数
        :return: 经过半径滤波后的点云 (o3d.geometry.PointCloud)
        """

        # 检查点云是否为空
        if cloud.is_empty():
            print("radius_outlier_removal: Input cloud is empty!")
            return cloud

        # 分割点云为近处和远处两部分
        points = np.asarray(cloud.points)
        near_mask = points[:, 0] < 13.0  # x 坐标小于 13.0 为近处点云
        far_mask = points[:, 0] >= 13.0  # x 坐标大于等于 13.0 为远处点云

        near_cloud = cloud.select_by_index(np.where(near_mask)[0])
        far_cloud = cloud.select_by_index(np.where(far_mask)[0])

        # 半径滤波 - 近处点云
        filtered_near_cloud = o3d.geometry.PointCloud()
        if not near_cloud.is_empty():
            filtered_near_cloud, _ = near_cloud.remove_radius_outlier(nb_points=near_neighbor, radius=near_radius)

        # 半径滤波 - 远处点云
        filtered_far_cloud = o3d.geometry.PointCloud()
        if not far_cloud.is_empty():
            filtered_far_cloud, _ = far_cloud.remove_radius_outlier(nb_points=far_neighbor, radius=far_radius)

        # 合并过滤后的点云
        filtered_cloud = filtered_near_cloud + filtered_far_cloud

        return filtered_cloud

    def remove_overlapping_points(self,live_cloud, field_cloud):
        """
        去除动态点云的重叠点并进行障碍物分析和发布。
        :param live_cloud: 当前点云 (o3d.geometry.PointCloud)
        :param field_cloud: 目标点云 (o3d.geometry.PointCloud)

        """
        # 差异计算
        live_points = np.asarray(live_cloud.points)
        field_points = np.asarray(field_cloud.points)
        diff_points = [p for p in live_points if tuple(p) not in set(map(tuple, field_points))]

        dynamic_obstacles = o3d.geometry.PointCloud()
        dynamic_obstacles.points = o3d.utility.Vector3dVector(diff_points)

        # Y 轴滤波
        y_filtered = []
        for point in np.asarray(dynamic_obstacles.points):
            if -15 <= point[1] <= -0.6:
                y_filtered.append(point)
        dynamic_obstacles.points = o3d.utility.Vector3dVector(np.array(y_filtered))

        # X 轴滤波
        x_filtered = []
        for point in np.asarray(dynamic_obstacles.points):
            if 0.0 <= point[0] <= 27.5:
                x_filtered.append(point)
        dynamic_obstacles.points = o3d.utility.Vector3dVector(np.array(x_filtered))

        # Z 轴滤波
        z_filtered = []
        for point in np.asarray(dynamic_obstacles.points):
            if 0.0 <= point[2] <= 1.4:
                z_filtered.append(point)
        dynamic_obstacles.points = o3d.utility.Vector3dVector(np.array(z_filtered))

        # 发布动态障碍物点云
        ros_dynamic_obstacles = self.open3d_to_ros(dynamic_obstacles, frame_id="livox_frame")
        self.obstacle_cloud_pub.publish(ros_dynamic_obstacles)

        # 计算质心并发布
        dynamic_obstacles=self.radius_outlier_removal(dynamic_obstacles, 0.5, 2, 0.1, 10)
        centroids = self.fast_euclidean_cluster(dynamic_obstacles,0.5, 50, 2, 50000)
        # 累加点云
        self.upsample_voxel_grid(dynamic_obstacles, 0.1, 500)# 每个体素中生成10个点
        self.publish_centroid_markers(centroids)

    def fast_euclidean_cluster(self,cloud, radius, search_max_size, min_cluster_size, max_cluster_size):
        """
        快速欧式聚类函数。
        :param cloud: 输入点云 (o3d.geometry.PointCloud)
        :param radius: 搜索半径
        :param search_max_size: 最大邻域点数
        :param min_cluster_size: 最小聚类点数
        :param max_cluster_size: 最大聚类点数
        :return: 合并后的点云，质心列表
        """
        start_time = time.time()
        labels = []
        # 实例化聚类对象
        self.fec = FastEuclideanCluster(cloud)
        self.fec.setInputCloud(cloud)
        self.fec.setClusterTolerance(radius)
        self.fec.setSearchMaxsize(search_max_size)
        self.fec.setMinClusterSize(min_cluster_size)
        self.fec.setMaxClusterSize(max_cluster_size)

        # 提取聚类
        labels = self.fec.euclidean_cluster(labels)

        # 保存聚类结果和质心
        centroids = []
        cluster_result = o3d.geometry.PointCloud()

        for label in labels:
            # 提取当前聚类的点云
            cluster_points = np.asarray(cloud.points)[label]
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)

            # 合并聚类结果
            cluster_result += cluster_cloud

            # 计算质心
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

        elapsed_time = (time.time() - start_time) * 1000
        print(f"快速欧式聚类用时：{elapsed_time:.2f} ms")
        self.centroids = centroids
        return cluster_result, centroids

    def statistical_outlier_removal(self,cloud, num_neighbors, std_ratio):
        """
        使用统计滤波去除点云中的离群点。
        :param cloud: 输入点云 (o3d.geometry.PointCloud)
        :param num_neighbors: 设置领域点的个数 (int)
        :param std_ratio: 设置离群点的阈值 (float)
        :return: 过滤后的点云 (o3d.geometry.PointCloud)
        """
        # 执行统计滤波
        cloud_filtered, ind = cloud.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
        return cloud_filtered

    def cloud_callback(self, input_cloud):
        # 将 ROS 点云消息转换为 Open3D 点云
        source_cloud = self.ros_to_open3d(input_cloud)

        # 初始化水平角度
        if not self.isLevel:
            # 示例：直接标记为已完成，未实现 RANSAC 拟合
            self.isLevel = True
        else:
            # 应用旋转到原始点云
            source_cloud.transform(np.linalg.inv(self.level_transform))

        # 轴向过滤
        source_cloud = self.pass_through_filter(source_cloud, "x", 5.0, 32.0)
        source_cloud = self.pass_through_filter(source_cloud, "y", -8.0, 8.0)
        source_cloud = self.pass_through_filter(source_cloud, "z", -10, 3)

        # 累加远处点
        source_points = np.asarray(source_cloud.points)
        accumulated_points = []
        for point in source_points:
            if point[0] > 15.0:  # x > 15 的点
                accumulated_points.extend([point] * 10)  # 累加 10 次
            else:
                accumulated_points.append(point)
        source_cloud.points = o3d.utility.Vector3dVector(np.array(accumulated_points))

        # 根据处理阶段，决定发布或缓存点云
        if not self.isINITFinish:
            ros_adjusted_cloud = self.open3d_to_ros(source_cloud, frame_id="livox_frame")
            self.adjusted_pub.publish(ros_adjusted_cloud)
        else:
            # 应用初始变换
            source_cloud.transform(self.initial_alignment_transform)
            ros_adjusted_cloud = self.open3d_to_ros(source_cloud, frame_id="livox_frame")
            self.adjusted_pub.publish(ros_adjusted_cloud)

            # 缓存点云
            if len(self.cloud_buffer) < 20:
                self.cloud_buffer.append(source_cloud)
                if len(self.cloud_buffer) == 19:
                    self.readyICP_cloud = self.merge_point_clouds(self.cloud_buffer)
                    self.isreadyICP = True

        # 如果 ICP 已完成，移除重叠点
        if self.isICPFinish:
            self.remove_overlapping_points(source_cloud, self.field_cloud)

    def pass_through_filter(self, cloud, axis, min_value, max_value):
        """
        Open3D 模拟 PCL 的 PassThrough 滤波器。
        """
        points = np.asarray(cloud.points)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        mask = (points[:, axis_index] >= min_value) & (points[:, axis_index] <= max_value)
        filtered_cloud = cloud.select_by_index(np.where(mask)[0])
        return filtered_cloud

    def main(self):
        """
        ROS 节点的主入口函数。
        """
        rospy.init_node("pcl_matcher_node", anonymous=True)

        # 初始化 PCLMatcher 实例
        matcher = PCLMatcher()

        # 加载 PCD 文件
        pcd_file_path = "/"
        matcher.load_pcd(pcd_file_path)

        # 开始 ROS spin 循环
        rospy.spin()


if __name__ == "__main__":
    matcher = PCLMatcher()
    matcher.main()
