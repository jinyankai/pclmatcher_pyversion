import numpy as np
import open3d as o3d
import time
from fasteuclideancluster import FastEuclideanCluster
from get_initial_pose import Initial_pose

class PCLMatcher:
    def __init__(self):
        self.field_cloud = None
        self.filtered_cloud = None
        self.level_transform = np.eye(4)
        self.is_level = False
        self.ready_icp_cloud = None
        self.icp_cloud = None
        self.ground_ready_icp = None
        self.ground_field = None
        self.cumulative_transform = np.eye(4)
        self.max_icp_iterations = 10
        self.fitness_score_threshold = 0.01
        self.cloud_buffer = []
        self.map_feature_points = []
        self.sensor_feature_points = []
        self.get_initial_pose = Initial_pose()
        self.initial_alignment_transform = None
        self.m_centroids = []
        self.is_init_finish = False
###使用ros通讯功能未完成 部分使用self/print替代
    def init_initial_alignment(self):
        self.initial_alignment_transform = Initial_pose.img_to_lidar(self.get_initial_pose)


    def clicked_point_callback(self, point):
        print(f"x: {point[0]}, y: {point[1]}, z: {point[2]}")
        vector_point = np.array(point)
        if len(self.map_feature_points) < 4:
            self.map_feature_points.append(vector_point)
            print(f"Map feature point {len(self.map_feature_points)} added: {vector_point}")
        elif len(self.sensor_feature_points) < 4:
            self.sensor_feature_points.append(vector_point)
            print(f"Sensor feature point {len(self.sensor_feature_points)} added: {vector_point}")

        if len(self.map_feature_points) == 4 and len(self.sensor_feature_points) == 4:
            self.compute_initial_alignment()

    def compute_initial_alignment(self):
        map_points = o3d.geometry.PointCloud()
        sensor_points = o3d.geometry.PointCloud()
        # Using SVD for rigid transformation estimation
        for i in range(len(self.map_feature_points)):
            map_points.points.append(self.map_feature_points[i])
            sensor_points.points.append(self.sensor_feature_points[i])
        pcl = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        pcl.compute_transformation(sensor_points, map_points, self.initial_alignment_transform)





    def load_pcd(self, file_path):
        self.field_cloud = o3d.io.read_point_cloud(file_path)
        print(f"Loaded {len(self.field_cloud.points)} data points from {file_path}")

        # Voxel grid filtering
        voxel_size = 0.1
        self.field_cloud = self.field_cloud.voxel_down_sample(voxel_size)
        tmp = o3d.geometry.PointCloud()
        tmp.points = self.field_cloud.points
        self.field_cloud = tmp

    def icp_function(self, source_cloud, target_cloud, max_correspondence_distance,
                     ):
        start_time = time.time()

        # 创建 ICP 对象
        icp = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            max_correspondence_distance,
            np.eye(4),  # 初始变换矩阵
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        # 执行 ICP 迭代
        result = icp.run()

        # 将变换应用于源点云
        align_cloud = icp.get_source().transform(result.transformation)

        # 获取最终变换矩阵
        final_transform = result.transformation

        end_time = time.time()
        # 计算并打印 ICP 算法耗时（秒）
        duration_seconds = end_time - start_time
        print(f"Duration in seconds: {duration_seconds} s")

        return final_transform


    def icp_run(self):
        while True:  # This would normally be tied to a condition
            if self.is_ready_icp:
                final_transform = self.icp_function(self.ready_icp_cloud, self.field_cloud,max_correspondence_distance=1)
                self.initial_alignment_transform = np.matmul(self.initial_alignment_transform, final_transform)

                print("icp_run Initial alignment transform matrix:")
                print(self.initial_alignment_transform)

                self.is_icp_finish = True

            if self.is_icp_finish:
                self.is_ready_icp = False
                # Publish adjusted cloud here
                print("Publishing adjusted cloud...")  # Replace with actual publishing logic

            time.sleep(0.1)  # Simulate rate sleep


    def publish_centroid_markers(self, centroids):
        # Visualization logic (e.g., using Matplotlib or another library)
        print("Centroids:")
        for centroid in centroids:
            print(f"Centroid: x={centroid[0]}, y={centroid[1]}, z={centroid[2]}")

    def upsample_voxel_grid(self, cloud, leaf_size, points_per_voxel):
        start_time = time.time()

        # Downsampling using voxel grid
        downsampled = cloud.voxel_down_sample(leaf_size)

        upsampled = o3d.geometry.PointCloud()
        points = np.asarray(downsampled.points)

        for point in points:
            if point[0] > 15.0:  # Only upsample if x > 15
                for _ in range(points_per_voxel):
                    new_point = point + np.random.uniform(-leaf_size / 2, leaf_size / 2, 3)
                    upsampled.points.append(new_point)
            else:
                for _ in range(points_per_voxel // 2):
                    new_point = point + np.random.uniform(-leaf_size / 2, leaf_size / 2, 3)
                    upsampled.points.append(new_point)

        cloud.points = o3d.utility.Vector3dVector(np.asarray(upsampled.points))
        duration_ms = (time.time() - start_time) * 1000
        print(f"upsampleVoxelGrid: {duration_ms:.2f} ms")

    def merge_point_clouds(self, cloud_list):
        merged_cloud = o3d.geometry.PointCloud()
        for cloud in cloud_list:
            merged_cloud += cloud
        return merged_cloud

    def remove_overlapping_points(self, live_cloud, field_cloud):
        # Placeholder for removing overlapping points
        # Use Open3D functions to handle point cloud differences
        # This would typically involve voxel grid filtering or other techniques
        # Compute differences between two point clouds

        distances = live_cloud.compute_point_cloud_distance(field_cloud)
        dynamic_obstacles = live_cloud.select_by_index(np.where(np.array(distances) > 0.1)[0])

        # Apply filtering on the Y-axis
        dynamic_obstacles = dynamic_obstacles.select_by_index(
            np.where((np.asarray(dynamic_obstacles.points)[:, 1] >= -15) &
                     (np.asarray(dynamic_obstacles.points)[:, 1] <= -0.6))[0]
        )

        # Apply filtering on the X-axis
        dynamic_obstacles = dynamic_obstacles.select_by_index(
            np.where((np.asarray(dynamic_obstacles.points)[:, 0] >= 0.0) &
                     (np.asarray(dynamic_obstacles.points)[:, 0] <= 27.5))[0]
        )

        # Apply filtering on the Z-axis
        dynamic_obstacles = dynamic_obstacles.select_by_index(
            np.where((np.asarray(dynamic_obstacles.points)[:, 2] >= 0.0) &
                     (np.asarray(dynamic_obstacles.points)[:, 2] <= 1.4))[0]
        )

        # Apply radius outlier removal
        dynamic_obstacles = self.radius_outlier_removal(dynamic_obstacles, 0.5, 2, 0.1, 10)

        # Perform fast Euclidean clustering (placeholder)

        self.fast_euclidean_cluster(dynamic_obstacles, 0.5, 50, 2, 50000)
        self.upsample_voxel_grid(dynamic_obstacles, 0.1, 500)

    def print_centroids(self, centroids):
        if not centroids:
            print("No centroids available to display.")
            return

        for index, centroid in enumerate(centroids, start=1):
            print(f"Cluster {index} centroid: X={centroid[0]}, Y={centroid[1]}, Z={centroid[2]}")

    def radius_outlier_removal(self, cloud, near_radius, near_neighbor, far_radius, far_neighbor):
        # 检查输入点云是否为空
        if cloud.is_empty():
            print("Input cloud is empty or null!")
            return

        # 分割点云为近处和远处两部分
        near_cloud_points = []
        far_cloud_points = []

        for point in np.asarray(cloud.points):
            if point[0] >= 13.0:
                far_cloud_points.append(point)
            else:
                near_cloud_points.append(point)

        # 应用半径滤波 - 近处点云
        filtered_near_cloud = self.filter_radius(np.array(near_cloud_points), near_radius, near_neighbor)

        # 应用半径滤波 - 远处点云
        filtered_far_cloud = self.filter_radius(np.array(far_cloud_points), far_radius, far_neighbor)

        # 合并过滤后的点
        filtered_points = np.vstack((filtered_near_cloud, filtered_far_cloud))
        cloud.points = o3d.utility.Vector3dVector(filtered_points)

    def filter_radius(self, points, radius, min_neighbors):
        filtered_points = []
        for i, point in enumerate(points):
            # 计算该点与其他点的距离
            distances = np.linalg.norm(points - point, axis=1)
            # 统计在指定半径内的邻居数量
            count = np.sum(distances < radius)
            if count >= min_neighbors:
                filtered_points.append(point)
        return np.array(filtered_points)


    def statistical_outlier_removal(self, cloud, num_neighbors, std_ratio):
        # 实现统计离群点移除
        pass

    def fast_euclidean_cluster(self, cloud, cluster_tolerance, min_cluster_size, max_cluster_size, max_clusters,centroids):

        labels = []
        fce = FastEuclideanCluster(cloud)
        fce.setClusterTolerance(cluster_tolerance)
        fce.setMinClusterSize(min_cluster_size)
        fce.setMaxClusterSize(max_cluster_size)
        fce.setSearchMaxsize(max_clusters)
        fce.euclidean_cluster(labels)
        centroids.clear()
        cluster_results = o3d.geometry.PointCloud()
        cluster_tmp = o3d.geometry.PointCloud()
        for i in range(len(labels)):

            cluster_indices = np.where(labels == labels[i])[0]
            cluster_points = cloud.points[cluster_indices]
            cluster_tmp = o3d.utility.Vector3dVector(cluster_points)
            cluster_results += cluster_tmp
        # 假设 `cluster_tmp` 是一个 Open3D 点云对象，包含聚类后的点
        # `centroids` 是一个列表，用于存储质心坐标

        # 计算质心
        centroid = np.mean(cluster_tmp.points, axis=0)

        # 将质心坐标转换为 Open3D 的 Eigen::Vector3f 类型并添加到列表中
        # Open3D 使用 o3d.utility.Vector3dVector 来表示三维向量
        centroids.append(o3d.utility.Vector3dVector([centroid[0], centroid[1], centroid[2]]))

        # 打印聚类用时
        #print("快速欧式聚类用时：", time.toc(), "ms")

        # 如果需要替换原始点云，可以使用以下代码
        cloud.points = cluster_tmp.points

        return fce


# 使用示例
'''if __name__ == "__main__":
    matcher = PCLMatcher()
    matcher.load_pcd("path/to/your/pointcloud.pcd")
    # 添加更多处理逻辑'''