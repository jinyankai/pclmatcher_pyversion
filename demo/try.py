import open3d as o3d
import numpy as np

class PCLMatcher:
    def __init__(self):
        self.m_centroids = []

    def remove_overlapping_points(self, live_cloud, field_cloud):
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

        # Upsample the point cloud
        self.upsample_voxel_grid(dynamic_obstacles, 0.1, 500)

        # Publish or visualize dynamic obstacles (you can implement your own method)
        self.publish_dynamic_obstacles(dynamic_obstacles)
        self.print_centroids(self.m_centroids)

    def radius_outlier_removal(self, cloud, near_radius, near_neighbors, far_radius, far_neighbors):
        # Placeholder for radius outlier removal logic
        return cloud  # Modify this to implement actual filtering

    def fast_euclidean_cluster(self, cloud, cluster_tolerance, min_cluster_size, max_cluster_size, max_clusters):
        # Placeholder for clustering logic
        pass

    def upsample_voxel_grid(self, cloud, leaf_size, points_per_voxel):
        # Placeholder for upsampling logic
        pass

    def publish_dynamic_obstacles(self, cloud):
        # Placeholder for publishing logic
        print("Dynamic obstacles published!")

    def print_centroids(self, centroids):
        if not centroids:
            print("No centroids available to display.")
            return
        for index, centroid in enumerate(centroids, start=1):
            print(f"Cluster {index} centroid: X={centroid[0]}, Y={centroid[1]}, Z={centroid[2]}")