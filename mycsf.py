import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class BucketSet:
    def __init__(self, size):
        self.bucket = np.zeros(size, dtype=bool)
        self.size = 0
        self.now_first = -1

    def insert(self, value):
        if not self.bucket[value]:
            self.bucket[value] = True
            self.size += 1
            if value < self.now_first or self.now_first == -1:
                self.now_first = value

    def erase(self, value):
        if self.bucket[value]:
            self.bucket[value] = False
            self.size -= 1
            if self.size == 0:
                self.now_first = -1
            elif value == self.now_first:
                for i in range(value + 1, len(self.bucket)):
                    if self.bucket[i]:
                        self.now_first = i
                        break

def differing_dbscan(cloud, zero_pos, eps, min_points_k):
    # Check if the point cloud is empty
    if len(cloud.points) == 0:
        return []

    # Convert points to numpy array
    points = np.asarray(cloud.points)
    
    # Build the KDTree for fast neighbor search
    kdtree = KDTree(points)
    
    # Calculate neighbors for each point in parallel
    nbs = []
    with ThreadPoolExecutor() as executor:
        nbs = list(executor.map(lambda pt: kdtree.query_ball_point(pt, eps), points))
    
    # Initialize labels with -2 (unvisited)
    labels = np.full(len(points), -2)
    cluster_label = 0
    
    # DBSCAN clustering
    for idx in range(len(points)):
        # Skip already visited points
        if labels[idx] != -2:
            continue

        # Set min_points based on distance from zero_pos
        min_points = min_points_k / np.linalg.norm(points[idx] - zero_pos)**2
        
        # If the point has fewer neighbors than the threshold, label as noise
        if len(nbs[idx]) < min_points:
            labels[idx] = -1
            continue

        # Use BucketSet for efficient management of neighbors and visited points
        nbs_next = BucketSet(len(points))
        nbs_visited = BucketSet(len(points))
        
        for nb in nbs[idx]:
            nbs_next.insert(nb)
        nbs_visited.insert(idx)
        
        labels[idx] = cluster_label
        
        # BFS to expand cluster
        while nbs_next.size > 0:
            nb = nbs_next.now_first
            nbs_next.erase(nb)
            nbs_visited.insert(nb)

            # If the neighbor is not visited, label it with the current cluster
            if labels[nb] == -2:
                labels[nb] = cluster_label

            # If the neighbor has enough points, add its neighbors to the list
            if len(nbs[nb]) >= min_points:
                for qnb in nbs[nb]:
                    if not nbs_visited.bucket[qnb]:
                        nbs_next.insert(qnb)

        cluster_label += 1

    return labels


def normal_dbscan(cloud, eps, min_points):
    # Check if the point cloud is empty
    if len(cloud.points) == 0:
        return []

    # Convert points to numpy array
    points = np.asarray(cloud.points)

    # Build the KDTree for fast neighbor search
    kdtree = KDTree(points)

    # Calculate neighbors for each point in parallel
    nbs = []
    with ThreadPoolExecutor() as executor:
        nbs = list(executor.map(lambda pt: kdtree.query_ball_point(pt, eps), points))

    # Initialize labels with -2 (unvisited)
    labels = np.full(len(points), -2)
    cluster_label = 0

    # DBSCAN clustering
    for idx in range(len(points)):
        # Skip already visited points
        if labels[idx] != -2:
            continue

        # If the point has fewer neighbors than the threshold, label as noise
        if len(nbs[idx]) < min_points:
            labels[idx] = -1
            continue

        # Use BucketSet for efficient management of neighbors and visited points
        nbs_next = BucketSet(len(points))
        nbs_visited = BucketSet(len(points))
        
        for nb in nbs[idx]:
            nbs_next.insert(nb)
        nbs_visited.insert(idx)
        
        labels[idx] = cluster_label
        
        # BFS to expand cluster
        while nbs_next.size > 0:
            nb = nbs_next.now_first
            nbs_next.erase(nb)
            nbs_visited.insert(nb)

            # If the neighbor is not visited, label it with the current cluster
            if labels[nb] == -2:
                labels[nb] = cluster_label

            # If the neighbor has enough points, add its neighbors to the list
            if len(nbs[nb]) >= min_points:
                for qnb in nbs[nb]:
                    if not nbs_visited.bucket[qnb]:
                        nbs_next.insert(qnb)

        cluster_label += 1

    return labels


# Example usage
if __name__ == "__main__":
    # Create a point cloud (for demonstration purposes)
    pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")

    zero_pos = np.array([0.0, 0.0, 0.0])  # Reference position
    eps = 0.1  # Neighborhood radius
    min_points_k = 10  # Minimum points in a neighborhood (for differing DBSCAN)
    min_points = 5  # Minimum points in a neighborhood (for normal DBSCAN)

    # Call the DBSCAN functions
    labels_differing = differing_dbscan(pcd, zero_pos, eps, min_points_k)
    labels_normal = normal_dbscan(pcd, eps, min_points)

    print("Differing DBSCAN labels:", labels_differing)
    print("Normal DBSCAN labels:", labels_normal)
