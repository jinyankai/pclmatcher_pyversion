import open3d as o3d
import numpy as np

def compute_similarity(pcd1, pcd2, distance_threshold=0.01):
    """
    计算两个点云的相似部分，返回相似部分的点。
    
    参数：
    - pcd1: Open3D 点云对象 1
    - pcd2: Open3D 点云对象 2
    - distance_threshold: 相似部分的最大距离，默认 0.01
    
    返回：
    - similar_points: 相似部分的点列表
    """
    # 使用KDTree来查找最近的点
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    similar_points = []

    # 对每个点云中的点进行遍历，找到距离较近的点
    for point in np.asarray(pcd1.points):
        [_, idx, _] = pcd1_tree.search_knn_vector_3d(point, 1)
        dist = np.linalg.norm(np.asarray(pcd2.points)[idx] - point)
        
        if dist < distance_threshold:
            similar_points.append(point)
    
    return np.array(similar_points)

def compute_difference(pcd1, pcd2, distance_threshold=0.01):
    """
    计算两个点云的差异部分，即去除相似部分后的剩余部分。
    
    参数：
    - pcd1: Open3D 点云对象 1
    - pcd2: Open3D 点云对象 2
    - distance_threshold: 相似部分的最大距离，默认 0.01
    
    返回：
    - diff_pcd: 相差部分的点云
    """
    # 计算相似部分
    similar_points = compute_similarity(pcd1, pcd2, distance_threshold)
    
    # 删除相似部分的点
    pcd1_diff_points = [point for point in np.asarray(pcd1.points) if np.linalg.norm(similar_points - point, axis=1).min() >= distance_threshold]
    pcd2_diff_points = [point for point in np.asarray(pcd2.points) if np.linalg.norm(similar_points - point, axis=1).min() >= distance_threshold]

    # 将差异点创建为点云
    diff_pcd = o3d.geometry.PointCloud()
    diff_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd1_diff_points, pcd2_diff_points], axis=0))
    
    return diff_pcd

# 读取两个点云文件（这里假设是ply格式）
pcd1 = o3d.io.read_point_cloud("point_cloud_1.ply")
pcd2 = o3d.io.read_point_cloud("point_cloud_2.ply")

# 计算差异点云
diff_pcd = compute_difference(pcd1, pcd2)

# 可视化差异部分
o3d.visualization.draw_geometries([diff_pcd])
