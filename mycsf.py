import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def register_point_clouds(source, target):
    # 这里可以使用 ICP 或其他配准方法
    threshold = 0.02
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_icp.transformation

def point_cloud_subtraction(source, target):
    source.transform(target)
    points_source = np.asarray(source.points)
    points_target = np.asarray(target.points)
    
    # 点云相减
    diff = points_source - points_target
    # 计算差值的范数以确定哪些点是前景点
    norm_diff = np.linalg.norm(diff, axis=1)
    
    # 设定一个阈值来过滤背景点
    threshold = 0.1
    foreground_indices = np.where(norm_diff > threshold)[0]
    foreground_points = points_source[foreground_indices]
    
    # 创建新的点云对象
    foreground_pcd = o3d.geometry.PointCloud()
    foreground_pcd.points = o3d.utility.Vector3dVector(foreground_points)
    
    return foreground_pcd

def main():
    # 读取点云
    source_pcd = load_point_cloud("source.ply")
    target_pcd = load_point_cloud("target.ply")
    
    # 配准点云
    transformation = register_point_clouds(source_pcd, target_pcd)
    
    # 点云相减
    foreground_pcd = point_cloud_subtraction(source_pcd, target_pcd)
    
    # 可视化
    o3d.visualization.draw_geometries([foreground_pcd])

if __name__ == "__main__":
    main()
