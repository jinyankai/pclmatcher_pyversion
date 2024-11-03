import open3d as o3d
import numpy as np
import time

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def compute_difference(Acloud, Bcloud, threshold):
    # KDTree 搜索
    kdtree = o3d.geometry.KDTreeFlann(Acloud)
    
    # 标记重合点
    LA01 = np.zeros(len(Acloud.points), dtype=int)  # Acloud 中的重合标记
    LB01 = np.zeros(len(Bcloud.points), dtype=int)  # Bcloud 中的重合标记

    for i in range(len(Bcloud.points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(Bcloud.points[i], 1)
        if k > 0 and np.linalg.norm(Bcloud.points[i] - Acloud.points[idx[0]]) <= threshold:
            LB01[i] = 1
            LA01[idx[0]] = 1

    # 提取不同点
    LA = [i for i in range(len(Acloud.points)) if LA01[i] == 0]
    LB = [i for i in range(len(Bcloud.points)) if LB01[i] == 0]

    Arecloud = Acloud.select_by_index(LA)  # A 中剩余的点
    Brecloud = Bcloud.select_by_index(LB)  # B 中剩余的点

    return Arecloud, Brecloud

def visualize_point_clouds(Acloud, Bcloud, Brecloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(Acloud)
    vis.add_geometry(Bcloud)
    vis.add_geometry(Brecloud)

    vis.run()
    vis.destroy_window()

def main():
    start_time = time.time()

    # 读取点云
    Acloud = load_point_cloud("1 - Cloud.pcd")
    Bcloud = load_point_cloud("yv1 Cloud.pcd")

    # 设置阈值
    threshold = 1e-5  # 你可以根据需要调整这个值

    # 计算差异
    Arecloud, Brecloud = compute_difference(Acloud, Bcloud, threshold)

    # 保存结果
    o3d.io.write_point_cloud("FISHNEEDLVBO.pcd", Brecloud)

    # 可视化
    visualize_point_clouds(Acloud, Bcloud, Brecloud)

    end_time = time.time()
    print(f"处理时间: {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    main()
