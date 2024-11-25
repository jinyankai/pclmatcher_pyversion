import cupy as cp
import numpy as np
from scipy.spatial import distance

# 生成点云数据，假设有20000个3D点
num_points = 20000
points_cpu = np.random.rand(num_points, 3).astype(np.float32)

# 将数据从CPU传到GPU
points_gpu = cp.asarray(points_cpu)

# 计算欧氏距离矩阵（可以用CuPy来加速计算）
def compute_distance_matrix(points_gpu):
    # 计算点之间的欧氏距离矩阵
    # 点云数据是N x 3维度，使用广播机制来计算所有点对的距离
    dist_matrix = cp.linalg.norm(points_gpu[:, cp.newaxis] - points_gpu, axis=2)
    return dist_matrix

# 计算距离矩阵
dist_matrix = compute_distance_matrix(points_gpu)

# 设置一个距离阈值，来判断哪些点是重叠的
threshold = 0.02  # 设置重叠点的距离阈值
overlap_mask = dist_matrix < threshold

# 设置对角线为False，因为每个点与自己距离为0
cp.fill_diagonal(overlap_mask, False)

# 通过mask去除重叠点
# 重叠点将被标记为True，我们根据这个mask来去除重复点
unique_points_mask = cp.any(overlap_mask, axis=1)

# 得到不重复的点云
unique_points = points_gpu[~unique_points_mask]

# 将结果从GPU转回到CPU
unique_points_cpu = cp.asnumpy(unique_points)

# 打印去重后的点云
print(f"去重后的点云数量: {unique_points_cpu.shape[0]}")
