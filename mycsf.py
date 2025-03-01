import torch
import numpy as np
import time

class FastSearch:
    def __init__(self, device="cuda", Radius_upperbound=1):
        self.dynamic_pcd = None  # 动态障碍物点云，是寻找的目标
        self.device = device
        self.src_pcd = None
        self.r = Radius_upperbound  # 设定最大半径上限

    # 计算所有点到目标点的欧氏距离（只计算xy维度的距离）
    def compute_distances(self, point_cloud_tensor, point_0_tensor):
        """
        计算点云中所有点到点云0的欧氏距离，只计算xy维度
        :param point_cloud_tensor: 点云数据，PyTorch Tensor，形状为 (N, 3)
        :param point_0_tensor: 目标点，PyTorch Tensor，形状为 (3,)
        :return: 每个点到目标点的欧氏距离（xy维度），PyTorch Tensor，形状为 (N,)
        """
        # 只选择 x 和 y 维度
        point_cloud_xy = point_cloud_tensor[:, :2]
        point_0_xy = point_0_tensor[:2]

        # 计算 xy 维度的欧氏距离
        return torch.norm(point_cloud_xy - point_0_xy, dim=1)

    # 查找距离目标点最近的点，并判断最小距离是否超过设定的阈值
    def find_nearest_point(self, point_cloud, point_0):
        """
        找到点云中距离点云0最近的点（只考虑xy维度）
        :param point_cloud: 点云数据，NumPy 数组，形状为 (N, 3)
        :param point_0: 查询点，NumPy 数组，形状为 (3,)
        :return: 最近点，NumPy 数组，形状为 (3,)
        :return: 最近点的距离，float
        :return: 是否超过阈值，bool
        """
        # 转换为 PyTorch 张量，并移动到指定设备
        point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32).to(self.device)
        point_0_tensor = torch.tensor(point_0, dtype=torch.float32).to(self.device)

        # 计算每个点到点云0的距离（只考虑xy维度）
        distances = self.compute_distances(point_cloud_tensor, point_0_tensor)

        # 找到最小距离的索引
        min_distance_idx = torch.argmin(distances)

        # 获取距离最近的点
        nearest_point = point_cloud_tensor[min_distance_idx]

        # 获取最小距离
        min_distance = distances[min_distance_idx]

        # 判断最小距离是否超过阈值
        distance_exceeds_threshold = min_distance.item() > self.r

        # 转换为 NumPy 数组并返回
        nearest_point_xyz = nearest_point.cpu().numpy()  # 移动回 CPU 并转为 NumPy 数组
        return nearest_point_xyz, min_distance.item(), distance_exceeds_threshold


    # 示例使用
if __name__ == "__main__":
    # 假设 point_cloud 和 point_0 是你已经有的 NumPy 数组
    point_cloud = np.random.rand(15000, 3)  # 示例点云数据
    point_0 = np.array([1.0, 2.0, 3.0])  # 示例查询点
    f = FastSearch(device = "cuda",Radius_upperbound = 5)
    t1 = time.time()
    # 查找距离最近的点
    nearest_point, min_distance, ans = f.find_nearest_point(point_cloud, point_0)
    t2 = time.time()
    print("用时：",(t2-t1)*1000,"ms")
    print(f"最接近的点是: {nearest_point}, 距离是: {min_distance}")
