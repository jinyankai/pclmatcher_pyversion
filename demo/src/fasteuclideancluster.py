import open3d as o3d
import numpy as np
import random
class FastEuclideanCluster:
    def __init__(self, pcd):
        self.m_tolerance = 0.02
        self.max_size = 50
        self.m_min_cluster_size = 100
        self.m_max_cluster_size = 25000
        self.m_pcd = pcd
    def setInputCloud(self,pcd): # 输入点云
            self.m_pcd = pcd
    def setSearchMaxsize(self,max_size):
            self.max_size = max_size
    def setClusterTolerance(self,tolerance):# 搜索半径
            self.m_tolerance = tolerance
    def setMinClusterSize(self,min_cluster_size):# 聚类最小点数
            self.m_min_cluster_size = min_cluster_size
    def setMaxClusterSize(self,max_cluster_size):# 聚类最大点数
            self.m_max_cluster_size = max_cluster_size
    def euclidean_cluster(self,cluster_indices):# 欧式聚类

            labels = np.zeros(len(self.m_pcd), dtype=int)  # 初始化标签为0
            seg_lab = 1
            cluster_indices = []

            # 创建kd-tree
            pcd_tree = o3d.geometry.KDTreeFlann(self.m_pcd)

            for i in range(len(self.m_pcd)):
                if labels[i] == 0:  # 标签为0
                    [k, idx, _] = pcd_tree.search_radius_vector_3d(self.m_pcd[i], self.m_tolerance)
                    n_labs = {labels[j] for j in idx if labels[j] != 0}  # 使用集合推导

                        # 找到最小标签

                    min_seg_lab = min(n_labs) if n_labs else seg_lab

                    # 合并标签
                    for n_lab in n_labs:
                        if n_lab > min_seg_lab:
                            labels[labels == n_lab] = min_seg_lab

                    labels[idx] = min_seg_lab  # 标记邻近点
                    seg_lab += 1

            # 根据标签生成聚类索引
            seg_id = {}
            index = 1

            for i, label in enumerate(labels):
                if label not in seg_id:
                    seg_id[label] = index
                    cluster_indices.append([])
                    index += 1
                cluster_indices[seg_id[label] - 1].append(i)

            # 筛选符合条件的聚类
            valid_clusters = [cluster for cluster in cluster_indices if
                              self.m_tolerance < len(cluster) < self.m_max_cluster_size]

            return valid_clusters

        # 使用示例
        # pcd = o3d.io.read_point_cloud("your_point_cloud.ply")
        # clusterer = FastEuclideanCluster(pcd, tolerance=0.05, min_cluster_size=10, max_cluster_size=100)
        # clusters = clusterer.extract()
    def clusterColor(self,pcd): # 聚类结果分类渲染
            R = random.randint(0,255)
            G = random.randint(0,255)
            B = random.randint(0,255)
            for i in range(len(pcd.points)):
                pcd.colors[i] = [R,G,B]
            return pcd