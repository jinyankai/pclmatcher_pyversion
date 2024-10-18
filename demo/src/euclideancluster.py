import open3d as o3d
import numpy as np
import random
class euclideancluster:
    def __init__(self, pcd):
        self.m_tolerance = 0.02
        self.m_min_cluster_size = 100
        self.m_max_cluster_size = 25000
        self.m_pcd = pcd
    def setInputCloud(self,pcd): # 输入点云
            self.m_pcd = pcd
    def setClusterTolerance(self,tolerance):# 搜索半径
            self.m_tolerance = tolerance
    def setMinClusterSize(self,min_cluster_size):# 聚类最小点数
            self.m_min_cluster_size = min_cluster_size
    def setMaxClusterSize(self,max_cluster_size):# 聚类最大点数
            self.m_max_cluster_size = max_cluster_size
    def euclidean_cluster(self,cluster_indices):# 欧式聚类
        nn_indices = []
        nn_dist = []
        processed = []
        pcd_tree = o3d.geometry.KDTreeFlann(self.m_pcd).setInputCloud(self.m_pcd)
        for i in range(len(cluster_indices)):
            if(processed[i]):
                continue
            seed_queue = []
            sq_indx = 0
            seed_queue.append(cluster_indices[i])
            processed[i] = True

            while(sq_indx < len(seed_queue)):
                if(pcd_tree.search_knn_vector_3d(self.m_pcd.points[seed_queue[sq_indx]], self.m_tolerance, nn_indices, nn_dist) > 0):
                    sq_indx += 1
                    continue
                for j in range(len(nn_indices)):
                    if(processed[nn_indices[j]]):
                        continue
                    seed_queue.append(nn_indices[j])
                    processed[nn_indices[j]] = True
                sq_indx += 1



    def clusterColor(self,pcd): # 聚类结果分类渲染
        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)
        for i in range(len(pcd.points)):
            pcd.colors[i] = [R,G,B]
        return pcd


