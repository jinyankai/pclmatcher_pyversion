import CSF
import numpy as np
import open3d as o3d

class ClothSimulationFilter:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.csf = CSF.CSF()
        self.initialize_config()

    def initialize_config(self):
        with open(self.config_file_path, 'r') as f:
            config = {}
            for line in f:
                key, value = line.strip().split('=')
                config[key] = value

        # 从配置文件读取参数并设置 CSF 实例的参数
        self.csf.params.bSloopSmooth = config.get("slop_smooth", "false").lower() == "true"
        self.csf.params.class_threshold = float(config.get("class_threshold", 0.5))
        self.csf.params.cloth_resolution = float(config.get("cloth_resolution", 0.1))
        self.csf.params.interations = int(config.get("iterations", 500))
        self.csf.params.rigidness = int(config.get("rigidness", 3))
        self.csf.params.time_step = float(config.get("time_step", 0.65))

    def filter_ground_from_point_cloud(self, input_cloud):
        # 将输入的 Open3D 点云转换为 numpy 数组
        points = np.asarray(input_cloud.points)

        # 设置输入点云到 CSF 库
        self.csf.setPointCloud(points)

        # 执行地面提取操作
        ground_indexes = []
        off_ground_indexes = []
        self.csf.do_filtering(ground_indexes, off_ground_indexes, exportCloth=False)

        # 使用 Open3D 根据索引创建地面点云
        ground_cloud = o3d.geometry.PointCloud()
        ground_cloud.points = o3d.utility.Vector3dVector(points[ground_indexes])

        return ground_cloud


# 示例用法
if __name__ == "__main__":
    # 假设您有一个配置文件 'config.txt' 和一个点云文件 'point_cloud.ply'
    config_file_path = 'params.cfg'
    point_cloud_file = 'point_cloud.pcd'

    # 加载点云
    input_cloud = o3d.io.read_point_cloud(point_cloud_file)

    # 创建 ClothSimulationFilter 实例并滤除地面点
    csf_filter = ClothSimulationFilter(config_file_path)
    ground_cloud = csf_filter.filter_ground_from_point_cloud(input_cloud)

    # 保存地面点云或可视化
    o3d.io.write_point_cloud('ground_cloud.ply', ground_cloud)
    o3d.visualization.draw_geometries([ground_cloud])