^Vimport rosbag
import matplotlib.pyplot as plt
import numpy as np

# 定义函数来提取rosbag文件中的数据
def extract_x_data_from_bag(bagfile, topic_name):
    x_data = []
    timestamps = []

    with rosbag.Bag(bagfile, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            x_data.append(msg.pose.position.x if topic == "/ground_truth/state_triangle_link_base" else msg.pose.position.x)
          timestamps.append(t.to_sec())  # 时间戳转换为秒
    return np.array(timestamps), np.array(x_data)

# 设置文件路径和主题名称
bag_files = ["bag1.bag", "bag2.bag", "bag3.bag"]  # 请根据实际情况替换文件路径
topic_ground_truth = "/ground_truth/state_triangle_link_base"
topic_target = "/target"

# 为了画6根曲线，6种不同颜色，2个文件各画两条曲线
colors = ['b', 'r', 'g', 'c', 'm', 'y']  # 选择6种颜色

# 创建图形
plt.figure(figsize=(10, 6))

# 遍历每个bag文件并提取数据
for i, bag_file in enumerate(bag_files):
    # 提取ground_truth数据
    timestamps_gt, x_data_gt = extract_x_data_from_bag(bag_file, topic_ground_truth)
    # 提取target数据
    timestamps_target, x_data_target = extract_x_data_from_bag(bag_file, topic_target)

    # 绘制曲线，每个bag文件两根曲线
    plt.plot(timestamps_gt, x_data_gt, color=colors[i*2], label=f'GT {bag_file} - x', linewidth=2)
    plt.plot(timestamps_target, x_data_target, color=colors[i*2+1], label=f'Target {bag_file} - x', linewidth=2)

# 添加图例和标签
plt.title('Comparison of x Data from /ground_truth and /target Topics')
plt.xlabel('Time (s)')
plt.ylabel('X Position')
plt.legend(loc='best')
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

