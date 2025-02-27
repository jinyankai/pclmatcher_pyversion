import rosbag
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 定义函数来提取rosbag文件中的数据
def extract_error_data_from_bag(bagfile, topic_name):
    error_data = []
    timestamps = []
    
    with rosbag.Bag(bagfile, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            error_data.append(msg.data)  # 假设/error消息包含一个data字段
            timestamps.append(t.to_sec())  # 时间戳转换为秒
    return np.array(timestamps), np.array(error_data)

# 设置文件路径和主题名称
bag_files = ["bag1.bag", "bag2.bag", "bag3.bag"]  # 请根据实际情况替换文件路径
topic_error = "/error"

# 为了画3根曲线，选择颜色
colors = ['b', 'g', 'r']

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 用来存放MSE的列表
mse_values = []

# 获取最小时间范围，进行对齐
all_timestamps = []
for bag_file in bag_files:
    timestamps, _ = extract_error_data_from_bag(bag_file, topic_error)
    all_timestamps.append(timestamps)

# 获取对齐后时间的最小时间和最大时间
min_time = max([t.min() for t in all_timestamps])
max_time = min([t.max() for t in all_timestamps])

# 遍历每个bag文件并提取数据
for i, bag_file in enumerate(bag_files):
    timestamps, error_data = extract_error_data_from_bag(bag_file, topic_error)
    
    # 对齐数据，只选择在最小和最大时间范围内的数据
    mask = (timestamps >= min_time) & (timestamps <= max_time)
    timestamps_aligned = timestamps[mask]
    error_data_aligned = error_data[mask]

    # 绘制错误数据曲线
    ax1.plot(timestamps_aligned, error_data_aligned, color=colors[i], label=f'Error {bag_file}')
    
    # 计算MSE
    mse = mean_squared_error(error_data_aligned, np.zeros_like(error_data_aligned))  # 假设目标值为零
    mse_values.append(mse)

# 绘制左侧图表
ax1.set_title('Error Curves from Different Rosbags')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Error')
ax1.legend(loc='upper right')
ax1.grid(True)

# 绘制右侧MSE柱状图
ax2.bar(range(1, len(mse_values) + 1), mse_values, color=colors[:len(mse_values)])
ax2.set_title('Mean Squared Error (MSE)')
ax2.set_xlabel('Rosbag')
ax2.set_ylabel('MSE')
ax2.set_xticks(range(1, len(mse_values) + 1))
ax2.set_xticklabels([f'Bag {i+1}' for i in range(len(mse_values))])

# 添加颜色对应的bag名称
for i, bag_file in enumerate(bag_files):
    ax1.text(0.5, -0.12 - i * 0.05, f'{bag_file}', color=colors[i], ha='center', va='center', transform=ax1.transAxes)

# 调整布局
plt.tight_layout()
plt.show()
