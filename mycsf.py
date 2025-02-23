import rosbag
import matplotlib.pyplot as plt
import numpy as np

# 从rosbag文件中提取/error话题的数据
def extract_error_data(bag_file, topic='/error'):
    timestamps = []
    error_values = []

    # 打开rosbag文件
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            timestamps.append(t.to_sec())  # 转换为秒
            error_values.append(msg.data)  # 假设/error话题包含data字段

    return np.array(timestamps), np.array(error_values)

# 对两个rosbag文件的时间戳范围进行截断，确保它们的x轴有相同的数据范围
def truncate_data(timestamps1, error_values1, timestamps2, error_values2):
    # 计算两个时间范围的交集
    start_time = max(timestamps1.min(), timestamps2.min())
    end_time = min(timestamps1.max(), timestamps2.max())

    # 截取数据，使得两个时间轴在同一范围内
    mask1 = (timestamps1 >= start_time) & (timestamps1 <= end_time)
    mask2 = (timestamps2 >= start_time) & (timestamps2 <= end_time)

    return timestamps1[mask1], error_values1[mask1], timestamps2[mask2], error_values2[mask2]

# 绘制两个rosbag文件中的/error话题数据
def plot_error_data(file1, file2):
    # 提取两个rosbag文件中的/error数据
    timestamps1, error_values1 = extract_error_data(file1)
    timestamps2, error_values2 = extract_error_data(file2)

    # 截断数据以确保时间范围一致
    timestamps1, error_values1, timestamps2, error_values2 = truncate_data(timestamps1, error_values1, timestamps2, error_values2)

    # 创建一个图形
    plt.figure(figsize=(10, 6))

    # 绘制第一个rosbag的数据曲线
    plt.plot(timestamps1, error_values1, label='Bag 1', color='b')

    # 绘制第二个rosbag的数据曲线
    plt.plot(timestamps2, error_values2, label='Bag 2', color='r')

    # 添加标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Error Value')
    plt.title('Error Topic Comparison')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

# 调用函数，传入两个rosbag文件路径
file1 = '/home/nvidia/rosbag/bag/2025-02-22-20-33-46.bag'  # 替换为第一个rosbag文件路径
file2 = '/home/nvidia/rosbag/target/2025-02-23-15-57-32.bag'  # 替换为第二个rosbag文件路径
plot_error_data(file1, file2)
