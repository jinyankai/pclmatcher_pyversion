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

# 绘制两个rosbag文件中的/error话题数据
def plot_error_data(file1, file2):
    # 提取两个rosbag文件中的/error数据
    timestamps1, error_values1 = extract_error_data(file1)
    timestamps2, error_values2 = extract_error_data(file2)

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
file1 = '/path/to/your/first_file.bag'  # 替换为第一个rosbag文件路径
file2 = '/path/to/your/second_file.bag'  # 替换为第二个rosbag文件路径
plot_error_data(file1, file2)
