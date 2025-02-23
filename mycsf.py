import rosbag
import numpy as np
import pandas as pd
import os

# 计算MSE
def calculate_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# 读取并处理一个rosbag文件
def process_rosbag_file(bag_file):
    mse_results = []  # 保存每个话题的MSE结果

    # 打开rosbag文件
    bag = rosbag.Bag(bag_file)

    # 用于存储x值
    ground_truth_x = []
    target_x = []

    try:
        # 遍历rosbag中的消息
        for topic, msg, t in bag.read_messages(topics=['/ground_truth/state_triangle_link_base', '/target']):
            # 提取消息中的x值
            if topic == '/ground_truth/state_triangle_link_base':
                x_value = msg.pose.pose.position.x
                ground_truth_x.append(x_value)
            elif topic == '/target':
                x_value = msg.pose.pose.position.x
                target_x.append(x_value)

        # 计算MSE
        if ground_truth_x and target_x:
            mse = calculate_mse(np.array(ground_truth_x), np.array(target_x))
            mse_results.append({'bag_file': bag_file, 'mse': mse})

    except Exception as e:
        print(f"Error processing bag file {bag_file}: {e}")
    finally:
        bag.close()

    return mse_results

# 处理文件夹中的所有rosbag文件
def process_rosbag_folder(folder_path):
    all_mse_results = []  # 用来保存所有文件的MSE结果

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.bag'):  # 只处理rosbag文件
            bag_file_path = os.path.join(folder_path, filename)
            print(f"Processing {bag_file_path}...")
            mse_results = process_rosbag_file(bag_file_path)
            all_mse_results.extend(mse_results)  # 将当前文件的结果添加到总结果中

    # 将所有结果保存到一个新的CSV文件
    mse_df = pd.DataFrame(all_mse_results)
    mse_df.to_csv('mse_result.csv', index=False)
    print("MSE results have been saved to mse_result.csv.")

# 设置文件夹路径并调用函数
folder_path = '/path/to/your/folder'  # 请替换为实际文件夹路径
process_rosbag_folder(folder_path)
