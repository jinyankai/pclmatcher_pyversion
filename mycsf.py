import rosbag
import numpy as np
import pandas as pd
import os


# 计算MSE
def calculate_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


# 读取并处理一个rosbag文件
def process_rosbag_file(bag_file):
    total_mse = 0.0  # 总MSE
    total_count = 0  # 数据对数

    # 打开rosbag文件
    bag = rosbag.Bag(bag_file)

    # 用于存储x值
    error_=[]

    try:
        # 遍历rosbag中的消息
        for topic, msg, t in bag.read_messages(topics=['/error']):
            # 提取消息中的x值
            x = msg.data
            error_.append(x)
            # 计算MSE
            total_mse += calculate_mse(x, 0.0)
            total_count += 1
            

    except Exception as e:
        print(f"Error processing bag file {bag_file}: {e}")
    finally:
        bag.close()

    return total_mse, total_count


# 处理文件夹中的所有rosbag文件
def process_rosbag_folder(folder_path):
    all_mse_results = []  # 用来保存所有文件的MSE结果

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.bag'):  # 只处理rosbag文件
            bag_file_path = os.path.join(folder_path, filename)
            print(f"Processing {bag_file_path}...")
            total_mse, total_count = process_rosbag_file(bag_file_path)

            # 计算平均MSE并保存结果
            if total_count > 0:
                avg_mse = total_mse / total_count
                all_mse_results.append({'bag_file': bag_file_path, 'avg_mse': avg_mse})

    # 将所有结果保存到一个新的CSV文件
    mse_df = pd.DataFrame(all_mse_results)
    mse_df.to_csv('mse_result.csv', index=False)
    print("MSE results have been saved to mse_result.csv.")


# 设置文件夹路径并调用函数
folder_path = '/'  # 请替换为实际文件夹路径
process_rosbag_folder(folder_path)
