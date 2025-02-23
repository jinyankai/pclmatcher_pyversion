import os
import rosbag
import csv
import rospy

def convert_bag_to_csv(bag_file, output_folder):
    """
    将 rosbag 文件中的所有话题提取并转换为 CSV 文件
    :param bag_file: 输入的 rosbag 文件路径
    :param output_folder: 输出的 CSV 文件存储文件夹
    """
    # 提取文件名并构造输出文件的路径
    bag_filename = os.path.basename(bag_file)
    csv_filename = os.path.splitext(bag_filename)[0] + '.csv'
    csv_filepath = os.path.join(output_folder, csv_filename)

    # 打开 rosbag 文件
    bag = rosbag.Bag(bag_file)
    
    # 打开 CSV 文件进行写入
    with open(csv_filepath, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Topic', 'Timestamp', 'Message'])  # CSV 文件的头部

        # 遍历 rosbag 中的每个消息
        for topic, msg, t in bag.read_messages():
            # 写入话题名，时间戳和消息内容
            writer.writerow([topic, t.to_sec(), str(msg)])

    bag.close()
    rospy.loginfo(f"文件 {csv_filepath} 已保存！")

def convert_all_bags_in_folder(input_folder, output_folder):
    """
    读取指定文件夹中的所有 bag 文件并转换为 csv 文件
    :param input_folder: 存放 .bag 文件的文件夹
    :param output_folder: 存放 .csv 文件的文件夹
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 .bag 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".bag"):
            bag_file = os.path.join(input_folder, filename)
            convert_bag_to_csv(bag_file, output_folder)

if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = "/path/to/your/bag/files"  # 替换成包含 .bag 文件的文件夹路径
    output_folder = "/path/to/save/csv"  # 替换成保存 .csv 文件的文件夹路径

    convert_all_bags_in_folder(input_folder, output_folder)
    print("所有文件转换完成！")
