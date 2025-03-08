
from communication.Messager import Messager
from detect.Detector import Detector
from detect.Video import Video
from detect.Capture import Capture
# from Lidar.Lidar import Lidar
from Lidar.Converter import Converter , ROISelector
from Log.Log import RadarLog
from Car.Car import *
import numpy as np
import cv2
import time
import open3d as o3d
from collections import deque
from ruamel.yaml import YAML
import os

mode = "video" # "video" or "camera" , 如果纯视频模式选用video,需要播放录制livox mid-70的rosbag获得点云信息
save_video = False # 是否保存视频

if __name__ == '__main__':
    video_path = "./data/video.avi" # 请改为/path/to/video.avi
    detector_config_path = "./configs/detector_config.yaml"
    binocular_camera_cfg_path = "./configs/bin_cam_config.yaml"
    main_config_path = "./configs/main_config.yaml"
    converter_config_path = "./configs/converter_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))
    # 全局变量
    global_my_color = main_cfg['global']['my_color']
    is_debug = main_cfg['global']['is_debug']

    # 设置保存路径
    save_video_folder_path = "/home/nvidia/RadarWorkspace/code/Radar_Develop/data/train_record/"  # 保存视频的文件夹
    today = time.strftime("%Y%m%d", time.localtime()) # 今日日期，例如2024年5月6日则为20240506
    today_video_folder_path = save_video_folder_path + today + "/" # 今日的视频文件夹
    if not os.path.exists(today_video_folder_path): # 当天的视频文件夹不存在则创建
        os.makedirs(today_video_folder_path)
    video_name = time.strftime("%H%M%S", time.localtime()) # 视频名称，以时分秒命名，19：29：30则为192930
    video_save_path = today_video_folder_path + video_name + ".mp4" # 视频保存路径

    logger = RadarLog("main")

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码器
        out = cv2.VideoWriter(video_save_path, fourcc, 24, (1920, 1080))  # 文件名，编码器，帧率，帧大小
    # 传递的绘值队列
    draw_queue = deque(maxlen=10)
    # 类初始化
    detector = Detector(detector_config_path)
    # lidar = Lidar(main_cfg)
    converter = Converter(global_my_color,converter_config_path)  # 传入的是path
    carList = CarList(main_cfg)
    logger.log("carList init")

    messager = Messager(main_cfg , draw_queue)
    logger.log("messager init")




    if mode == "video":
        capture = Video(video_path)
    elif mode == "camera":
        capture = Capture(binocular_camera_cfg_path,"new_cam")
    else:
        print("mode error")
        exit(1)


    # 场地解算初始化
    converter.camera_to_field_init(capture)

    # ROI初始化
    # roi_selector = ROISelector(capture)

    start_time = time.time()
    # fps计算
    N = 10
    fps_queue = deque(maxlen=10)


    # 开启激光雷达线程

    detector.create(capture)
    detector.start()
    # lidar.start()
    messager.start()

    # 创建一个空列表来存储所有检测的结果
    all_detections = []

    # 当前帧ID
    frame_id = 1
    counter = 0

    # 可视化小地图绘制queue


    print("enter main loop")
    try:
        # 主循环
        while True:

            # 计算fps
            now = time.time()
            fps = 1 / (now - start_time)
            start_time = now
            # 将FPS值添加到队列中
            fps_queue.append(fps)
            # 计算平均FPS
            avg_fps = sum(fps_queue) / len(fps_queue)

            print("fps:",avg_fps)

            # 获得推理结果
            infer_result = detector.get_results()

            # 需要打包一份给carList
            carList_results = []
            result_img = None


            # 确保推理结果不为空且可以解包
            if infer_result is not None:
                # print(infer_result)
                result_img, results = infer_result

                if results is not None:
                    print("results is not none")
                    # if lidar.pcdQueue.point_num == 0:
                    #     print("no pcd")
                    #     continue
                    # 
                    # pc_all = lidar.get_all_pc()
                    # 
                    # 
                    # # 创建总体点云pcd
                    # pcd_all = o3d.geometry.PointCloud()
                    # pcd_all.points = o3d.utility.Vector3dVector(pc_all)
                    # 
                    # 
                    # # 将总体点云转到相机坐标系下
                    # converter.lidar_to_camera(pcd_all)
                    # 
                    # # 检测框对应点云
                    # box_pcd = o3d.geometry.PointCloud()

                    # 对每个结果进行分析 , 进行目标定位
                    for result in results:

                        # 对每个检测框进行处理，获取对应点云
                        # 结果：[xyxy_box, xywh_box , track_id , label ]
                        xyxy_box, xywh_box ,  track_id , label = result # xywh的xy是中心点的xy

                        # 如果没有分类出是什么车，或者是己方车辆，直接跳过
                        if label == "NULL":
                            continue
                        if global_my_color == "Red" and carList.get_car_id(label) < 100 and carList.get_car_id(label) != 7:
                            continue
                        if global_my_color == "Blue" and carList.get_car_id(label) > 100 and carList.get_car_id(label) != 107:
                            continue


                        # 获取新xyxy_box , 原来是左上角和右下角，现在想要中心点保持不变，宽高设为原来的一半，再计算一个新的xyxy_box,可封装
                        div_times = 1.01
                        new_w = xywh_box[2] / div_times
                        new_h = xywh_box[3] / div_times
                        new_xyxy_box = [xywh_box[0] - new_w / 2, xywh_box[1] - new_h / 2, xywh_box[0] + new_w / 2, xywh_box[1] + new_h / 2]
                        center = converter.detection_main(new_xyxy_box, pcd_all)
                        distance = converter.get_distance(center)
                        # # 获取检测框内numpy格式pc
                        # box_pc = converter.get_points_in_box(pcd_all.points, new_xyxy_box)
                        # print(len(box_pc))
                        # # 如果没有获取到点，直接continue
                        # if len(box_pc) == 0:
                        #     print("no points in box")
                        #     continue
                        # 
                        # box_pcd.points = o3d.utility.Vector3dVector(box_pc)
                        # 
                        # # 点云过滤
                        # box_pcd = converter.filter(box_pcd)
                        # 
                        # # 获取box_pcd的中心点
                        # cluster_result = converter.cluster(box_pcd) # 也就6帧变7帧，还是启用
                        # 
                        # _, center = cluster_result
                        # # # 如果聚类结果为空，则用中值取点
                        # if center[0] == 0 and center[1] == 0 and center[2] == 0:
                        #     center = converter.get_center_mid_distance(box_pcd)
                        # 
                        # 
                        # # 计算距离
                        # distance = converter.get_distance(center)
                        if distance == 0:
                            continue

                        # 将点转到赛场坐标系下
                        field_xyz = center
                        # 计算赛场坐标系下的距离
                        field_distance = converter.get_distance(field_xyz)

                        # 在图像上写距离,位置为xyxy_box的左上角,可以去掉
                        if is_debug:
                            cv2.putText(result_img, "distance: {:.2f}".format(field_distance), (int(xyxy_box[0]), int(xyxy_box[1]),), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
                            cv2.putText(result_img, "x: {:.2f}".format(field_xyz[0])+"y:{:.2f}".format(field_xyz[1])+"z:{:.2f}".format(field_xyz[2]), (int(xyxy_box[2]), int(xyxy_box[3]+10),),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)

                        # 将结果打包
                        carList_results.append([track_id , carList.get_car_id(label) , xywh_box , 1 , center , field_xyz])

                    # 将结果传入carList
            carList.update_car_info(carList_results)


            all_infos = carList.get_all_info() # 此步不做trust的筛选，留给messager做
            my_car_infos = []
            enemy_car_infos = []
            # result in results:[car_id , center_xy , camera_xyz , field_xyz]
            # 如果是我方车辆，找到所有敌方车辆，计算与每一台敌方车辆距离，并在图像两车辆中心点之间画线，线上写距离
            for all_info in all_infos:
                track_id , car_id , center_xy , camera_xyz , field_xyz , color , is_valid = all_info


                # 将信息分两个列表存储
                if color == global_my_color:
                    if track_id == -1:
                        continue
                    my_car_infos.append(all_info)
                else:
                    enemy_car_infos.append(all_info)
                    if track_id != -1:
                        # 将每个检测结果添加到列表中，增加frame_id作为每一帧的ID
                        all_detections.append([frame_id] + list(all_info))

            # 通信
            messager.update_enemy_car_infos(enemy_car_infos)


            # 画线
            for my_car_info in my_car_infos:
                my_track_id , my_car_id , my_center_xy , my_camera_xyz , my_field_xyz , my_color , my_is_valid= my_car_info
                # 将相机的xyz坐标点投影到图像上，并画一个红色的点
                if is_debug:
                    my_camera_xyz = my_camera_xyz.reshape(1, -1)
                    my_reprojected_point = converter.camera_to_image(my_camera_xyz)[0] # u,v是图像坐标系下的坐标
                    cv2.circle(result_img, (int(my_reprojected_point[0]), int(my_reprojected_point[1])), 5, (0, 0, 255), -1)
                if my_car_id == carList.sentinel_id and my_is_valid:
                    # 记录符合距离要求的距离最近的车
                    min_distance_car_id = -1
                    min_distance = 1000
                    min_distance_angle = -1
                    for enemy_car_info in enemy_car_infos:
                        enemy_track_id , enemy_car_id , enemy_center_xy , enemy_camera_xyz , enemy_field_xyz , enemy_color , enemy_is_valid= enemy_car_info
                        # 如果不可信，跳过
                        if not enemy_is_valid or enemy_track_id == -1: # 不可信或未初始化
                            continue
                        # 将相机的xyz坐标点投影到图像上，并画一个红色的点
                        if is_debug:
                            enemy_camera_xyz = enemy_camera_xyz.reshape(1, -1)
                            enemy_reprojected_point = converter.camera_to_image(enemy_camera_xyz)[0]  # u,v是图像坐标系下的坐标
                        # 计算距离
                        distance = np.linalg.norm(np.array(my_field_xyz) - np.array(enemy_field_xyz))
                        if is_debug:
                            cv2.circle(result_img, (int(enemy_reprojected_point[0]), int(enemy_reprojected_point[1])), 5,
                                       (0, 0, 255), -1)
                            # 画线,从我方车辆中心点到敌方车辆中心点
                            cv2.line(result_img, (int(my_reprojected_point[0]), int(my_reprojected_point[1])),
                                     (int(enemy_reprojected_point[0]), int(enemy_reprojected_point[1])), (0, 255, 122), 2)
                            # 写距离
                            cv2.putText(result_img, "distance: {:.2f}".format(distance), (
                                int((my_center_xy[0] + enemy_center_xy[0]) / 2),
                                int((my_center_xy[1] + enemy_center_xy[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                        (0, 255, 122),
                                        2)
                        # 判断距离是否符合
                        if distance < carList.sentinel_min_alert_distance or distance > carList.sentinel_max_alert_distance:
                            continue

                        if distance < min_distance:
                            # 计算角度，设赛场x轴正方向为0度，顺时针为正
                            angle = np.arctan2(enemy_field_xyz[1] - my_field_xyz[1], enemy_field_xyz[0] - my_field_xyz[0]) * 180 / np.pi
                            min_distance = distance
                            min_distance_angle = angle
                            min_distance_car_id = enemy_car_id
                    # 在哨兵重投影点上写上最近预警车辆的id，距离和角度
                    if min_distance_car_id != -1:
                        if is_debug:
                            cv2.putText(result_img, "id: {}".format(min_distance_car_id), (int(my_reprojected_point[0]), int(my_reprojected_point[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
                            cv2.putText(result_img, "distance: {:.2f}".format(min_distance), (int(my_reprojected_point[0]), int(my_reprojected_point[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
                            cv2.putText(result_img, "angle: {:.2f}".format(min_distance_angle), (int(my_reprojected_point[0]), int(my_reprojected_point[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
                        # 将角度转为象限 ， carID , distance , quadrant
                        quadrant = converter.angle_to_quadrant(min_distance_angle)
                        # zip
                        sentinel_alert_info = [min_distance_car_id, min_distance, quadrant]
                        messager.update_sentinel_alert_info(sentinel_alert_info)





            if result_img is None:
                print("result_img is none")
                continue
            if is_debug:
                cv2.putText(result_img, "fps: {:.2f}".format(avg_fps), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122),
                        2)
                result_img = cv2.resize(result_img, (1920, 1080))
            if save_video:
                out.write(result_img)

            if is_debug: # 用于调试
                cv2.imshow("frame", result_img) # 不加3帧
            frame_id += 1
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("keyboard")
        pass
    finally: #确保程序结束时关闭所有线程

        print("finally")

        cv2.destroyAllWindows()
        if save_video:
            if out is not None:
                out.release()
        detector.stop_save_video()
        print(1)
        detector.stop()
        print(2)
        # detector.release()
        print(3)
        capture.release()
        print(4)

        print(4.5)
        messager.receiver.stop()

        print(5)
        messager.stop()
        print(6)
        lidar.stop()
        print(7)
