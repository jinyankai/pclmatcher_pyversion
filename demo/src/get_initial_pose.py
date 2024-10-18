import cv2
import numpy as np
import yaml
import open3d as o3d
from Capture import Capture
from camera.Python.MvImport import MvCameraControl_class as hk
#Tinit = np.array([4.89800383e-04, -9.99968337e-01, 7.98446536e-03, 5.30701329e-02,-0.0173332 , -0.00799176, -0.999818 , 0.0880809,9.99849635e-01, 3.51240220e-04, -1.73365401e-02, 8.69298613e-02,0., 0., 0., 1.]).reshape(4,4)

def __callback_1(event, x, y, flags, param):
    '''
    鼠标回调函数
    鼠标点击点：确认标定点并在图像上显示
    鼠标位置：用来生成放大图
    '''
    # using EPS and MAX_ITER combine
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)
    if event == cv2.EVENT_MOUSEMOVE:
        # 周围200*200像素放大图
        rect = cv2.getWindowImageRect(param["pick_winname"])
        img_cut = np.zeros((200, 200, 3), np.uint8)
        img_cut[max(-y + 100, 0):min(param["pick_img"].shape[0] + 100 - y, 200),
        max(-x + 100, 0):min(param["pick_img"].shape[1] + 100 - x, 200)] = \
            param["pick_img"][max(y - 100, 0):min(y + 100, param["pick_img"].shape[0]),
            max(x - 100, 0):min(x + 100, param["pick_img"].shape[1])]
        cv2.circle(img_cut, (100, 100), 1, (0, 255, 0), 1)
        cv2.imshow(param["zoom_winname"], img_cut)
        cv2.moveWindow(param["zoom_winname"], rect[0] - 400, rect[1] + 200)
        cv2.resizeWindow(param["zoom_winname"], 400, 400)
    if event == cv2.EVENT_LBUTTONDOWN and not param["pick_flag"]:
        param["pick_flag"] = True
        print(f"pick ({x:d},{y:d})")
        # 亚像素精确化
        corner = cv2.cornerSubPix(param["pick_img_raw"], np.float32([x, y]).reshape(1, 1, 2), (5, 5), (-1, -1),
                                  stop_criteria).reshape(2)
        param["pick_point"] = [corner[0], corner[1]]
        cv2.circle(param["pick_img"], (x, y), 2, (0, 255, 0), 1)

class Initial_pose:
    def __init__(self,data_loader_path = 'config/main_cofig.yaml'):
        with open(data_loader_path,'r') as f:
            data_loader = yaml.safe_load(f)
        self.Tinit = np.array(data_loader['Tinit']).reshape(4,4)#lidar extrinsic
        # 获取R和T，并将它们转换为NumPy数组
        self.R = np.array(data_loader['calib']['extrinsic']['R']['data']).reshape(
            (data_loader['calib']['extrinsic']['R']['rows'], data_loader['calib']['extrinsic']['R']['cols']))
        self.T = np.array(data_loader['calib']['extrinsic']['T']['data']).reshape(
            (data_loader['calib']['extrinsic']['T']['rows'], data_loader['calib']['extrinsic']['T']['cols']))
        # 获取相机内参
        self.cx = data_loader['calib']['intrinsic']['cx']
        self.cy = data_loader['calib']['intrinsic']['cy']
        self.fx = data_loader['calib']['intrinsic']['fx']
        self.fy = data_loader['calib']['intrinsic']['fy']
        self.max_depth = data_loader['params']['max_depth']
        self.width = data_loader['params']['width']
        self.height = data_loader['params']['height']
        # 去畸变参数
        self.distortion_matrix = np.array(data_loader['calib']['distortion']['data'])
        # 相机坐标系到图像坐标系的内参矩阵，3*3的矩阵
        self.intrinsic_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        # 图像坐标系到相机坐标系的内参矩阵，3*3的矩阵
        self.intrinsic_matrix_inv = np.linalg.inv(self.intrinsic_matrix)
        # 激光雷达到相机的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.extrinsic_matrix = np.hstack((self.R, self.T))
        self.extrinsic_matrix = np.vstack((self.extrinsic_matrix, [0, 0, 0, 1]))
        # 相机到激光雷达的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)
        self.initial_pose = None
        self.camera_extrinsic = None
        self.cap = Capture()
    def open_cam(self):
        self.cap = Capture()
        self.cap.get_camera()

    def get_image(self):
        return self.cap.get_frame()



    def set_target(self,target_loader_path):
        with open(target_loader_path, 'r') as f:
            data_loader = yaml.safe_load(f)

    def is_open(self,frame):
        return frame == hk.MV_OK

    def locate_pick(self, camera_type, home_size=False, video_test=False):
        '''
        手动四点标定

        :param cap:Camera_Thread object
        :param enemy:enemy number
        :param camera_type:camera number
        :param home_size: 选用在家里测试时的尺寸
        :param video_test: 是否用视频测试，以减慢播放速度

        :return: 读取成功标志，旋转向量，平移向量
        '''

        # 窗口下方提示标定哪个目标
        #tips = \
        #   {
        #       '00': ['red_base', 'blue_outpost', 'b_right_top', 'b_left_top'],
        #        '01': ['red_outpost', 'red_base', 'b_right_top', 'b_left_top'],
        #        '10': ['blue_base', 'red_outpost', 'r_right_top', 'r_left_top'],
        #        '11': ['blue_outpost', 'blue_base', 'r_righttop', 'r_left_top'],
        #   }
        #设定世界坐标
        frame = self.get_image()
        world_ = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # 标定目标提示位置
        tip_w = frame.shape[1] // 2
        tip_h = frame.shape[0] - 200

        # OpenCV窗口参数
        info = {}
        info["pick_img"] = frame
        info["pick_img_raw"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        info["pick_winname"] = "pick_corner"
        info["zoom_winname"] = "zoom_in"
        info["pick_flag"] = False
        info["pick_point"] = None  # 回调函数中点击的点

        cv2.namedWindow(info["pick_winname"], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(info["pick_winname"], 1280, 780)
        cv2.setWindowProperty(info["pick_winname"], cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(info["pick_winname"], 500, 300)
        cv2.namedWindow(info["zoom_winname"], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(info["zoom_winname"], 400, 400)
        cv2.setWindowProperty(info["zoom_winname"], cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback("pick_corner", __callback_1(), info)

        pick_point = []
        while True:
            # draw tips
            #cv2.putText(frame, tips[str(enemy) + str(camera_type)][len(pick_point)], (tip_w, tip_h),
            #           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

            # draw the points having been picked
            for select_p in pick_point:
                cv2.circle(frame, (int(select_p[0]), int(select_p[1])), 1, (0, 255, 0), 2)

            # draw the connecting line following the picking order
            for p_index in range(1, len(pick_point)):
                cv2.line(frame, (int(pick_point[p_index - 1][0]), int(pick_point[p_index - 1][1])),
                         (int(pick_point[p_index][0]), int(pick_point[p_index][1])), (0, 255, 0), 2)

            cv2.imshow(info["pick_winname"], info["pick_img"])

            if info["pick_flag"]:  # 当在回调函数中触发点击事件
                pick_point.append(info["pick_point"])
                # draw the points having been picked
                for select_p in pick_point:
                    cv2.circle(frame, (int(select_p[0]), int(select_p[1])), 1, (0, 255, 0), 2)

                # draw the connecting line following the picking order
                for p_index in range(1, len(pick_point)):
                    cv2.line(frame, (int(pick_point[p_index - 1][0]), int(pick_point[p_index - 1][1])),
                             (int(pick_point[p_index][0]), int(pick_point[p_index][1])), (0, 255, 0), 2)
                # 四点完成，首尾相连
                if len(pick_point) == 4:
                    cv2.line(frame, (int(pick_point[3][0]), int(pick_point[3][1])),
                             (int(pick_point[0][0]), int(pick_point[0][1])), (0, 255, 0), 2)

                cv2.imshow(info["pick_winname"], info["pick_img"])
                # 将刚加入的pop出等待确认后再加入
                pick_point.pop()
                key = cv2.waitKey(0)
                if key == ord('c') & 0xFF:  # 确认点加入
                    pick_point.append(info["pick_point"])

                    print(f"You have pick {len(pick_point):d} point.")

                if key == ord('z') & 0xFF:  # 将上一次加入的点也删除（这次的也不要）
                    if len(pick_point):
                        pick_point.pop()
                    print("drop last")

                if key == ord('q') & 0xFF:  # 直接退出标定，比如你来不及了
                    cv2.destroyWindow(info["pick_winname"])
                    cv2.destroyWindow(info["zoom_winname"])
                    return False, None, None
                info["pick_flag"] = False
            else:
                # 当未点击时，持续输出视频
                if video_test:
                    cv2.waitKey(80)
                else:
                    cv2.waitKey(1)
            if len(pick_point) == 4:  # 四点全部选定完成，进行PNP
                break
            frame = self.get_image()
            if not self.is_open(frame):
                cv2.destroyWindow(info["pick_winname"])
                cv2.destroyWindow(info["zoom_winname"])
                return False, None, None
            info["pick_img"] = frame

        pick_point = np.float64(pick_point).reshape(-1, 1, 2)
        flag, rvec, tvec = cv2.solvePnP(world_, pick_point,cameraMatrix=self.intrinsic_matrix,distCoeffs=self.distortion_matrix , flags=cv2.SOLVEPNP_P3P)
        cv2.destroyWindow(info["pick_winname"])
        cv2.destroyWindow(info["zoom_winname"])
        self.camera_extrinsic = np.hstack((rvec, tvec))
        self.camera_extrinsic = np.vstack((self.camera_extrinsic, np.array([0, 0, 0, 1])))
        return flag, rvec, tvec

    def img_to_lidar(self):
        camera_extrinsic = self.camera_extrinsic
        camera_to_lidar = self.extrinsic_matrix_inv
        self.initial_pose = np.dot(camera_to_lidar, camera_extrinsic)
        return self.initial_pose



