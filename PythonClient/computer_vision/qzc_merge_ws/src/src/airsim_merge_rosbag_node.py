# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
''' script to merge msg to rosbag '''
# import setup_path
import airsim

import pprint
import tempfile
import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
from time import sleep
import rosbag
import rospy
import glob
import os
from cv_bridge import CvBridge
import tqdm
import numpy as np
import copy

from rosbag import bag

# tf
import tf
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
from sensor_msgs.msg import CameraInfo

import json
import open3d as o3d
from scipy.spatial.transform import Rotation
# https://vimsky.com/examples/usage/python-scipy.spatial.transform.Rotation.html
# from Perception.msg import *
# from airsim_merge_rosbag.msg import *
from perception_msg.msg import *


# we check if there is msg with same time stamp with msg in bag
# if match we write the msg just after the msg in bag

def read_tform(data, key):
    R = data[key]['R']
    R = np.matrix(R).reshape(3, 3)
    t = data[key]['T']
    t = np.matrix(t).reshape(3, 1)
    T = np.concatenate((R, t), axis=1)
    T = np.concatenate((T, np.matrix([[0, 0, 0, 1]])), axis=0)
    return T

import pandas as pd
CLASS_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
                'furniture', 'objects', 'picture', 'sofa', 'table',
                'tv', 'wall', 'window','base-cabinet','countertop', 'void_color']

FOCUSED_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
                'furniture', 'objects', 'picture', 'sofa', 'table',
                'tv', 'wall', 'window', 'base-cabinet','countertop','void_color']

CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                (137, 28, 157), (150, 255, 255), (54, 114, 113),
                (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                (255, 150, 255), (255, 180, 10), (101, 70, 86),
                (38, 230, 0), (154,229,206), (255,0,0),(0,0, 0)]

CLASS_DICT = dict(zip(CLASS_NAMES, CLASS_COLORS))

OBJECT_OF_INTEREST_CLASSES_OUTDOOR = [
    "ground",
    # "trafficLight",
    "streetLight",
    "manhole",
    "fixLight",
    "garbageCan",
    "bench",
    "fireHydrant",
    "chair",
    "table",
    "CameraActor",
    "sewerGrate",
    "tableUmbrella",
    "electricalBox",
    "Awning",
    "PowerLine",
    "crosswalkSign",
    "Building",
    "metalPillar",
    "tree",
    "sky",

    # other
    "chainlinkFence",
    "tarp",
    "vent",
    "door",
    "ac",
    "StoreSign",
    "pylong_Sml",
    "grass",
    "parkingMeter",
    # "hedge_Short",
    "hedge",
    "metalFence",
    "barrier_Lrg",
    "TrashBag",
    "potSquare",
    "cementBarrier",
    "fern",
    "flag",
    "sign2"
]

# https://tool.lu/color/
# rgb
CLASS_COLORS_OUTDOOR = [
    (135, 169, 180), # 浅灰色
    (112, 105, 191), # 深紫色
    (89, 121, 72), # 深绿色
    (190, 225, 64), # 深绿色-2
    (206, 190, 59), # 深黄色-2
    (81, 13, 36), # 深粉色
    (115, 176, 195), # 深蓝色-2
    (161, 171, 27), #深绿色-3
    (153, 108, 6), # 深米黄色
    (29, 26, 199), # 深蓝色
    (102, 16, 239), # 深紫色
    (242, 107, 146), # 老红色
    (156, 198, 23), # 浅绿色
    (49, 89, 160), #浅蓝色+2
    (68, 218, 116), # 深绿色-3
    (196, 30, 8), # 红色
    (121, 67, 28), # 棕色
    (0, 53, 65), # 蓝绿色
    (11, 236, 9), # 绿色
    (54, 72, 205), # 蓝色
    (146, 52, 70), # 红色-3
    (226, 149, 143), # 红色-2
    (151, 126, 171), # 浅紫色-2
    (194, 39, 7), # 红色2
    (205, 120, 161), # 红色2-2
    (212, 51, 60), # 红色3
    (103, 252, 157), # 绿色2
    (211, 80, 208), # 粉色
    (195, 237, 132), # 绿色+3
    (189, 135, 188), # 粉色+3
    (124, 21, 123), # 紫色2
    (19, 132, 69), # 绿色3
    (94, 253, 175), # 绿色-3
    (90, 162, 242), # 蓝色-1
    (182, 251, 87), # 绿色+2
    (199, 29, 1), # 红色3
    (254, 12, 229) # 粉色-4
]

CLASS_DICT_OUTDOOR = dict(zip(OBJECT_OF_INTEREST_CLASSES_OUTDOOR, CLASS_COLORS_OUTDOOR))


# Get the default directory for AirSim
airsim_path = os.path.join(os.path.expanduser('~'), 'Documents', 'AirSim')

# Load the settings file
with open(os.path.join(airsim_path, 'settings.json'), 'r') as fp:
    data = json.load(fp)

# Get the camera intrinsics
capture_settings = data['CameraDefaults']['CaptureSettings'][0]
img_width = capture_settings['Width']
img_height = capture_settings['Height']
img_fov = capture_settings['FOV_Degrees']

# Compute the focal length
fov_rad = img_fov * np.pi/180
g_fd = (img_width/2.0) / np.tan(fov_rad/2.0)

def DepthConversion(PointDepth, f):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]
    i_c = float(H) / 2 - 1
    j_c = float(W) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
    return PlaneDepth.astype(np.float32)

class SemanticLabelWithRGBDirMessages:
    def __init__(self, folder, topic_name, frame_id):
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.D = {}
        self.instance_id_to_name = self.read_id_to_name_csv()
        self.object_name_dict = self.read_name_to_color_csv()
        self.map_to_class_labels = np.vectorize(
            # lambda x: self.object_name_dict.get(self.instance_id_to_name.get(x, "void_color"))
            lambda x: self.object_name_dict.get(self.instance_id_to_name.get(x, 0), CLASS_DICT["void_color"])
        )
        self.all_occur_label = {}
        self.debug_cnt1 = 0

        debug1 = False
        tmpMap = {}
        for filename in glob.glob(folder+'/*_label.png'):
            # debug : print all used label
            if debug1 == True:
                gray = cv2.imread(filename, 0)
                rows, cols = gray.shape
                for i in range(rows):
                    for j in range(cols):
                        pixel = gray[i, j]
                        if self.instance_id_to_name.get(pixel, 0) == 0:
                            continue
                        tmpMap[self.instance_id_to_name.get(pixel, 0)] = tmpMap.get(self.instance_id_to_name.get(pixel, 0), 0) + 1
            # debug : print all used label
            k = os.path.basename(filename).split('_label')[0]
            self.D[k] = filename

        if debug1 == True:
            for name, num in tmpMap:
                print("name %s, %d".format(name, num))

        print('topic: {} frame_id {} size {}'.format(self.topic_name, self.frame_id, len(self.D)))

        return

    def read_id_to_name_csv(self):
        df = pd.read_table('labels_ade20k.txt', header=None, encoding='utf-8')
        df.columns=['name']
        instance_id_to_name = {}
        for i in range(len(df)):
            instance_id_to_name[i] = df['name'][i]
        return instance_id_to_name

    def read_name_to_color_csv(self):
        # TODO aqiu : change to args
        df = pd.read_csv('semantic_with_colors_zed.csv', header=0, sep=',',
                         usecols=[0, 1, 2, 3, 4], encoding='utf-8')
        print(df.head(3))
        print(df.columns[0])
        print(df.shape)
        print(df['name'])

        class_name = list(df['name'])
        class_colors = list(zip(df['red'], df['green'], df['blue']))
        class_dict = dict(zip(class_name, class_colors))

        print("read finish!")
        return class_dict


    def compose_msg(self, filename, sec, nsec):
        gray = cv2.imread(filename, 0)
        # img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mask = np.unique(gray)

        # tmp = {}
        for v in mask:
            old_num = self.all_occur_label.get(v, 0)
            self.all_occur_label[v] = old_num+1
            #tmp[v] = np.sum(data == v)
        if (self.debug_cnt1 % 50 == 0):
            print(self.all_occur_label)
        self.debug_cnt1 = self.debug_cnt1 + 1

        semantic = np.asarray(self.map_to_class_labels(gray))
        semantic = np.stack((semantic[0], semantic[1], semantic[2]), axis=2)
        semantic = semantic.astype(np.uint8)

        msg = CvBridge().cv2_to_imgmsg(semantic, "rgb8")
        msg.header.stamp.secs = sec
        msg.header.stamp.nsecs = nsec
        msg.header.frame_id = self.frame_id

        return msg

    # return None if not match, otherwise return a msg and topic name
    def check_match(self, sec, nsec):
        k = '{}.{}'.format(sec, nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            return self.topic_name, self.compose_msg(filename, sec,nsec)
        return None

class SemanticLabelWithRGBOutdoorDirMessages:
    def __init__(self, folder, topic_name, frame_id):
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.D = {}
        # for filename in glob.glob(folder+'/*_label.png'):
        #     k = os.path.basename(filename).split('_label')[0]
        #     self.D[k] = filename

        self.id_color_dict={}
        for key, val in zip(OBJECT_OF_INTEREST_CLASSES_OUTDOOR, range(len(CLASS_COLORS_OUTDOOR))):
            self.id_color_dict[val] = CLASS_COLORS_OUTDOOR[val]
        self.id_color_dict[255]=(0,0,0)

        self.map_label_to_color = np.vectorize(
            # lambda x: self.object_name_dict.get(self.instance_id_to_name.get(x, "void_color"))
            lambda x: self.id_color_dict.get(x, self.id_color_dict[255])
        )

        self.semantic_rgb_path = ""
        save_semantic_rgb_image = False
        if save_semantic_rgb_image:
            self.semantic_rgb_path = os.path.join(folder, "..", str(20))
            try:
                os.makedirs(self.semantic_rgb_path)
            except OSError:
                pass

        # airsim
        for filename in glob.glob(folder+'/*.png'):
            k = os.path.basename(filename).split('_2.png')[0]
            k =k[:-3]+'.'+k[-3:]
            self.D[k] = filename

        print('folder: {} topic: {} frame_id {} size {}'.format(folder, self.topic_name, self.frame_id, len(self.D)))
        return

    def compose_msg(self, filename, sec, nsec):
        use_ir_image = False # !!! shoud chang with other variable
        if use_ir_image == False:
            img = cv2.imread(filename)
            msg = CvBridge().cv2_to_imgmsg(img,"bgr8")
            # msg = CvBridge().cv2_to_imgmsg(img,"rgb8")

            msg.header.stamp.secs = sec
            msg.header.stamp.nsecs = nsec
            msg.header.frame_id = self.frame_id
            return msg
        else:
            gray = cv2.imread(filename, 0)
            if gray is None:
                print(f"sec: {sec} nsec:{nsec}")

            semantic = np.asarray(self.map_label_to_color(gray))
            semantic = np.stack((semantic[0], semantic[1], semantic[2]), axis=2)
            semantic = semantic.astype(np.uint8)

            # msg = CvBridge().cv2_to_imgmsg(semantic, "bgr8")
            msg = CvBridge().cv2_to_imgmsg(semantic, "rgb8")

            if self.semantic_rgb_path:
                nsecs = round(nsec*1e-6)
                if nsecs < 10:
                    k = '{}00{}'.format(sec,nsecs)
                elif nsecs < 100:
                    k = '{}0{}'.format(sec,nsecs)
                else:
                    k = '{}{}'.format(sec,nsecs)
                semantic_bgr = cv2.cvtColor(semantic , cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.normpath(os.path.join(self.semantic_rgb_path, k + "_" + str(2) + '.png')),  semantic_bgr)

            msg.header.stamp.secs = sec
            msg.header.stamp.nsecs = nsec
            msg.header.frame_id = self.frame_id

            return msg

    # return None if not match, otherwise return a msg and topic name
    def check_match(self,sec,nsec):
        k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            return self.topic_name, self.compose_msg(filename, sec,nsec)
        return None

    def check_match_airsim(self,sec,nsecs, vio_pose_msg, sparse_dataset=False):
        nsec = round(nsecs*1e-6)
        if nsec < 10:
             k = '{}.00{}'.format(sec,nsec)
        elif nsec < 100:
            k = '{}.0{}'.format(sec,nsec)
        else:
            k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            if sparse_dataset == True:
                sec = vio_pose_msg.transforms[0].header.stamp.secs
                nsecs = vio_pose_msg.transforms[0].header.stamp.nsecs
            return self.topic_name, self.compose_msg(filename, sec,nsecs)
        return None

class SemanticLabelDirMessages:
    def __init__(self, folder, topic_name, frame_id):
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.D = {}
        # for filename in glob.glob(folder+'/*_label.png'):
        #     k = os.path.basename(filename).split('_label')[0]
        #     self.D[k] = filename

        # airsim
        for filename in glob.glob(folder+'/*.png'):
            k = os.path.basename(filename).split('_2.png')[0]
            k =k[:-3]+'.'+k[-3:]
            self.D[k] = filename

        print('folder: {} topic: {} frame_id {} size {}'.format(folder, self.topic_name, self.frame_id, len(self.D)))
        return

    def compose_msg(self, filename, sec, nsec):
        img = cv2.imread(filename,0)
        # print("filename: ", filename)
        if img is None:
            print(f"sec: {sec} nsec:{nsec}")
        msg = CvBridge().cv2_to_imgmsg(img,"mono8")
        msg.header.stamp.secs = sec
        msg.header.stamp.nsecs = nsec
        msg.header.frame_id = self.frame_id
        return msg

    # return None if not match, otherwise return a msg and topic name
    def check_match(self,sec,nsec):
        k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            return self.topic_name, self.compose_msg(filename, sec,nsec)
        return None

    def check_match_airsim(self,sec,nsecs, vio_pose_msg, sparse_dataset=False):
        nsec = round(nsecs*1e-6)
        if nsec < 10:
             k = '{}.00{}'.format(sec,nsec)
        elif nsec < 100:
            k = '{}.0{}'.format(sec,nsec)
        else:
            k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            if sparse_dataset == True:
                sec = vio_pose_msg.transforms[0].header.stamp.secs
                nsecs = vio_pose_msg.transforms[0].header.stamp.nsecs
            return self.topic_name, self.compose_msg(filename, sec,nsecs)
        return None

class StereoDepthDirMessages:
    def __init__(self, folder, topic_name, frame_id):
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.D = {}
        # for filename in glob.glob(folder+'/*.npy'):
        #     k = os.path.basename(filename).split('.npy')[0]
        #     self.D[k] = filename

        #  airsim
        for filename in glob.glob(folder+'/*.pfm'):
            k = os.path.basename(filename).split('_1.pfm')[0]
            k =k[:-3]+'.'+k[-3:]
            self.D[k] = filename

        print('folder: {} topic: {} frame_id {} size {}'.format(folder, self.topic_name, self.frame_id, len(self.D)))
        return

    def compose_msg(self, filename, sec, nsec):
        # img = np.load(filename)
        img_array, scale = airsim.read_pfm(filename)
        img_array = DepthConversion(img_array, g_fd)

        msg = CvBridge().cv2_to_imgmsg(img_array, "32FC1")
        msg.header.stamp.secs = sec
        msg.header.stamp.nsecs = nsec
        msg.header.frame_id = self.frame_id
        return msg

    # return None if not match, otherwise return a msg and topic name
    def check_match(self,sec,nsec):
        k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            return self.topic_name, self.compose_msg(filename, sec,nsec)
        return None

    def check_match_airsim(self,sec,nsecs, vio_pose_msg, sparse_dataset=False):
        nsec = round(nsecs*1e-6)
        if nsec < 10:
             k = '{}.00{}'.format(sec,nsec)
        elif nsec < 100:
            k = '{}.0{}'.format(sec,nsec)
        else:
            k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            if sparse_dataset == True:
                sec = vio_pose_msg.transforms[0].header.stamp.secs
                nsecs = vio_pose_msg.transforms[0].header.stamp.nsecs
            return self.topic_name, self.compose_msg(filename, sec,nsecs)
        return None

class StereoImageDirMessages:
    def __init__(self, folder, topic_name, frame_id):
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.D = {}
        # for filename in glob.glob(folder+'/*.png'):
        #     k = os.path.basename(filename).split('.png')[0]
        #     self.D[k] = filename

        # airsim
        for filename in glob.glob(folder+'/*.png'):
            k = os.path.basename(filename).split('_0.png')[0]
            k =k[:-3]+'.'+k[-3:]
            self.D[k] = filename

        print('folder: {} topic: {} frame_id {} size {}'.format(folder, self.topic_name, self.frame_id, len(self.D)))
        return

    def compose_msg(self, filename, sec, nsec):
        img = cv2.imread(filename)
        # msg = CvBridge().cv2_to_imgmsg(img,"rgb8")
        msg = CvBridge().cv2_to_imgmsg(img,"bgr8")
        msg.header.stamp.secs = sec
        msg.header.stamp.nsecs = nsec
        msg.header.frame_id = self.frame_id
        return msg

    # return None if not match, otherwise return a msg and topic name
    def check_match(self,sec,nsec):
        k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            return self.topic_name, self.compose_msg(filename, sec,nsec)
        return None

    def check_match_airsim(self,sec,nsecs, vio_pose_msg, sparse_dataset=False):
        nsec = round(nsecs*1e-6)
        if nsec < 10:
             k = '{}.00{}'.format(sec,nsec)
        elif nsec < 100:
            k = '{}.0{}'.format(sec,nsec)
        else:
            k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            if sparse_dataset == True:
                sec = vio_pose_msg.transforms[0].header.stamp.secs
                nsecs = vio_pose_msg.transforms[0].header.stamp.nsecs
            return self.topic_name, self.compose_msg(filename, sec,nsecs)
        return None

class BoundingBoxDirMessages:
    def __init__(self, folder, topic_name, frame_id):
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.D = {}

        # for bounding box
        self.bbs_id=0 # instance id
        self.bbs = {} # all boundingbox
        # for filename in glob.glob(folder+'/*_label.png'):
        #     k = os.path.basename(filename).split('_label')[0]
        #     self.D[k] = filename

        # airsim
        for filename in glob.glob(folder+'/*.txt'):
            k = os.path.basename(filename).split('_0.txt')[0]
            k =k[:-3]+'.'+k[-3:]
            self.D[k] = filename

        print('folder: {} topic: {} frame_id {} size {}'.format(folder, self.topic_name, self.frame_id, len(self.D)))
        return

##### Object_3D
# uint32 id
# uint32 category
# vision_msgs/BoundingBox2D bbox_2d
# vision_msgs/BoundingBox3D bbox_3d
# float32 conf
# float32 is_occluded
# float32 is_cropped

# # The 2D position (in pixels) and orientation of the bounding box center.
# geometry_msgs/Pose2D center
#         float64 x
#         float64 y
#         float64 theta
# # The size (in pixels) of the bounding box surrounding the object relative
# #   to the pose of its center.
# float64 size_x
# float64 size_y

# header = "box2D.max.x_val box2D.max.y_val box2D.min.x_val box2D.min.y_val \
# box3D.max.x_val box3D.max.y_val box3D.max.z_val \
# box3D.min.x_val box3D.min.y_val box3D.min.z_val \
# geo_point.altitude geo_point.latitude geo_point.longitude \
# name \
# relative_pose.orientation.w_val relative_pose.orientation.x_val relative_pose.orientation.y_val relative_pose.orientation.z_val \
# relative_pose.position.x_val relative_pose.position.y_val relative_pose.position.z_val label"

    def compose_msg(self, filename, sec, nsec, vio_pose_msg):
        vio_position = vio_pose_msg.transforms[0].transform.translation
        vio_rotation = vio_pose_msg.transforms[0].transform.rotation
        Tw1c = np.eye(4)
        Tw1c[:3, 3] = [vio_position.x, vio_position.y, vio_position.z]
        Tw1c[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((vio_rotation.w, vio_rotation.x, vio_rotation.y, vio_rotation.z))
        Tcw1=np.linalg.inv(Tw1c)

        C = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])
        C1 = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])
        C2 = np.array([
            [ 0,  1,  0,  0],
            [ 1,  0,  0,  0],
            [ 0,  0,  -1,  0],
            [ 0,  0,  0,  1]
        ])
        C3=C1 @ C2 # cam(ned) --> cam(右x下y前z)

        msg =  Perception()
        df = pd.read_csv(filename, sep='\s+')
        for frame in (range(0, df.shape[0], 1)):
            max_x, max_y, min_x, min_y, name, label, max_x_3d, max_y_3d, max_z_3d, min_x_3d, min_y_3d, min_z_3d, qw, qx, qy, qz, x, y, z, \
            o_x, o_y, o_z, o_qw, o_qx, o_qy, o_qz \
            = df.iloc[frame][[
                'box2D.max.x_val', 'box2D.max.y_val',
                'box2D.min.x_val', 'box2D.min.y_val',
                'name',
                'label',
                'box3D.max.x_val', 'box3D.max.y_val', 'box3D.max.z_val',
                'box3D.min.x_val', 'box3D.min.y_val', 'box3D.min.z_val',
                'relative_pose.orientation.w_val', 'relative_pose.orientation.x_val', 'relative_pose.orientation.y_val', 'relative_pose.orientation.z_val',
                'relative_pose.position.x_val','relative_pose.position.y_val','relative_pose.position.z_val',
                'pose_in_w.position.x_val', 'pose_in_w.position.y_val', 'pose_in_w.position.z_val',
                'pose_in_w.orientation.w_val', 'pose_in_w.orientation.x_val', 'pose_in_w.orientation.y_val', 'pose_in_w.orientation.z_val'
            ]]
            obj_3d = Object_3D()
            if name in self.bbs:
                obj_3d.id = self.bbs[name][1]
            else:
                self.bbs_id=self.bbs_id+1
                obj_3d.id = self.bbs_id
                self.bbs[name]=[label, self.bbs_id]

            obj_3d.category = label
            obj_3d.bbox_2d.center.x = (max_x+min_x)/2
            obj_3d.bbox_2d.center.y = (max_y+min_y)/2
            obj_3d.bbox_2d.center.theta = 0
            obj_3d.bbox_2d.size_x = (max_x-min_x)/2
            obj_3d.bbox_2d.size_y =  (max_y-min_y)/2

            obj_3d.bbox_3d.size.x =abs(max_x_3d-min_x_3d)
            obj_3d.bbox_3d.size.y =abs(max_y_3d-min_y_3d)
            obj_3d.bbox_3d.size.z =abs(max_z_3d-min_z_3d)

            # obj_3d.bbox_3d.center.position.x =(max_x_3d+min_x_3d)/2
            # obj_3d.bbox_3d.center.position.y =(max_y_3d+min_y_3d)/2
            # obj_3d.bbox_3d.center.position.z =(max_z_3d+min_z_3d)/2
            # # !!!!!!!!!!! notice : relative_pose.position.z_val not correct (below z) !!!!!!!!!!!!!!!!!!!!!
            # # obj_3d.bbox_3d.center.position.x =x
            # # obj_3d.bbox_3d.center.position.y =y
            # # obj_3d.bbox_3d.center.position.z =z

            # obj_3d.bbox_3d.center.orientation.w = qw
            # obj_3d.bbox_3d.center.orientation.x = qx
            # obj_3d.bbox_3d.center.orientation.y = qy
            # obj_3d.bbox_3d.center.orientation.z = qz

            ########################## 方式二：用 simGetObjectPose 获取的 box 结果 #######################################
            # 1 先转到世界坐标系下
            T2 = [o_x, o_y, o_z, 1]
            R2 = np.eye(4)
            R2[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((o_qw, o_qx, o_qy, o_qz))
            t_w1o1 = C.T @ C3 @ T2
            F21 = C.T@ C3 @ R2
            F22 = F21[:3, :3]
            R22 = Rotation.from_matrix(F22)
            q_w1o1 = R22.as_quat()
            # new_quat = q_w1o1

            # 2 进行z方向的补偿（沿着box朝向）
            # 沿着box的z方向补偿一半的高度
            sign = 1
            if max_z_3d-min_z_3d < 0:
                sign = -1
            T_WO = np.eye(4)
            T_WO[:,3] =t_w1o1
            r_wo2 = F21
            r_wo2[:3, 3] = [0,  0,  sign * (max_z_3d-min_z_3d)/2]
            T_WO = T_WO @ r_wo2
            # 沿着box的z方向补偿一半的高度
            t_w1o2 = T_WO[:, 3]
            # new_p = t_w1o2

            # 以下为将box位姿转到相机下 t_co
            # Tco = Tcw1 * Tw1o
            t_co = Tcw1 @ t_w1o2
            r_co = Tcw1 @ F21
            r_co1 = Rotation.from_matrix(r_co[:3,:3])
            q_co = r_co1.as_quat()
            ########################## 方式二：用 simGetObjectPose 获取的 box 结果 #######################################
            obj_3d.bbox_3d.center.position.x = t_co[0]
            obj_3d.bbox_3d.center.position.y = t_co[1]
            obj_3d.bbox_3d.center.position.z = t_co[2]

            obj_3d.bbox_3d.center.orientation.w = q_co[3]
            obj_3d.bbox_3d.center.orientation.x = q_co[0]
            obj_3d.bbox_3d.center.orientation.y = q_co[1]
            obj_3d.bbox_3d.center.orientation.z = q_co[2]

            msg.obj_3d.append(obj_3d)

        msg.header.stamp.secs = sec
        msg.header.stamp.nsecs = nsec
        msg.header.frame_id = self.frame_id
        return msg

    # return None if not match, otherwise return a msg and topic name
    def check_match(self,sec,nsec):
        k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            return self.topic_name, self.compose_msg(filename, sec,nsec)
        return None

    def check_match_airsim(self,sec,nsecs, vio_pose_msg, sparse_dataset=False):
        nsec = round(nsecs*1e-6)
        if nsec < 10:
             k = '{}.00{}'.format(sec,nsec)
        elif nsec < 100:
            k = '{}.0{}'.format(sec,nsec)
        else:
            k = '{}.{}'.format(sec,nsec)
        if k in self.D:
            filename = self.D[k]
            # print('matched: '+filename)
            self.D.pop(k)
            if sparse_dataset == True:
                sec = vio_pose_msg.transforms[0].header.stamp.secs
                nsecs = vio_pose_msg.transforms[0].header.stamp.nsecs
            return self.topic_name, self.compose_msg(filename, sec,nsecs,vio_pose_msg)
        return None

    def save_bbs_label_id_map(self):
        with open("bbs_label_id_map.txt",'w') as f:    #设置文件对象
            header="name label id"
            f.writelines(header+"\n")
            for k in self.bbs:
                f.writelines(f'{k} {self.bbs[k][0]} {self.bbs[k][1]}\n')

def get_static_transform(to_frame_id, from_frame_id, transform):
    t = transform[0:3, 3]
    q = tf.transformations.quaternion_from_matrix(transform)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = to_frame_id
    tf_msg.child_frame_id = from_frame_id
    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def inv(transform):
    "Invert rigid body transformation matrix"
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv

from tf2_msgs.msg import TFMessage
def save_static_transforms(bag, transforms, timestamps):
    print("Exporting static transformations")
    tfm = TFMessage()
    for transform in transforms:
        t = get_static_transform(to_frame_id=transform[0], from_frame_id=transform[1], transform=transform[2])
        tfm.transforms.append(t)
    for timestamp in timestamps:
        time = timestamp
        # time = rospy.Time.from_sec(float(timestamp.strftime("%s.%f")))
        for i in range(len(tfm.transforms)):
            tfm.transforms[i].header.stamp = time
        bag.write('/tf_static', tfm, t=time)

import tf2_msgs.msg
import geometry_msgs.msg
def transform_msg_from_csv(path_to_csv, child_frame_id, frame_id):
    tfs = pd.read_csv(path_to_csv)
    tf_array = []
    for index, tf in tfs.iterrows():
        tf_msg = tf2_msgs.msg.TFMessage()
        tf_stamped = geometry_msgs.msg.TransformStamped()
        tf_stamped.header.frame_id = frame_id
        tf_stamped.child_frame_id = child_frame_id
        # tf_stamped.header.stamp = rospy.Time.from_sec(tf["#timestamp"]*1e-9) # ns to sec
        # Assumes timestamps are in the first column
        tf_stamped.header.stamp = rospy.Time.from_sec(tf[0]*1e-9) # ns to sec
        tf_stamped.transform.translation.x = tf['x']
        tf_stamped.transform.translation.y = tf['y']
        tf_stamped.transform.translation.z = tf['z']
        tf_stamped.transform.rotation.x = tf['qx']
        tf_stamped.transform.rotation.y = tf['qy']
        tf_stamped.transform.rotation.z = tf['qz']
        tf_stamped.transform.rotation.w = tf['qw']
        tf_msg.transforms.append(tf_stamped)
        tf_array.append(tf_msg)
    return tf_array

#  !!!!!!!!!!! not ok !!!!!!!!!
def transform_msg_from_txt(path_to_airsim_txt, child_frame_id, frame_id):
    tf_array = []

    fin = open(path_to_airsim_txt, "r")
    line = fin.readline().strip()
    line = fin.readline().strip()
    while (line):
        parts = line.split("\t")
        timestamp = parts[1] # ms
        timestamp = float(parts[1]) / 1000.0 # s

        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])

        qw = float(parts[5])
        qx = float(parts[6])
        qy = float(parts[7])
        qz = float(parts[8])


        tf_msg = tf2_msgs.msg.TFMessage()
        tf_stamped = geometry_msgs.msg.TransformStamped()
        tf_stamped.header.frame_id = frame_id
        tf_stamped.child_frame_id = child_frame_id
        # tf_stamped.header.stamp = rospy.Time.from_sec(tf["#timestamp"]*1e-9) # ns to sec
        # Assumes timestamps are in the first column
        tf_stamped.header.stamp = rospy.Time.from_sec(timestamp)
        tf_stamped.transform.translation.x = -y
        tf_stamped.transform.translation.y = -z
        tf_stamped.transform.translation.z = -x

        tf_stamped.transform.rotation.x = qx
        tf_stamped.transform.rotation.y = qy
        tf_stamped.transform.rotation.z = qz
        tf_stamped.transform.rotation.w = qw
        tf_msg.transforms.append(tf_stamped)
        tf_array.append(tf_msg)

        line = fin.readline().strip()

    print('frame_id {} size {}'.format(frame_id, len(tf_array)))
    return tf_array

def transform_msg_from_txt2(path_to_airsim_txt, child_frame_id, frame_id):
    tf_array = []

    fin = open(path_to_airsim_txt, "r")
    line = fin.readline().strip()
    line = fin.readline().strip()
    while (line):
        parts = line.split("\t")
        timestamp = parts[1] # ms
        timestamp = float(parts[1]) / 1000.0 # s

        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])

        qw = float(parts[5])
        qx = float(parts[6])
        qy = float(parts[7])
        qz = float(parts[8])

        # === Create the transformation matrix ===
        T = np.eye(4)
        T[:3,3] = [-y, -z, -x]

        R = np.eye(4)
        # get_rotation_matrix_from_quaternion (w,x,y,z)
        R[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((qw, qy, qz, qx)) # TODO:(qzc) (qw,qx,qy,qz) ?

        # test trasformation
        # R_test = np.eye(3)
        # R_test = o3d.geometry.get_rotation_matrix_from_quaternion((-0.8733046, 0, -0.4871745, 0)) # TODO:(qzc) (qw,qx,qy,qz) ?
        # R_test1 = Rotation.from_matrix(R_test)
        # Q_test = R_test1.as_quat() # Q_test[3] = qw

        C = np.array([
                [ 1,  0,  0,  0],
                [ 0,  0, -1,  0],
                [ 0,  1,  0,  0],
                [ 0,  0,  0,  1]
            ])
        F = R.T @ T @ C

        # add by aqiu
        # print(F)
        F=np.linalg.inv(F)
        # print(F)

        F_T = F[:, 3]
        F_R = F[:3, :3]
        R1 = Rotation.from_matrix(F_R)
        F_Q = R1.as_quat()
        # === Create the transformation matrix ===

        tf_msg = tf2_msgs.msg.TFMessage()
        tf_stamped = geometry_msgs.msg.TransformStamped()
        tf_stamped.header.frame_id = frame_id
        tf_stamped.child_frame_id = child_frame_id
        # tf_stamped.header.stamp = rospy.Time.from_sec(tf["#timestamp"]*1e-9) # ns to sec
        # Assumes timestamps are in the first column
        tf_stamped.header.stamp = rospy.Time.from_sec(timestamp)
        tf_stamped.transform.translation.x = F_T[0]
        tf_stamped.transform.translation.y = F_T[1]
        tf_stamped.transform.translation.z = F_T[2]


        tf_stamped.transform.rotation.x = F_Q[0]
        tf_stamped.transform.rotation.y = F_Q[1]
        tf_stamped.transform.rotation.z = F_Q[2]
        tf_stamped.transform.rotation.w = F_Q[3]
        tf_msg.transforms.append(tf_stamped)
        tf_array.append(tf_msg)

        line = fin.readline().strip()

    print('frame_id {} size {}'.format(frame_id, len(tf_array)))
    return tf_array

from nav_msgs.msg import Odometry
def generate_odometry_msg(vio_pose_msg, child_frame_id, frame_id):
    vio_timestamp = vio_pose_msg.transforms[0].header.stamp
    vio_position = vio_pose_msg.transforms[0].transform.translation
    vio_rotation = vio_pose_msg.transforms[0].transform.rotation
    # Transform transform
        #  Vector3 translation
        # Quaternion rotation

    odom_local = Odometry()
    odom_local.header.stamp = vio_timestamp
    odom_local.header.frame_id = frame_id
    odom_local.child_frame_id = child_frame_id

    # geometry_msgs/PoseWithCovariance pose
        # Point position
        # Quaternion orientation
    odom_local.pose.pose.position.x = vio_position.x
    odom_local.pose.pose.position.y = vio_position.y
    odom_local.pose.pose.position.z = vio_position.z
    odom_local.pose.pose.orientation = vio_rotation

    return odom_local

import sys
def status(length, percent, index=0, remain_time=1000):
    sys.stdout.write('\x1B[2K') # Erase entire current line
    sys.stdout.write('\x1B[0E') # Move to the beginning of the current line
    progress = "Progress: ["
    for i in range(0, length):
        if i < length * percent:
            progress += '='
        else:
            progress += ' '
    remain_time_str = f'{remain_time:.2f}' # 小数点后保留两位，四舍五入
    progress += "] " + str(round(percent * 100.0, 2)) + "%" + " : " + str(index) + " : " + remain_time_str +"h"
    # progress += "] " + str(round(percent * 100.0, 2)) + "%\n"
    sys.stdout.write(progress)
    sys.stdout.flush()

import yaml
import subprocess
def get_rosbag_info(rosbag_path):
    # yaml>5.1
    return yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_path], stdout=subprocess.PIPE).communicate()[0], Loader=yaml.FullLoader)
    # return yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_path], stdout=subprocess.PIPE).communicate()[0], Loader=yaml.CLoader)
    #备注：yaml版本5.1之后弃用，YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated
    # return yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_path], stdout=subprocess.PIPE).communicate()[0])


def parser():
    import argparse
    basic_desc = "Add imu in a excel file to a rosbag in IMU format and time synchronized order."

    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))
    input_opts = shared_parser.add_argument_group("input options")
    output_opts = shared_parser.add_argument_group("output options")

    in_bag = "/media/qiuzc/Document/4_data/kimera_data/true_data/indoor/2021-11-02-12-08-40.bag"
    # in_bag = "/media/qiuzc/Document/4_data/kimera_data/true_data/indoor/2021-11-02-12-08-40-10s.bag"
    in_imu_data = "imu_data.xlsx"
    calib_file = "lidar_cam_imu_calib_test_car_v0.0.2.yaml"
    downsample = True
    out_bag = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/bgr_depth_ir_3dbb_full_lcd_0.5hz_2022032301.bag"
    # out_bag = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/bgr_depth_ir_bb_small_ok_2022031001.bag"

    # for tf
    vio_pose_csv = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_rec.txt"
    # vio_pose_csv = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_rec_small1.txt"
    vio_topic = "/local/odometry"
    world_frame_id = "world"
    body_frame_id = "imu_baselink"  # same as imu_frame_id

    # for imu
    imu_topic = "/imu/data"
    imu_frame_id = "imu_baselink"  # same as body_frame_id

    # left cam
    left_image_frame_id = 'left_cam'
    left_image_topic = '/adu/camera_body_front_center/stereo/image_rect_color'

    # right cam
    right_image_frame_id = 'right_cam'
    right_image_topic = '/zed2/zed_node/stereo/image_rect_color_stereo_right'

    # depth
    # left_depth_frame_id = 'left_depth'
    left_depth_frame_id = 'left_cam'  # TODO same as semantic
    left_depth_topic = '/px/perception/estimated_depth/image_raw'

    # semantic
    # left_semantic_frame_id = 'left_semantic'
    left_semantic_frame_id = 'left_cam'  # TODO same as depth
    left_semantic_topic = '/px/perception/semantic_labels/image_raw'

    left_semantic_for_display_topic = '/px/perception/visual/main'

    # perception
    left_perception_frame_id = 'left_cam'
    left_perception_topic = '/px/perception/result'

    input_opts.add_argument("--input_rosbag_path", type=str,
                            help="Path to the input rosbag.",
                            # default="kimera_semantics_demo_only_imu.bag")
                            default=in_bag)
    input_opts.add_argument("--input_excel_path", type=str,
                            help="Path to the excel file with imu information.",
                            default=in_imu_data)
    input_opts.add_argument("--calib_file", type=str,
                            help="Path to the calib file.",
                            default=calib_file)
    input_opts.add_argument("--vio_pose_csv", type=str,
                            help="Path to the vio pose file.",
                            default=vio_pose_csv)
    input_opts.add_argument("--world_frame_id", type=str,
                            help="world_frame_id.",
                            default=world_frame_id)
    input_opts.add_argument("--downsample", type=str,
                            help="downsample to 960*540.",
                            default=downsample)

    output_opts.add_argument("--output_rosbag_path", type=str,
                             help="Path to the output rosbag.",
                             default=out_bag)

    output_opts.add_argument("--vio_topic", type=str,
                             help="vio topic (i.e. /local/odometry)",
                             default=vio_topic)
    output_opts.add_argument("--body_frame_id", type=str,
                             help="Frame id corresponding to the trajectory csv (i.e. base_link_DVIO_pgo)",
                             default=body_frame_id)

    output_opts.add_argument("--imu_topic", type=str,
                             help="IMU topic (i.e. /tesse/imu)",
                             default=imu_topic)
    output_opts.add_argument("--imu_frame_id", type=str,
                             help="IMU frame id (i.e. base_link)",
                             default=imu_frame_id)
    output_opts.add_argument("--left_image_topic", type=str,
                             help="left_image_topic topic (i.e. /zed/left_cam)",
                             default=left_image_topic)
    output_opts.add_argument("--left_image_frame_id", type=str,
                             help="left cam frame id (i.e. left_cam)",
                             default=left_image_frame_id)
    output_opts.add_argument("--right_image_topic", type=str,
                             help="right_image_topic topic (i.e. /zed/right_cam)",
                             default=right_image_topic)
    output_opts.add_argument("--right_image_frame_id", type=str,
                             help="right cam frame id (i.e. right_cam)",
                             default=right_image_frame_id)

    output_opts.add_argument("--left_depth_topic", type=str,
                             help="left_depth_topic topic (i.e. /zed2/zed_node/stereo/image_rect_color_stereo_depth)",
                             default=left_depth_topic)
    output_opts.add_argument("--left_depth_frame_id", type=str,
                             help="left_depth_frame_id  (i.e. left_depth)",
                             default=left_depth_frame_id)
    output_opts.add_argument("--left_semantic_topic", type=str,
                             help="left_semantic_topic  (i.e. /zed2/zed_node/stereo/image_rect_color_semantic)",
                             default=left_semantic_topic)
    output_opts.add_argument("--left_semantic_for_display_topic", type=str,
                             help="left_semantic_for_display_topic  (i.e.  /px/perception/visual/main)",
                             default=left_semantic_for_display_topic)
    output_opts.add_argument("--left_semantic_frame_id", type=str,
                             help="left_semantic_frame_id  (i.e. left_semantic)",
                             default=left_semantic_frame_id)

    output_opts.add_argument("--left_perception_topic", type=str,
                             help="left_perception_topic  (i.e. /px/perception/result)",
                             default=left_perception_topic)
    output_opts.add_argument("--left_perception_frame_id", type=str,
                             help="left_perception_frame_id  (i.e. left_cam)",
                             default=left_perception_frame_id)

    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser

import argcomplete
import sys

if __name__ == "__main__":
    start = time.perf_counter()  # 返回系统运行时间

    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # data_path =  "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_small"
    data_path =  "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_1circle"

    # for convient
    args.output_rosbag_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/bgr_depth_ir_3dbb_full_lcd_0.5hz_2022032401.bag"
    capture_hz = 20
    # args.output_rosbag_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/bgr_depth_ir_3dbb_full_lcd_0.2hz_2022032401.bag"
    # capture_hz = 50
    # args.output_rosbag_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/bgr_depth_ir_3dbb_full_lcd_0.1hz_2022032401.bag"
    # capture_hz = 100
    for_lcd = True

    # args.output_rosbag_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/bgr_depth_ir_3dbb_full_10hz_2022032401.bag"
    # for_lcd = False
    print(f"for_lcd:{for_lcd}, hz:{capture_hz}, outbag:{args.output_rosbag_path}\n")

    # stereo_right_msgs = StereoImageDirMessages(
    #     "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_ir/0",
    #     args.right_image_topic, args.right_image_frame_id)

    stereo_left_msgs = StereoImageDirMessages(
        data_path +"/0",
        args.left_image_topic, args.left_image_frame_id)

    stereo_depth_msgs = StereoDepthDirMessages(
        data_path +"/1",
        args.left_depth_topic, args.left_depth_frame_id)

    # semantic_rgb_msgs = SemanticLabelWithRGBDirMessages(
    semantic_rgb_msgs = SemanticLabelDirMessages(
       data_path +"/2",
        args.left_semantic_topic, args.left_semantic_frame_id)

    use_ir_image = False # !!! shoud chang with other variable
    if use_ir_image == False:
        semantic_rgb_with_rgb_outdoor_msgs = SemanticLabelWithRGBOutdoorDirMessages(
            data_path +"/20",
            args.left_semantic_for_display_topic, args.left_semantic_frame_id)
    else:
        semantic_rgb_with_rgb_outdoor_msgs = SemanticLabelWithRGBOutdoorDirMessages(
            data_path +"/2",
            args.left_semantic_for_display_topic, args.left_semantic_frame_id)

    bb_msgs = BoundingBoxDirMessages(
       data_path +"/21",
        args.left_perception_topic, args.left_perception_frame_id)


    tf_array_vio = transform_msg_from_txt2(args.vio_pose_csv, args.body_frame_id, args.world_frame_id)
    tf_array_idx_vio = 0

    # ---------------------------- for debug begin --------------------------------
    # cur_dir = args.vio_pose_csv[:args.vio_pose_csv.rfind(os.path.sep)] + os.path.sep
    # print(cur_dir)
    # This is for logging progress
    duration = len(tf_array_vio)
    start_index = 0
    interval = 10 / duration
    last_percent = 0
    start_time = time.time()
    status(40, 0)
    # ---------------------------- for debug end --------------------------------
    skip_cnt = 0
    if for_lcd == False:
        with rosbag.Bag(args.output_rosbag_path, 'w') as outbag:
            for i, vio_pose_msg in enumerate(tf_array_vio):
                vio_timestamp = vio_pose_msg.transforms[0].header.stamp

                # TODO:(qzc) check
                # vio odometry msg
                if skip_cnt > 10:
                    break

                odometry_msg = generate_odometry_msg(vio_pose_msg, args.body_frame_id, args.world_frame_id)
                outbag.write(args.vio_topic, odometry_msg, vio_timestamp)

                for local_msgs in [semantic_rgb_msgs, semantic_rgb_with_rgb_outdoor_msgs, stereo_depth_msgs, stereo_left_msgs, bb_msgs]:
                # for local_msgs in [semantic_rgb_msgs, stereo_depth_msgs, stereo_left_msgs, bb_msgs]:
                    ret = local_msgs.check_match_airsim(vio_timestamp.secs, vio_timestamp.nsecs, vio_pose_msg, False)
                    if not ret is None:
                        topic, msg = ret
                        outbag.write(topic, msg, vio_timestamp)
                    else:
                        skip_cnt=skip_cnt+1
                        print("not find {}.{} ".format(vio_timestamp.secs, vio_timestamp.nsecs))


                percent = (i - start_index) / duration
                if percent - last_percent > interval:
                    last_percent = percent

                    now_time = time.time()
                    remain_time = float(now_time - start_time) * (duration-i) / (i * 3600)
                    status(40, percent, i, remain_time)

            bb_msgs.save_bbs_label_id_map()
            status(40, 1, duration, 0)
    else:
        with rosbag.Bag(args.output_rosbag_path, 'w') as outbag:
            origin_start_index = 0.0
            origin_end_time = 0.0
            origin_duration = 0.0
            origin_len = len(tf_array_vio)
            for i, vio_pose_msg in enumerate(tf_array_vio):
                vio_timestamp = vio_pose_msg.transforms[0].header.stamp
                if i == 0:
                    origin_start_index = vio_timestamp
                elif i == origin_len-1:
                    origin_end_time = vio_timestamp
            origin_duration = origin_end_time - origin_start_index

            # 第一次取偶数帧
            for i, vio_pose_msg in enumerate(tf_array_vio):
                if i % capture_hz != 0:
                    continue
                vio_timestamp = vio_pose_msg.transforms[0].header.stamp

                # TODO:(qzc) check
                # vio odometry msg
                if skip_cnt > 10:
                    break

                odometry_msg = generate_odometry_msg(vio_pose_msg, args.body_frame_id, args.world_frame_id)
                outbag.write(args.vio_topic, odometry_msg, vio_timestamp)

                # for local_msgs in [semantic_rgb_msgs, stereo_depth_msgs, stereo_left_msgs,]:
                for local_msgs in [semantic_rgb_msgs, semantic_rgb_with_rgb_outdoor_msgs, stereo_depth_msgs, stereo_left_msgs, bb_msgs]:
                # for local_msgs in [semantic_rgb_msgs, stereo_depth_msgs, stereo_left_msgs, bb_msgs]:
                    ret = local_msgs.check_match_airsim(vio_timestamp.secs, vio_timestamp.nsecs, vio_pose_msg, False)
                    if not ret is None:
                        topic, msg = ret
                        outbag.write(topic, msg, vio_timestamp)
                    else:
                        skip_cnt=skip_cnt+1
                        print("not find {}.{} ".format(vio_timestamp.secs, vio_timestamp.nsecs))


                percent = (i - start_index) / duration
                if percent - last_percent > interval:
                    last_percent = percent

                    now_time = time.time()
                    remain_time = float(now_time - start_time) * (duration-i) / (i * 3600)
                    status(40, percent, i, remain_time)

            # 第二次取奇数帧，但是时间戳需要偏移整个数据集长度
            start_index = 0
            last_percent = 0
            for i, vio_pose_msg in enumerate(tf_array_vio):
                if i % capture_hz != capture_hz//2:
                    continue
                origin_vio_timestamp = vio_pose_msg.transforms[0].header.stamp
                vio_timestamp = origin_vio_timestamp + origin_duration

                # TODO:(qzc) check
                # vio odometry msg
                if skip_cnt > 10:
                    break

                vio_pose_msg.transforms[0].header.stamp = vio_timestamp
                odometry_msg = generate_odometry_msg(vio_pose_msg, args.body_frame_id, args.world_frame_id)
                outbag.write(args.vio_topic, odometry_msg, vio_timestamp)

                # for local_msgs in [semantic_rgb_msgs, stereo_depth_msgs, stereo_left_msgs,]:
                for local_msgs in [semantic_rgb_msgs, semantic_rgb_with_rgb_outdoor_msgs, stereo_depth_msgs, stereo_left_msgs, bb_msgs]:
                # for local_msgs in [semantic_rgb_msgs, stereo_depth_msgs, stereo_left_msgs, bb_msgs]:
                    ret = local_msgs.check_match_airsim(origin_vio_timestamp.secs, origin_vio_timestamp.nsecs, vio_pose_msg, True)
                    if not ret is None:
                        topic, msg = ret
                        outbag.write(topic, msg, vio_timestamp)
                    else:
                        skip_cnt=skip_cnt+1
                        print("not find {}.{} ".format(origin_vio_timestamp.secs, origin_vio_timestamp.nsecs))


                percent = (i - start_index) / duration
                if percent - last_percent > interval:
                    last_percent = percent

                    now_time = time.time()
                    remain_time = float(now_time - start_time) * (duration-i) / (i * 3600)
                    status(40, percent, i, remain_time)


            bb_msgs.save_bbs_label_id_map()
            status(40, 1, duration, 0)


    end = time.perf_counter()
    print('\n用时：{:.4f}s'.format(end - start))

