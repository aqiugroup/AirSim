# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
''' script to merge msg to rosbag '''
import setup_path
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

        print('topic: {} frame_id {} size {}'.format(self.topic_name, self.frame_id, len(self.D)))
        return

    def compose_msg(self, filename, sec, nsec):
        img = cv2.imread(filename,0)
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

    def check_match_airsim(self,sec,nsecs):
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
        print('topic: {} frame_id {} size {}'.format(self.topic_name, self.frame_id, len(self.D)))
        return

    def compose_msg(self, filename, sec, nsec):
        # img = np.load(filename)
        img_array, scale = airsim.read_pfm(filename)

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

    def check_match_airsim(self,sec,nsecs):
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
        print('topic: {} frame_id {} size {}'.format(self.topic_name, self.frame_id, len(self.D)))
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

    def check_match_airsim(self,sec,nsecs):
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
            return self.topic_name, self.compose_msg(filename, sec,nsecs)
        return None


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

def transform_msg_from_txt(path_to_airsim_txt, child_frame_id, frame_id):
    tf_array = []

    fin = open(path_to_airsim_txt, "r")
    line = fin.readline().strip()
    line = fin.readline().strip()
    while (line):
        parts = line.split("\t")
        timestamp = parts[1] # ms
        timestamp = float(parts[1]) / 1000.0 # s

        pos_x = float(parts[2])
        pos_y =float( parts[3])
        pos_z = float(parts[4])

        quat_w = float(parts[5])
        quat_x = float(parts[6])
        quat_y = float(parts[7])
        quat_z = float(parts[8])

        tf_msg = tf2_msgs.msg.TFMessage()
        tf_stamped = geometry_msgs.msg.TransformStamped()
        tf_stamped.header.frame_id = frame_id
        tf_stamped.child_frame_id = child_frame_id
        # tf_stamped.header.stamp = rospy.Time.from_sec(tf["#timestamp"]*1e-9) # ns to sec
        # Assumes timestamps are in the first column
        tf_stamped.header.stamp = rospy.Time.from_sec(timestamp)
        tf_stamped.transform.translation.x = pos_x
        tf_stamped.transform.translation.y = pos_y
        tf_stamped.transform.translation.z = pos_z
        tf_stamped.transform.rotation.x = quat_x
        tf_stamped.transform.rotation.y = quat_y
        tf_stamped.transform.rotation.z = quat_z
        tf_stamped.transform.rotation.w = quat_w
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
def status(length, percent):
    sys.stdout.write('\x1B[2K') # Erase entire current line
    sys.stdout.write('\x1B[0E') # Move to the beginning of the current line
    progress = "Progress: ["
    for i in range(0, length):
        if i < length * percent:
            progress += '='
        else:
            progress += ' '
    progress += "] " + str(round(percent * 100.0, 2)) + "%"
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
    out_bag = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/bgr_depth_ir_pose_2022030901.bag"
    # out_bag = "/media/qiuzc/Document/4_data/kimera_data/true_data/indoor/2021-11-02-12-08-40_test.bag"

    # for tf
    vio_pose_csv = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_rec.txt"
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
    output_opts.add_argument("--left_semantic_frame_id", type=str,
                             help="left_semantic_frame_id  (i.e. left_semantic)",
                             default=left_semantic_frame_id)

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

    # stereo_right_msgs = StereoImageDirMessages(
    #     "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_drone_ir/0",
    #     args.right_image_topic, args.right_image_frame_id)

    stereo_left_msgs = StereoImageDirMessages(
        "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_drone_small/0",
        args.left_image_topic, args.left_image_frame_id)

    stereo_depth_msgs = StereoDepthDirMessages(
        "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_drone_small/1",
        args.left_depth_topic, args.left_depth_frame_id)

    # semantic_rgb_msgs = SemanticLabelWithRGBDirMessages(
    semantic_rgb_msgs = SemanticLabelDirMessages(
        "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_drone_small/2",
        args.left_semantic_topic, args.left_semantic_frame_id)

    tf_array_vio = transform_msg_from_txt(args.vio_pose_csv, args.body_frame_id, args.world_frame_id)
    tf_array_idx_vio = 0
    # ---------------------------- for debug begin --------------------------------
    # cur_dir = args.vio_pose_csv[:args.vio_pose_csv.rfind(os.path.sep)] + os.path.sep
    # print(cur_dir)
    # This is for logging progress
    duration = len(tf_array_vio)
    start_time = 0
    interval = 10 / duration
    last_percent = 0
    status(40, 0)
    # ---------------------------- for debug end --------------------------------

    skip_cnt = 0
    with rosbag.Bag(args.output_rosbag_path, 'w') as outbag:
        for i, vio_pose_msg in enumerate(tf_array_vio):
            vio_timestamp = vio_pose_msg.transforms[0].header.stamp

            # TODO:(qzc) check
            # vio odometry msg
            if skip_cnt > 10:
                break

            odometry_msg = generate_odometry_msg(vio_pose_msg, args.body_frame_id, args.world_frame_id)
            outbag.write(args.vio_topic, odometry_msg, vio_timestamp)

            for local_msgs in [semantic_rgb_msgs, stereo_depth_msgs, stereo_left_msgs,]:
                ret = local_msgs.check_match_airsim(vio_timestamp.secs, vio_timestamp.nsecs) # ns to ms
                # ret = local_msgs.check_match(vio_timestamp.secs, vio_timestamp.nsecs)
                if not ret is None:
                    topic, msg = ret
                    outbag.write(topic, msg, vio_timestamp)

                else:
                    skip_cnt=skip_cnt+1
                    print("not find {}.{} ".format(vio_timestamp.secs, vio_timestamp.nsecs))


            percent = (i - start_time) / duration
            if percent - last_percent > interval:
                last_percent = percent
                status(40, percent)

        status(40, 1)

    end = time.perf_counter()
    print('\n用时：{:.4f}s'.format(end - start))

