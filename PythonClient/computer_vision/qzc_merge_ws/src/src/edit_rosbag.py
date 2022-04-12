import math
import os
import random
import sys
# sys.path.remove('/home/qiuzc/Documents/1_code/2_github/kimera/catkin_ws/devel/lib/python2.7/dist-packages')
# import sh
# sh.zsh('-c', 'source /home/qiuzc/Documents/1_code/2_github/kimera/python3_ros_ws/devel_isolated/setup.zsh')


import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as rosImage
from sensor_msgs.msg import CompressedImage

from PIL import Image as PIL_Image
import message_filters

import numpy as np
import pandas as pd
# from numpy import nan
import time


if sys.getdefaultencoding() != 'utf-8':
    sys.reload(sys)
    sys.setdefaultencoding('utf-8')


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


id_color_dict = {}
def read_csv_of_semantic_label(file):
    df = pd.read_csv(file, header=0, sep=',', usecols=[
                     5, 6, 7, 8], encoding='utf-8')
    print(df.head(4))
    print(df.columns[0])
    print(df.shape)

    labels = list(df['SemanticPixelSegmenation'])
    reds = list(df['red'])
    greens = list(df['green'])
    blues = list(df['blue'])

    print(labels)
    print(reds)
    print(greens)
    print(blues)
    for i in range(len(df)):
        label_id = labels[i]
        if np.isnan(label_id) == False:
            id_color_dict[np.int32(label_id)] = (reds[i], greens[i], blues[i])

    print("finish")


read_csv_of_semantic_label(
    "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/semantic_model_config.csv")

# for key, val in zip(OBJECT_OF_INTEREST_CLASSES_OUTDOOR, range(len(CLASS_COLORS_OUTDOOR))):
#     id_color_dict[val] = CLASS_COLORS_OUTDOOR[val]
id_color_dict[255] = (0, 0, 0)
map_label_to_color = np.vectorize(
    # lambda x: self.object_name_dict.get(self.instance_id_to_name.get(x, "void_color"))
    lambda x: id_color_dict.get(x, id_color_dict[255])
)

bag_name = '_2022-04-11-11-48-23_0.bag'  # 被修改的bag名
out_bag_name = 'out_2022-04-11-11-48-23_0.bag'  # 修改后的bag名
dst_dir = '/Users/aqiu/Documents/AirSim/2022-03-07-02-02/'  # 使用路径


with rosbag.Bag(dst_dir+out_bag_name, 'w') as outbag:
    stamp = None
    # topic:就是发布的topic msg:该topic在当前时间点下的message t:消息记录时间(非header)
    # read_messages内可以指定的某个topic
    bag_in = rosbag.Bag(dst_dir+bag_name)
    msg_count = bag_in.get_message_count()
    # ---------------------------- for debug begin --------------------------------
    # cur_dir = args.vio_pose_csv[:args.vio_pose_csv.rfind(os.path.sep)] + os.path.sep
    # print(cur_dir)
    # This is for logging progress
    duration = msg_count
    start_index = 0
    interval = 10 / duration
    last_percent = 0
    start_time = time.time()
    status(40, 0)
    # ---------------------------- for debug end --------------------------------
    i = 0
    for topic, msg, t in bag_in.read_messages():
        outbag.write(topic, msg, msg.header.stamp)
        if topic == '/px/perception/semantic_labels':
            msg_bak = msg

            cv_image = CvBridge().imgmsg_to_cv2(msg, "mono8")

            topic = '/px/perception/visual/main'
            semantic = np.asarray(map_label_to_color(cv_image))
            semantic = np.stack(
                (semantic[0], semantic[1], semantic[2]), axis=2)
            semantic = semantic.astype(np.uint8)

            # msg = CvBridge().cv2_to_imgmsg(semantic, "bgr8")
            msg = CvBridge().cv2_to_imgmsg(semantic, "rgb8")

            msg.header.stamp.secs = msg_bak.header.stamp.secs
            msg.header.stamp.nsecs = msg_bak.header.stamp.nsecs
            msg.header.frame_id = msg_bak.header.frame_id

            outbag.write(topic, msg, msg.header.stamp)

        i=i+1
        percent = (i - start_index) / duration
        if percent - last_percent > interval:
            last_percent = percent

            now_time = time.time()
            remain_time = float(now_time - start_time) * (duration-i) / (i * 3600)
            status(40, percent, i, remain_time)

status(40, 1, duration, 0)
print("finished")