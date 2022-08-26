import math
import os
import random
import sys
import shutil
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

bag_name = '2022-04-12-16-22-58.bag'  # 被修改的bag名
out_bag_name = 'out_2022-04-12-16-22-58_1.bag'  # 修改后的bag名
dst_dir = '/Users/aqiu/Documents/AirSim/2022-03-07-02-02/'  # 使用路径
WIDTH=960
HEIGHT=540
g_shape = (WIDTH, HEIGHT)


topic_odom = '/local/odometry'
topic_rgb = '/adu/camera_body_front_center/stereo/image_rect_color'
topic_semantic_confidence = '/px/perception/semantic_confidence'
topic_semantic_labels = '/px/perception/semantic_labels'
topic_semantic_rgb = '/px/perception/visual/main'

extract_bag_dir = dst_dir+bag_name[:-4]
rgb_path = extract_bag_dir + '/rgb'
semantic_confidence_path = extract_bag_dir + '/semantic_confidence'
semantic_labels_path = extract_bag_dir + '/semantic_labels'
semantic_labels_rgb_path = extract_bag_dir + '/semantic_labels_rgb'
semantic_labels_rgb_origin_path = extract_bag_dir + '/semantic_labels_rgb_origin'

## Try to remove tree; if failed show an error using try...except on screen
try:
    shutil.rmtree(extract_bag_dir)
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))
try:
    os.makedirs(extract_bag_dir)
    os.makedirs(rgb_path)
    os.makedirs(semantic_confidence_path)
    os.makedirs(semantic_labels_path)
    os.makedirs(semantic_labels_rgb_path)
    os.makedirs(semantic_labels_rgb_origin_path)
except OSError:
    if not os.path.isdir(extract_bag_dir):
        raise

odom_file = extract_bag_dir + '/odom.txt'
odom_file_handle = open(odom_file, 'w')
odom_file_handle.writelines("timestamp x y z qw qx qy qz"+"\n")

id_to_ros_time_file=extract_bag_dir+"/id_to_ros_time.txt"
id_to_ros_time_handle = open(id_to_ros_time_file, 'w')
id_to_ros_time_handle.writelines("id timestamp"+"\n")

# with open(bbs_file,'w') as f:    #设置文件对象
#     f.writelines(header+"\n")
#     for object_info in object_infos:
#         f.writelines(object_info+"\n")                 #将字符串写入文件中

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
    cnt_dict={}
    cnt_dict[1]=0
    cnt_dict[2]=0
    cnt_dict[3]=0
    cnt_dict[4]=0



    for topic, msg, t in bag_in.read_messages():
        i=i+1
        if topic == topic_odom:
            pose ="{} {} {} {} {} {} {} {}".format(msg.header.stamp.to_nsec(),
                                                            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                                                            msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                                            msg.pose.pose.orientation.z)
            odom_file_handle.write(pose+"\n")
        elif topic == topic_rgb:
            cnt_dict[1]=cnt_dict[1]+1
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            h=cv_image.shape[0]
            w=cv_image.shape[1]
            c=cv_image.shape[2]
            cv_image=cv_image[0:h, 0:np.int32(w/2), 0:c]
            cv_image = cv2.resize(cv_image, g_shape)
            cv2.imwrite(os.path.normpath(os.path.join(rgb_path, str(cnt_dict[1]) + '.png')), cv_image)

            # write id to ros time
            id_to_ros_time="{} {}".format(cnt_dict[1], msg.header.stamp.to_nsec())
            id_to_ros_time_handle.writelines(id_to_ros_time+"\n")
        elif topic == topic_semantic_confidence:
            cnt_dict[2]=cnt_dict[2]+1
            cv_image = CvBridge().imgmsg_to_cv2(msg, "mono8")
            cv_image = cv2.resize(cv_image, g_shape)
            cv2.imwrite(os.path.normpath(os.path.join(semantic_confidence_path, str(cnt_dict[2]) + '.png')), cv_image)
        elif topic == topic_semantic_labels:
            cnt_dict[3]=cnt_dict[3]+1
            cv_image = CvBridge().imgmsg_to_cv2(msg, "mono8")
            cv_image = cv2.resize(cv_image, g_shape)
            cv2.imwrite(os.path.normpath(os.path.join(semantic_labels_path, str(cnt_dict[3]) + '.png')), cv_image)

            semantic = np.asarray(map_label_to_color(cv_image))
            semantic = np.stack(
                (semantic[0], semantic[1], semantic[2]), axis=2)
            semantic = semantic.astype(np.uint8)
            cv2.imwrite(os.path.normpath(os.path.join(semantic_labels_rgb_path, str(cnt_dict[3]) + '.png')), semantic)
        elif topic == topic_semantic_rgb:
            cnt_dict[4]=cnt_dict[4]+1
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, g_shape)
            cv2.imwrite(os.path.normpath(os.path.join(semantic_labels_rgb_origin_path, str(cnt_dict[4]) + '.png')), cv_image)


        # write image
        # outbag.write(topic, msg, msg.header.stamp)
        if 0:
            if topic == '/px/perception/semantic_labels':
                msg_bak = msg

                cv_image = CvBridge().imgmsg_to_cv2(msg, "mono8")

                topic = '/px/perception/visual/main1'
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

        percent = (i - start_index) / duration
        if percent - last_percent > interval:
            last_percent = percent

            now_time = time.time()
            remain_time = float(now_time - start_time) * (duration-i) / (i * 3600)
            status(40, percent, i, remain_time)

status(40, 1, duration, 0)

odom_file_handle.close()
id_to_ros_time_handle.close()

print("finished")