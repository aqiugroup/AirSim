# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

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

file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_drone1/0/1646589723662_0.png" # sys.argv[1]
img = cv2.imread(file_path,0)
print(type(img))
print(img.shape)

file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_drone1/1/1646589723662_1.pfm" # sys.argv[1]
depth, scale = airsim.read_pfm(file_path)
print(file_path)
print(type(depth))

shape = depth.shape
print(shape)


if(0):
    # cv2::minMaxIdx(cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image,&min,&max);
    # cv2::Mat adjMap;
    # cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1) -> image.convertTo(adjMap,CV_8UC1,255/(max-min),-min);
    # imshow("Show DepthImage",adjMap);

    # https://bbs.huaweicloud.com/blogs/293564
    #图像翻转
    #0以X轴为对称轴翻转 >0以Y轴为对称轴翻转 <0X轴Y轴翻转
    img1 = cv2.flip(depth, 0)
    img2 = cv2.flip(depth, 1)
    img3 = cv2.flip(depth, -1)

    #显示图形
    titles = ['Source', 'Image1', 'Image2', 'Image3']
    images = [depth, img1, img2, img3]
    for i in range(4):
        plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


