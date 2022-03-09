# coding=utf-8
import keyboard
import airsim


import cv2
import numpy as np
import time
import threading
import os
import re
import subprocess
import random
import math

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(angles1) :
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0]*3.141592653589793/180.0
    theta[1] = angles1[1]*3.141592653589793/180.0
    theta[2] = angles1[2]*3.141592653589793/180.0
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    #print('dst:', R)
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    rvecstmp = np.zeros((1, 1, 3), dtype=np.float64)
    rvecs,_ = cv2.Rodrigues(R, rvecstmp)
    #print()
    return R,rvecs,x,y,z

def rotationMatrixToEulerAngles(rvecs):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvecs, R)
    sy = math.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2])
    sz = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    print('sy=',sy, 'sz=', sz)
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    #print('dst:', R)
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    return x,y,z

# https://zhuanlan.zhihu.com/p/336357646
# https://www.bilibili.com/video/BV1oa4y1v7TY?p=2
# https://www.jianshu.com/p/5e130c04a602
# https://zhuanlan.zhihu.com/p/85108850
#旋转矩阵 欧拉角

eulerAngles = np.zeros((3, 1), dtype=np.float64)
eulerAngles[0] = -90.0
eulerAngles[1] = 0.0
eulerAngles[2] = -90.0
R,rvecstmp,x,y,z = eulerAnglesToRotationMatrix(eulerAngles)
print('输入欧拉角：\n', eulerAngles)
print('旋转转矩：\n', R)
print('旋转向量：\n', rvecstmp)
print('计算后的欧拉角：\n', rotationMatrixToEulerAngles(rvecstmp))


import pprint
from decimal import *
def print_pose(client):
    # 1 vehicle pose
    pose = client.simGetVehiclePose()
    print("vehicle pose: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))

    angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))

     # 2 multirotor pose
    state = client.getMultirotorState()
    # s = pprint.pformat(state)
    # print("state: %s" % s)
    print("multirotor pose: x={}, y={}, z={}".format(state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val))

    angles = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
    print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))


    # kinematics = client.simGetGroundTruthKinematics()
    # environment = client.simGetGroundTruthEnvironment()

    # print("Kinematics: %s\nEnvironemt %s" % (
    #     pprint.pformat(kinematics), pprint.pformat(environment)))

     # 3 imu pose

     # 4 camera pose
    img_position = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])[0].camera_position
    img_orientation = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])[0].camera_orientation
    print("camera pose: x={}, y={}, z={}".format(img_position.x_val, img_position.y_val, img_position.z_val))

    angles = airsim.to_eularian_angles(img_orientation)
    print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
    # prefix="/Users/aqiu/Documents/AirSim"
    # state = client.getMultirotorState(vehicle_name = robot_name)
    # gt_name = prefix + '/groundtruth/data.tum'
    # gt_writer = open(gt_name, 'a')
    # gt_writer.write(
    #     str(Decimal(state.timestamp) / Decimal(1e9)) + ' ' +
    #     str(state.kinematics_estimated.position.x_val) + ' ' +
    #     str(state.kinematics_estimated.position.y_val) + ' ' +
    #     str(state.kinematics_estimated.position.z_val) + ' ' +
    #     str(state.kinematics_estimated.orientation.x_val) + ' ' +
    #     str(state.kinematics_estimated.orientation.y_val) + ' ' +
    #     str(state.kinematics_estimated.orientation.z_val) + ' ' +
    #     str(state.kinematics_estimated.orientation.w_val) + '\n'
    # )
    # z = -1
    # print("make sure we are hovering at {} meters...".format(-z))
    # client.moveToZAsync(z, 5).join()

def callBackFunc(x):
    w = keyboard.KeyboardEvent('down', 28, 'w')             # 前进
    s = keyboard.KeyboardEvent('down', 28, 's')             # 后退
    a = keyboard.KeyboardEvent('down', 28, 'a')             # 左移
    d = keyboard.KeyboardEvent('down', 28, 'd')             # 右移
    up = keyboard.KeyboardEvent('down', 28, 'up')           # 上升
    down = keyboard.KeyboardEvent('down', 28, 'down')       # 下降
    left = keyboard.KeyboardEvent('down', 28, 'left')       # 左转
    right = keyboard.KeyboardEvent('down', 28, 'right')     # 右转
    k = keyboard.KeyboardEvent('down', 28, 'k')             # 获取控制
    l = keyboard.KeyboardEvent('down', 28, 'l')             # 释放控制
    h = keyboard.KeyboardEvent('down', 28, 'h')             # 悬停


    if x.event_type == 'down' and x.name == w.name:
        # 前进
        client.moveByVelocityBodyFrameAsync(3, 0, 0, 0.5)
        print("前进")
    elif x.event_type == 'down' and x.name == s.name:
        # 后退
        client.moveByVelocityBodyFrameAsync(-3, 0, 0, 0.5)
        print("后退")
    elif x.event_type == 'down' and x.name == a.name:
        # 左移
        client.moveByVelocityBodyFrameAsync(0, -2, 0, 0.5)
        print("左移")
    elif x.event_type == 'down' and x.name == d.name:
        # 右移
        client.moveByVelocityBodyFrameAsync(0, 2, 0, 0.5)
        print("右移")
    elif x.event_type == 'down' and x.name == up.name:
        # 上升
        client.moveByVelocityBodyFrameAsync(0, 0, -0.5, 0.5)
        print("上升")
    elif x.event_type == 'down' and x.name == down.name:
        # 下降
        client.moveByVelocityBodyFrameAsync(0, 0, 0.5, 0.5)
        print("下降")
    elif x.event_type == 'down' and x.name == left.name:
        # 左转
        client.rotateByYawRateAsync(-20, 0.5)
        print("左转")
    elif x.event_type == 'down' and x.name == right.name:
        # 右转
        client.rotateByYawRateAsync(20, 0.5)
        print("右转")
    elif x.event_type == 'down' and x.name == k.name:
        # 无人机起飞
        # get control
        client.enableApiControl(True)
        print("get control")
        # unlock
        client.armDisarm(True)
        print("unlock")
        # Async methods returns Future. Call join() to wait for task to complete.
        client.takeoffAsync().join()
        print("takeoff")

        z = -1
        print("make sure we are hovering at {} meters...".format(-z))
        client.moveToZAsync(z, 0.5).join()

    elif x.event_type == 'down' and x.name == l.name:
        # 无人机降落
        client.landAsync().join()
        print("land")
        # lock
        client.armDisarm(False)
        print("lock")
        # release control
        client.enableApiControl(False)
        print("release control")
    elif x.event_type == 'down' and x.name == h.name:
        # 没有按下按键
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.5).join()
        client.hoverAsync().join()  # 第四阶段：悬停6秒钟
        print("悬停")

    print_pose(client)


if __name__ == '__main__':
    # 建立脚本与AirSim环境的连接
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # 监听键盘事件，执行回调函数
    keyboard.hook(callBackFunc)
    keyboard.wait()

    client.enableApiControl(False)
