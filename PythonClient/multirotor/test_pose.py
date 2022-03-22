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

from tf import transformations as tfs

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
eulerAngles[0] = 90.0
eulerAngles[1] = 0.0
eulerAngles[2] = 0.0
R,rvecstmp,x,y,z = eulerAnglesToRotationMatrix(eulerAngles) # 外旋 xyz : 左乘 = Rot(z) * Rot(y) * Rot(x) = Rot(gamma) * Rot(beta) * Rot(alpha)
print('输入欧拉角：\n', eulerAngles)
print('旋转转矩：\n', R)
print('旋转向量：\n', rvecstmp)
print('计算后的欧拉角：\n', rotationMatrixToEulerAngles(rvecstmp))

C1 = np.array([
        [ 0,  -1,  0,  0],
        [ 0,  0, -1,  0],
        [ -1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])
C2 = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1]
    ])
C3=C1.T@C2
print(C3)
C4 = np.array([
        [ 0,  1,  0,  0],
        [ 0,  0, 1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])
C5=C4.T@C2
print(C5)

alpha = eulerAngles[0]*3.141592653589793/180.0
beta  = eulerAngles[1]*3.141592653589793/180.0
gamma = eulerAngles[2]*3.141592653589793/180.0
Re = tfs.euler_matrix(alpha, beta, gamma, 'sxyz') # 外旋 xyz : 左乘 = Rot(z) * Rot(y) * Rot(x) = Rot(gamma) * Rot(beta) * Rot(alpha)
# print("tf sxyz R: \n", Re)
Re = tfs.euler_matrix(gamma, beta, alpha, 'rzyx') # 内旋 zyx : 右乘 = Rot(z) * Rot(y) * Rot(x) = Rot(gamma) * Rot(beta) * Rot(alpha)
# print("tf rzyx R: \n", Re)

# alpha, beta, gamma = 0.123, -1.234, 2.345
# origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
# I = tfs.identity_matrix()
# Rx = tfs.rotation_matrix(alpha, xaxis)
# Ry = tfs.rotation_matrix(beta, yaxis)
# Rz = tfs.rotation_matrix(gamma, zaxis)
# R = tfs.concatenate_matrices(Rx, Ry, Rz)
# euler = tfs.euler_from_matrix(R, 'rxyz')
# print(euler)
# Re = tfs.euler_matrix(alpha, beta, gamma, 'rxyz')
# print(tfs.is_same_transform(R, Re))
# al, be, ga = tfs.euler_from_matrix(Re, 'rxyz')
# print([al, be, ga])
# print(tfs.is_same_transform(R, tfs.euler_matrix(alpha, beta, gamma, 'rxyz')))
# print(tfs.is_same_transform(R, tfs.euler_matrix(al, be, ga, axes='sxyz')))

import pprint
pp = pprint.PrettyPrinter(indent=1)

from decimal import *
def print_pose(client):
    print(">>>>>>>>>>simGetGroundTruthEnvironment")
    pose = client.simGetGroundTruthEnvironment()
    p =pp.pprint(pose)
    # print("simGetGroundTruthEnvironment: %s" % pose)

    print(">>>>>>>>>>simGetGroundTruthKinematics")
    pose = client.simGetGroundTruthKinematics()
    p =pp.pprint(pose)
    # print("simGetGroundTruthKinematics: %s" % pose)



    # 1 vehicle pose
    print(">>>>>>>>>>simGetVehiclePose")
    pose = client.simGetVehiclePose()
    # print("vehicle pose: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
    p =pp.pprint(pose)
    # print("simGetVehiclePose: %s" % pose)

    # angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    # print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))

     # 2 multirotor pose
    print(">>>>>>>>>>getMultirotorState")
    state = client.getMultirotorState()
    p = pp.pformat(state)
    # print("getMultirotorState: %s" % p)
    # print("multirotor pose: x={}, y={}, z={}".format(state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val))
    # angles = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
    # print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))



     # 3 imu pose
    print(">>>>>>>>>>getImuData")
    pose = client.getImuData()
    p =pp.pprint(pose)
    # print("getImuData: %s" % pose)

     # 4 camera pose
    # img_position = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])[0].camera_position
    # img_orientation = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])[0].camera_orientation
    # print("camera pose: x={}, y={}, z={}".format(img_position.x_val, img_position.y_val, img_position.z_val))
    print(">>>>>>>>>>simGetCameraInfo")
    for camera_id in range(1):
        camera_info = client.simGetCameraInfo(str(camera_id))
        p = pp.pprint(camera_info)
        # print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))

    # angles = airsim.to_eularian_angles(img_orientation)
    # print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
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


if __name__ == '__main__':
    # 建立脚本与AirSim环境的连接
    if 1:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)

        print("-------------origin--------------")
        print_pose(client)
        print("-------------set to --------------")
        pose = client.simGetVehiclePose()
        pos_x = pose.position.x_val + 1
        pos_y = pose.position.y_val
        pos_z = pose.position.z_val
        quat_x = pose.orientation.x_val
        quat_y = pose.orientation.y_val
        quat_z = pose.orientation.z_val
        quat_w = pose.orientation.w_val
        print(f"x:{pos_x} y:{pos_y} z:{pos_z} qx:{quat_x} qy:{quat_y} qz:{quat_z} qw:{quat_w}")
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos_x, pos_y, pos_z), airsim.Quaternionr(quat_x,quat_y,quat_z,quat_w)), True)
        time.sleep(3)
        print_pose(client)

        pose = client.simGetVehiclePose()
        pos_x = pose.position.x_val
        pos_y = pose.position.y_val
        pos_z =  pose.position.z_val - 4
        quat_x = pose.orientation.x_val
        quat_y = pose.orientation.y_val
        quat_z = pose.orientation.z_val
        quat_w = pose.orientation.w_val
        print(f"x:{pos_x} y:{pos_y} z:{pos_z} qx:{quat_x} qy:{quat_y} qz:{quat_z} qw:{quat_w}")
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos_x, pos_y, pos_z), airsim.Quaternionr(quat_x,quat_y,quat_z,quat_w)), True)

        # print("-------------set to --------------")
        # pos_x = 0.0
        # pos_y = 0.0
        # pos_z = 1.0
        # quat_x = 0.0
        # quat_y = 0.0
        # quat_z = 0.0
        # quat_w = 1.0
        # print(f"x:{pos_x} y:{pos_y} z:{pos_z} qx:{quat_x} qy:{quat_y} qz:{quat_z} qw:{quat_w}")
        # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos_x, pos_y, pos_z), airsim.Quaternionr(quat_x,quat_y,quat_z,quat_w)), True)
        # time.sleep(3)
        # print_pose(client)


        client.enableApiControl(False)
    else:
        client = airsim.VehicleClient()
        client.confirmConnection()
        print_pose(client)




'''
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "SpringArmChase",
  "ClockSpeed": 0.3,
  "Vehicles": {
    "drone_1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisionPassthrogh": false,
      "EnableCollisions": true,
      "AllowAPIAlways": true,
      "RC": {
        "RemoteControlID": 0,
        "AllowAPIWhenDisconnected": false
      },
      "Sensors": {
        "Imu" : {
          "SensorType": 2,
          "Enabled": true
        }
      },
      "Cameras": {
        "front_center_custom": {
          "CaptureSettings": [
            {
              "PublishToRos": 1,
              "ImageType": 0,
              "Width": 800,
              "Height": 600,
              "FOV_Degrees": 120,
              "DepthOfFieldFstop": 2.8,
              "DepthOfFieldFocalDistance": 200.0,
              "DepthOfFieldFocalRegion": 200.0,
              "TargetGamma": 1.5
            }
          ],
          "X": 0.50, "Y": 0, "Z": 0.10,
          "Pitch": -90, "Roll": 0, "Yaw": 0
        }
      },
      "X": 2, "Y": 0, "Z": -2,
      "Pitch": 0, "Roll": 2, "Yaw": 0
    }
  },
  "SubWindows": [
    {"WindowID": 0, "ImageType": 0, "CameraName": "front_center_custom", "Visible": true}
  ]
}
'''