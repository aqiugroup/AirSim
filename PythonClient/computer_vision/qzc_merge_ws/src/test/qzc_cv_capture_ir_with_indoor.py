# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

# import setup_path
from noise_model import *
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd
import airsim

import pprint
import tempfile
import os
import sys
import time

import numpy as np
np.set_printoptions(linewidth=200, formatter={
                    'float': lambda x: "{:6.3f}".format(x)})


def status(length, percent, index=0, remain_time=0):
    sys.stdout.write('\x1B[2K')  # Erase entire current line
    sys.stdout.write('\x1B[0E')  # Move to the beginning of the current line
    progress = "Progress: ["
    for i in range(0, length):
        if i < length * percent:
            progress += '='
        else:
            progress += ' '
    remain_time_str = f'{remain_time:.2f}'  # 小数点后保留两位，四舍五入
    progress += "] " + str(round(percent * 100.0, 2)) + "%" + \
        " : " + str(index) + " : " + remain_time_str + "h"
    # progress += "] " + str(round(percent * 100.0, 2)) + "%\n"
    sys.stdout.write(progress)
    sys.stdout.flush()


def open_txt(csv_path):
    # tfs = pd.read_csv(csv_path)
    # tfs = pd.read_csv(csv_path, header=None)
    # columts = ['lidar_end', 'pred_qx']
    # tfs = pd.read_csv(csv_path)
    tfs = pd.read_csv(csv_path, delimiter='\s+')
    # tfs = pd.read_csv(csv_path, sep='\s+', header=0, names=columts, index_col=False)

    return tfs


def read_info_frome_csv(tfs):
    # lidar_end, pred_qx, pred_qy, pred_qz, pred_qw, pred_px, pred_py, pred_pz,
    # update_qx, update_qy, update_qz, update_qw, update_px, update_py, update_pz,
    # vio_qx, vio_qy, vio_qz, vio_qw, vio_px, vio_py, vio_pz
    return tfs['lidar_end'], tfs['pred_qx'], tfs['pred_qy'], tfs['pred_qz'], tfs['pred_qw'], tfs['pred_px'], tfs['pred_py'], tfs['pred_pz'], \
        tfs['update_qx'], tfs['update_qy'], tfs['update_qz'], tfs['update_qw'], tfs['update_px'], tfs['update_py'], tfs['update_pz'], \
        tfs['vio_qx'], tfs['vio_qy'], tfs['vio_qz'], tfs['vio_qw'], tfs['vio_px'], tfs[
            'vio_py'], tfs['vio_pz'], tfs['update_grav_x'], tfs['update_grav_y'], tfs['update_grav_z']


if __name__ == '__main__':
    # sys.argv[1]
    file_path = "/Users/aqiu/Documents/1_study/10_workspace/02_clion/qzc_slam/17_lio_ws2/Log/mat_pre.txt"
    cur_dir = file_path[:file_path.rfind(os.path.sep)] + os.path.sep
    print(cur_dir)

    pp = pprint.PrettyPrinter(indent=4)
    client = airsim.VehicleClient()
    # client = airsim.MultirotorClient()
    client.confirmConnection()

    for camera_id in range(2):
        camera_info = client.simGetCameraInfo(str(camera_id))
        print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))
    tmp_dir = os.path.join(cur_dir, "airsim_drone")
    print("Saving images to %s" % tmp_dir)
    try:
        for n in range(3):
            os.makedirs(os.path.join(tmp_dir, str(n)))
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    # ---------------------------- for debug begin --------------------------------
    fin0 = open(file_path, "r")
    pose_len = len(fin0.readlines())
    fin0.close()
    # This is for logging progress
    duration = pose_len
    interval = 1 / duration
    last_percent = 0
    start_time = time.time()
    start_index = 0
    index = 0
    status(40, 0, index, 1000)
    print("\n")
    # ---------------------------- for debug end --------------------------------

    df = open_txt(file_path)
    lidar_end, pred_qx, pred_qy, pred_qz, pred_qw, pred_px, pred_py, pred_pz,\
        update_qx, update_qy, update_qz, update_qw, update_px, update_py, update_pz,\
        vio_qx, vio_qy, vio_qz, vio_qw, vio_px, vio_py, vio_pz, update_grav_x, update_grav_y, update_grav_z = read_info_frome_csv(
            df)
    vio_qx = np.array(vio_qx)
    vio_qy = np.array(vio_qy)
    vio_qz = np.array(vio_qz)
    vio_qw = np.array(vio_qw)
    vio_px = np.array(vio_px)
    vio_py = np.array(vio_py)
    vio_pz = np.array(vio_pz)

    # rot_x_90 = R.from_euler('x', -90, degrees=True)
    # rot_z_90 = R.from_euler('z', -90, degrees=True)
    # print("!!!!!!!!aaaaa ", rot_z_90.as_matrix() * rot_x_90.as_matrix())

    pose = client.simGetVehiclePose()
    rot = R.from_quat([[pose.orientation.x_val, pose.orientation.y_val,
                        pose.orientation.z_val, pose.orientation.w_val]])
    print("init angle: ", rot.as_euler("xyz", degrees=True))
    print("init vehicle pose1: x={:6.3f}, y={:6.3f}, z={:6.3f}".format(
        pose.position.x_val, pose.position.y_val, pose.position.z_val))
    camera_info = client.simGetCameraInfo(str(0))
    print("camera   pose: x={:6.3f}, y={:6.3f}, z={:6.3f}".format(
        pose.position.x_val, pose.position.y_val, pose.position.z_val))

    last_z_diff = 0

    print(">>>>>>>>>>simGetGroundTruthKinematics")
    pose = client.simGetGroundTruthKinematics()
    p = pp.pprint(pose)
    angles = airsim.to_eularian_angles(pose.orientation)

    print(">>>>>>>>>>simGetGroundTruthEnvironment")
    pose1 = client.simGetGroundTruthEnvironment()
    p = pp.pprint(pose1)

    print("simGetGroundTruthKinematics: x={:6.3f}, y={:6.3f}, z={:6.3f}, rpy={:6.3f}, {:6.3f},{:6.3f} qwqxqyqz={:6.3f}, {:6.3f},{:6.3f},{:6.3f}".format(
        pose.position.x_val, pose.position.y_val, pose.position.z_val,
        angles[0], angles[1], angles[2],
        pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val))
    print("simGetGroundTruthEnvironment: x={:6.3f}, y={:6.3f}, z={:6.3f}".format(
        pose1.position.x_val, pose1.position.y_val, pose1.position.z_val))

    T_start_to_ue_offset = np.eye(4)
    # T_start_to_ue_offset[:3, 3] = [0, 0, -3]
    T_start_to_ue_offset[:3, 3] = [0, 2, -2]
    # rot_z_135 = R.from_euler('z', 135, degrees=True)
    rot_z_135 = R.from_euler('z', 40, degrees=True)
    T_start_to_ue_offset[:3, :3] = rot_z_135.as_matrix()

    # T_start_to_ue_offset[:3, 3] = [
    #     pose1.geo_point.latitude, pose1.geo_point.longitude, -pose1.geo_point.altitude]
    # rot_to_ue_offet = R.from_quat(
    #     [[pose2.orientation.x_val, pose2.orientation.y_val, pose2.orientation.z_val, pose2.orientation.w_val]])
    # T_start_to_ue_offset[:3, :3] = rot_to_ue_offet.as_matrix()

    rot_x_180 = R.from_euler('x', 180, degrees=True)
    # rot_x_180 = R.from_euler('x', 0, degrees=True)
    # print("rot_x_180:\n", rot_x_180.as_matrix())
    T_airsim_traj = np.eye(4)
    # T_airsim_traj[:3, 3] = [0, 1, -1]
    # T_airsim_traj[:3, :3] = rot_x_180.as_matrix()

    C = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ])
    C = np.array([
        [0,  0,  1],
        [1,  0,  0],
        [0,  1,  0],
    ])
    c_x_180 = R.from_matrix(C)
    print("euler: ", c_x_180.as_euler("xyz", degrees=True))

    data_len = len(vio_qx)
    for i in range(0, data_len):
        if i <= 100:
            continue
        # timestamp = parts[1]  # ms
        # timestamp = float(parts[1]) / 1000.0 # s

        # print("{} {} {} {} {} {} {}".format(
        #     vio_qx[i], vio_qy[i], vio_qz[i], vio_qw[i], vio_px[i], vio_py[i], vio_pz[i]))

        pos_x = vio_px[i]
        pos_y = vio_py[i]
        pos_z = vio_pz[i]

        quat_x = vio_qx[i]
        quat_y = vio_qy[i]
        quat_z = vio_qz[i]
        quat_w = vio_qw[i]

        rot = R.from_quat([[quat_x, quat_y, quat_z, quat_w]])
        trans = np.array([pos_x, pos_y, pos_z]).reshape(-1, 1)
        T_origin = np.concatenate(
            (np.matrix(rot.as_matrix()[0]),  np.matrix(trans)), axis=1)
        T_origin = np.concatenate(
            (T_origin, np.matrix([[0, 0, 0, 1]])), axis=0)

        T_final = T_start_to_ue_offset @ T_airsim_traj @ T_origin
        # pos_x = T_final[0, 3]
        pos_x = -T_final[0, 3]
        pos_y = T_final[1, 3]
        pos_z = T_final[2, 3]

        R1 = R.from_matrix(T_final[:3, :3])
        quat1 = R1.as_quat()
        quat_x = quat1[0]
        quat_y = quat1[1]
        quat_z = quat1[2]
        quat_w = quat1[3]

        # if i == 0:
        #     pos_x1 = pos_x
        #     pos_y1 = pos_y
        #     pos_z1 = pos_z
        #     quat_w1 = quat_w
        #     quat_x1 = quat_x
        #     quat_y1 = quat_y
        #     quat_z1 = quat_z
        # else:
        #     # pos_x1 -= 0.1
        #     pos_x = pos_x1
        #     # pos_y1 -= 0.1
        #     pos_y = pos_y1
        #     # pos_z = pos_z1 - last_z_diff
        #     pos_z1 -= 0.1
        #     pos_z = pos_z1
        #     quat_w = quat_w1
        #     quat_x = quat_x1
        #     quat_y = quat_y1
        #     quat_z = quat_z1

        # print("{} {} {} {} {} {} {}".format(
        #     quat_x, quat_y, quat_z, quat_w, pos_x, pos_y, pos_z))

        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(
            pos_x, pos_y, pos_z), airsim.Quaternionr(quat_x, quat_y, quat_z, quat_w)), True)
        time.sleep(0.1)

        # if i == 0:
        #     pose = client.simGetVehiclePose()
        #     last_z_diff = pos_z1 - pose.position.z_val

        print_log = False
        print_log = True
        if print_log:
            print(">>>>>>>>>>simGetVehiclePose")
            print("vehicle pose               : x={:6.3f}, y={:6.3f}, z={:6.3f},     i {}                    qwqxqyqz={:6.3f}, {:6.3f},{:6.3f},{:6.3f}".format(
                pos_x, pos_y, pos_z, i, quat_w, quat_x, quat_y, quat_z))

            pose = client.simGetVehiclePose()
            angles = airsim.to_eularian_angles(pose.orientation)
            # p = pp.pprint(pose)
            print("simGetVehiclePose          : x={:6.3f}, y={:6.3f}, z={:6.3f}, rpy={:6.3f}, {:6.3f},{:6.3f} qwqxqyqz={:6.3f}, {:6.3f},{:6.3f},{:6.3f}".format(
                pose.position.x_val, pose.position.y_val, pose.position.z_val,
                angles[0], angles[1], angles[2],
                pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val))

            # print(">>>>>>>>>>simGetGroundTruthKinematics")
            pose = client.simGetGroundTruthKinematics()
            # p = pp.pprint(pose)
            angles = airsim.to_eularian_angles(pose.orientation)
            print("simGetGroundTruthKinematics: x={:6.3f}, y={:6.3f}, z={:6.3f}, rpy={:6.3f}, {:6.3f},{:6.3f} qwqxqyqz={:6.3f}, {:6.3f},{:6.3f},{:6.3f}".format(
                pose.position.x_val, pose.position.y_val, pose.position.z_val,
                angles[0], angles[1], angles[2],
                pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val))

            # state = client.getMultirotorState()
            # angles = airsim.to_eularian_angles(
            #     state.kinematics_estimated.orientation)
            # print("getMultirotorState         : x={:6.3f}, y={:6.3f}, z={:6.3f}, rpy={:6.3f}, {:6.3f},{:6.3f} qwqxqyqz={:6.3f}, {:6.3f},{:6.3f},{:6.3f}".format(
            #     state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val,
            #     angles[0], angles[1], angles[2],
            #     state.kinematics_estimated.orientation.w_val, state.kinematics_estimated.orientation.x_val, state.kinematics_estimated.orientation.y_val, state.kinematics_estimated.orientation.z_val))

            # print(">>>>>>>>>>simGetGroundTruthEnvironment")
            pose = client.simGetGroundTruthEnvironment()
            # p = pp.pprint(pose)
            print("simGetGroundTruthEnvironment: x={:6.3f}, y={:6.3f}, z={:6.3f}".format(
                pose.position.x_val, pose.position.y_val, pose.position.z_val))

            # camera_info = client.simGetCameraInfo(str(0))
            # print("camera   pose: x={:6.3f}, y={:6.3f}, z={:6.3f},  qwqxqyqz={:6.3f}, {:6.3f},{:6.3f},{:6.3f}".format(
            #     camera_info.pose.position.x_val, camera_info.pose.position.y_val, camera_info.pose.position.z_val,
            #     camera_info.pose.orientation.w_val, camera_info.pose.orientation.y_val, camera_info.pose.orientation.z_val, camera_info.pose.orientation.z_val))
            # p = pp.pprint(camera_info)

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
            airsim.ImageRequest(
                "0", airsim.ImageType.DepthPerspective, True, False),
            airsim.ImageRequest("0", airsim.ImageType.Infrared, False, True),
            # airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            # airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
            # airsim.ImageRequest("0", airsim.ImageType.Infrared,False, False),
            # airsim.ImageRequest("0", airsim.ImageType.Segmentation,False, False)
        ])

        # time.sleep(3)

        # for i, response in enumerate(responses):
        #     if response.pixels_as_float:
        #         # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
        #         airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir,  str(i), str(
        #             timestamp) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
        #     else:
        #         # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
        #         airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(
        #             timestamp) + "_" + str(i) + '.png')), response.image_data_uint8)

        # if print_log:
        # print("vehicle pose2: x={:6.3f}, y={:6.3f}, z={:6.3f}".format(
        #     pose.position.x_val, pose.position.y_val, pose.position.z_val))
        # pose = client.simGetVehiclePose()
        # pp.pprint(pose)

        index = index + 1
        percent = (index - start_index) / duration
        if percent - last_percent > interval:
            last_percent = percent

            now_time = time.time()
            remain_time = float(now_time - start_time) * \
                (duration-index) / (index * 3600)
            # status(40, percent, index, remain_time)
    status(40, 1, duration, 0)

    # currently reset() doesn't work in CV mode. Below is the workaround
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(
        0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)
