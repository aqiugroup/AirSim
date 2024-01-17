import airsim
import os
import json
from time import sleep

client = airsim.MultirotorClient()
client.confirmConnection()

imu_data_file = open("imu_data.txt", 'w')
imu_data_file.write("t\t wx \t wy \t wz \t ax \t ay \t az \t qw \t qx \t qy \t qz\n")
gt_data_file = open("gt_data.txt", 'w')

imu_frequency = 200  # in hertz

while(client.isApiControlEnabled()):
    #sleep(1/imu_frequency)

    imu_data = client.getImuData()
    gt_data = client.simGetGroundTruthKinematics()
    gt_pos = gt_data.position
    gt_orient = gt_data.orientation

    imu_data_dict = {}
    imu_data_dict['angular_vel'] = [imu_data.angular_velocity.x_val, imu_data.angular_velocity.y_val, imu_data.angular_velocity.z_val]
    imu_data_dict['linear_acc'] = [imu_data.linear_acceleration.x_val, imu_data.linear_acceleration.y_val, imu_data.linear_acceleration.z_val]
    imu_data_dict['orientation'] = [imu_data.orientation.w_val, imu_data.orientation.x_val, imu_data.orientation.y_val, imu_data.orientation.z_val]
    imu_data_dict['timestamp'] = imu_data.time_stamp

    gt_data_dict = {}
    gt_data_dict['position'] = [gt_pos.x_val, gt_pos.y_val, gt_pos.z_val]
    gt_data_dict['orientation'] = [gt_orient.w_val, gt_orient.x_val, gt_orient.y_val, gt_orient.z_val]
    gt_data_dict['timestamp'] = imu_data.time_stamp

    imu_data_file.write(str(imu_data_dict['timestamp'])+ "\t" + str(imu_data_dict['angular_vel'][0]) + "\t" + str(imu_data_dict['angular_vel'][1]) + "\t" + str(imu_data_dict['angular_vel'][2]) + "\t" + str(imu_data_dict['linear_acc'][0]) + "\t" + str(imu_data_dict['linear_acc'][1]) + "\t" + str(imu_data_dict['linear_acc'][2]) + "\t" + str(imu_data_dict['orientation'][0]) + "\t" + str(imu_data_dict['orientation'][1]) + "\t" + str(imu_data_dict['orientation'][2]) + "\t" + str(imu_data_dict['orientation'][3]) + "\n")

    gt_data_file.write(str(gt_data_dict['timestamp']/1e9) + " " + str(gt_data_dict['position'][0]) + " " + str(gt_data_dict['position'][1])+ " " + str(gt_data_dict['position'][2]) + " " + str(gt_data_dict['orientation'][1])+ " " + str(gt_data_dict['orientation'][2]) + " " + str(gt_data_dict['orientation'][3]) + " " + str(gt_data_dict['orientation'][0]) + "\n")

imu_data_file.close()
gt_data_file.close()
print("imu reader disconnected...")
