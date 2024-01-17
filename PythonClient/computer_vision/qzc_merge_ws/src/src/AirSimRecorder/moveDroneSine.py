import airsim
import numpy as np
import time
import subprocess
import os
import keyboard

# Load path
path = []
is_first_line = True
with open('./sine_3d_waypoints.txt', 'r') as f:
    for line in f:
        items = line.strip().split()
        if(is_first_line):
            is_first_line = False
            start_x = float(items[0])
            start_y = float(items[1])
            start_z = float(items[2])
            continue
        path.append(airsim.Vector3r(float(items[0]), float(items[1]), float(items[2])))

# Setup airsim drone
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("taking off..")
client.takeoffAsync().join()
time.sleep(1)

print("moving to start point..")
client.moveToZAsync(-15, 3).join()
client.moveToPositionAsync(start_x, start_y, start_z, 3).join()
time.sleep(1)

print("rotate towards center..")
client.rotateToYawAsync(90,3).join()
time.sleep(1)

# Delete previous recordings
os.system('rd /s /q captures')
os.system('del gt_data.txt imu_data.txt')

# Wait for 'g' key press
print("Press 'g' (from inside unreal viewport) to start recording")
keyboard.wait('g')

# Start motion and recording
print("starting recording..")
camera_process = subprocess.Popen(['python', 'getCamera.py'])
imu_gt_process = subprocess.Popen(['python', 'getDroneData.py'])
time.sleep(2)

print("exciting z..")
client.moveToZAsync(start_z-3, 3).join()
client.moveToZAsync(start_z, 3).join()

time.sleep(0.5)

client.moveToZAsync(start_z-3, 3).join()
client.moveToZAsync(start_z, 3).join()

print("flying on smooth path..")
client.moveOnPathAsync(path, 3, np.inf, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,90)).join()

# End motion and recording
client.moveToPositionAsync(start_x, start_y, start_z, 3).join()
time.sleep(2)
client.enableApiControl(False)
time.sleep(2)
print("connection closed...")
