# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path
import airsim

import pprint
import tempfile
import os
import sys
import time


file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_rec.txt" # sys.argv[1]
cur_dir = file_path[:file_path.rfind(os.path.sep)] + os.path.sep
print(cur_dir)

pp = pprint.PrettyPrinter(indent=4)
client = airsim.VehicleClient()
client.confirmConnection()

# airsim.wait_key('Press any key to get camera parameters')
for camera_id in range(2):
    camera_info = client.simGetCameraInfo(str(camera_id))
    print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))
tmp_dir = os.path.join(cur_dir, "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    for n in range(3):
        os.makedirs(os.path.join(tmp_dir, str(n)))
except OSError:
    if not os.path.isdir(tmp_dir):
        raise


fin = open(file_path, "r")
line = fin.readline().strip()
line = fin.readline().strip()
while (line):
    parts = line.split("\t")
    timestamp = parts[1] # ms
    # timestamp = float(parts[1]) / 1000.0 # s

    pos_x = float(parts[2])
    pos_y =float( parts[3])
    pos_z = float(parts[4])

    quat_w = float(parts[5])
    quat_x = float(parts[6])
    quat_y = float(parts[7])
    quat_z = float(parts[8])

    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos_x, pos_y, pos_z), airsim.Quaternionr(quat_x,quat_y,quat_z,quat_w)), True)
    time.sleep(0.1)

    responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene),
    airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
    # airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
    airsim.ImageRequest("0", airsim.ImageType.Infrared)])

    for i, response in enumerate(responses):
        if response.pixels_as_float:
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir,  str(i), str(timestamp) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(timestamp) + "_" + str(i) + '.png')), response.image_data_uint8)

    pose = client.simGetVehiclePose()
    pp.pprint(pose)

    time.sleep(3)

    line = fin.readline().strip()


# pp = pprint.PrettyPrinter(indent=4)

# client = airsim.VehicleClient()

# airsim.wait_key('Press any key to get camera parameters')
# for camera_id in range(2):
# for camera_id in range(2):
#     camera_info = client.simGetCameraInfo(str(camera_id))
#     print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))

# airsim.wait_key('Press any key to get images')
# tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
# print ("Saving images to %s" % tmp_dir)
# try:
#     for n in range(3):
#         os.makedirs(os.path.join(tmp_dir, str(n)))
# except OSError:
#     if not os.path.isdir(tmp_dir):
#         raise

# for x in range(50): # do few times
#     #xn = 1 + x*5  # some random number
#     client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, 0, -2), airsim.to_quaternion(0, 0, 0)), True)
#     time.sleep(0.1)

#     responses = client.simGetImages([
#         airsim.ImageRequest("0", airsim.ImageType.Scene),
#         airsim.ImageRequest("1", airsim.ImageType.Scene),
#         airsim.ImageRequest("2", airsim.ImageType.Scene)])

#     for i, response in enumerate(responses):
#         if response.pixels_as_float:
#             print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
#             airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, str(x) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
#         else:
#             print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
#             airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(x) + "_" + str(i) + '.png')), response.image_data_uint8)

#     pose = client.simGetVehiclePose()
#     pp.pprint(pose)

#     time.sleep(3)

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)