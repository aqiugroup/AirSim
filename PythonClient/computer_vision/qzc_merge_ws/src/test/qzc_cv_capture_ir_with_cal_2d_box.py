# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

# import setup_path
import airsim

import pprint
import tempfile
import os
import sys
import time

import numpy as np
import cv2

from noise_model import *
DELAY = 0.01 # must be > 0

# RegEx for the object IDs of the objects of interest (that is, the objects
# around which you want the bounding boxes).
REGEX_OBJECTS_OF_INTEREST = [
    "Landscape[\w]*",
    "prp_trafficLight_Blueprint[\w]*",
    "prp_manhole[\w]*",
    "prp_light[\w]*",
    "prp_garbageCan[\w]*",
    "prp_bench[\w]*",
    "prp_fireHydrant[\w]*",
    "prp_chair[\w]*",
    "prp_table[\w]*",
    "CameraActor[\w]*",
    "prp_sewerGrate[\w]*",
    "prp_tableUmbrella[\w]*",
    "prp_electricalBox[\w]*",
    "Awning_mdl[\w]*",
    "PowerLine[\w]*",
    "prp_crosswalkSign[\w]*",
    "Building[\w]*",
    "prp_metalPillar[\w]*",
    "flg_tree[\w]*",


    "prp_chainlinkFence[\w]*",
    "prp_tarp[\w]*",
    "prp_vent[\w]*",
    "door_A[\w]*",
    "prp_ac[\w]*",
    "StoreSign[\w]*",
    "prp_pylong_Sml[\w]*",
    "flg_grass[\w]*",
    "prp_parkingMeter[\w]*",
    "Flg_hedge_Short[\w]*",
    "prp_metalFence[\w]*",
    "prp_barrier_Lrg[\w]*",
    "TrashBag[\w]*",
    "prp_potSquare[\w]*",
    "prp_cementBarrier[\w]*",
    "flg_fern[\w]*",
    "prp_flag[\w]*"
]

# Classes corresponding to the objects in the REGEX_OBJECTS_OF_INTEREST list above.
OBJECT_OF_INTEREST_CLASSES = [
    "ground",
    "trafficLight",
    "manhole",
    "fixLight",
    "garbageCan",
    "bench",
    "fireHydrant",
    "chair",
    "table",
    "CameraActor",
    "sewerGrate",
    "tableUmbrella",
    "electricalBox",
    "Awning",
    "PowerLine",
    "crosswalkSign",
    "Building",
    "metalPillar",
    "tree",


    "chainlinkFence",
    "tarp",
    "vent",
    "door",
    "ac",
    "StoreSign",
    "pylong_Sml",
    "grass",
    "parkingMeter",
    "hedge_Short",
    "metalFence",
    "barrier_Lrg",
    "TrashBag",
    "potSquare",
    "cementBarrier",
    "fern",
    "flag"
]

# An exhaustive list of all classes
# Note that this is separate to the above list, as we may want classes
# which are not currently exhibited in OBJECT_OF_INTEREST_CLASSES.
# CLASSES = ['mountainbike',
#            'car',
#            'truck',
#            'dog',
#            'horse',
#            'sheep',
#            'giraffe']

# Confusion matrix for object misclassification emulation.
# See https://en.wikipedia.org/wiki/Confusion_matrix for more information
CONFUSION_MATRIX = np.array(
        [[0.85714286,0.,0.14285714,0.,0.,0.,0.],
        [0.06666667,0.8,0.06666667,0.,0.,0.06666667,0.],
        [0.,0.,1.,0.,0.,0.,0.],
        [0.03636364,0.,0.01818182,0.94545455,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.01408451,0.,0.,0.98591549]]).T



##########################################################################
##########################################################################
##########################################################################

MESH_COLS = np.array([
          [57 ,181, 55],
          [6  ,108,153],
          [191,105,112],
          [72 ,121, 89],
          [64 ,225,190],
          [59 ,190,206],
          [36 ,13 , 81],
          [195,176,115],
          [27 ,171,161],
          [180,169,135],
          [199,26 ,29],
          [239,16 ,102],
          [146,107,242],
          [23 ,198,156],
          [160,89 ,49],
          [116,218,68],
        ])

# deal with swapped RB values
# cp = MESH_COLS.copy()
# MESH_COLS[:,0] = cp[:,2]
# MESH_COLS[:,2] = cp[:,0]

client = None

def setup():
    # Clear background
    msg = 'Setting everything to ID 255 (to clear unwanted objects)...'
    print(msg, end='')
    found = client.simSetSegmentationObjectID("[\w]*", 255, True);
    print(' ' * (65 - len(msg)) + ('[SUCCESS]' if found else '[FAILED!]'))

    # Set objects of interest
    for key, val in zip(REGEX_OBJECTS_OF_INTEREST, range(len(REGEX_OBJECTS_OF_INTEREST))):
        msg = 'Setting %s to ID %d...' % (key, val)
        print(msg, end='')
        found = client.simSetSegmentationObjectID(key, val, True);
        print(' ' * (40 - len(msg)) + ('[SUCCESS]' if found else '[FAILED!]'))


if __name__ == '__main__':
    file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_rec.txt" # sys.argv[1]
    cur_dir = file_path[:file_path.rfind(os.path.sep)] + os.path.sep
    print(cur_dir)

    pp = pprint.PrettyPrinter(indent=4)
    client = airsim.VehicleClient()
    client.confirmConnection()

    setup()

    # print_pose(client)

    # airsim.wait_key('Press any key to get camera parameters')
    for camera_id in range(2):
        camera_info = client.simGetCameraInfo(str(camera_id))
        print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))
    tmp_dir = os.path.join(cur_dir, "airsim_drone")
    print ("Saving images to %s" % tmp_dir)
    try:
        for n in range(4):
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
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("0", airsim.ImageType.Infrared,False, False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation,False, False)]
        )

        for i, response in enumerate(responses):
            if response.pixels_as_float:
                print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir,  str(i), str(timestamp) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
            else:
                print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(timestamp) + "_" + str(i) + '.png')), response.image_data_uint8)

                if  i == 3:
                    im = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                    im = im.reshape(response.height, response.width, 3)           # reshape array to 4 channel image
                    # im = np.flipud(im)                                            # original image is flipped vertically

                    # find unique colors and draw bounding boxes
                    colours = np.unique(im.reshape((-1,3)), axis=0)
                    print("",i, " ", len(colours))

                    # store the BBs and their classes
                    bbs = []
                    classes = []
                    for col in colours:
                        colours_of_interest = np.sum(np.all(MESH_COLS == col, axis=-1))

                        # ignore if this colour does not correspond to an object of interest.
                        if colours_of_interest == 0:
                            continue
                        elif colours_of_interest > 1:
                            print("[WARNING] Multiple objects have the same color in segmented view! Using lowest index...")

                        index = np.where(np.all(MESH_COLS == col, axis=-1))[0][0]
                        objClass = OBJECT_OF_INTEREST_CLASSES[index]

                        mask = np.all(im == col, axis=-1)
                        locs = np.array(np.where(mask))

                        # find the BB
                        min_x = np.min(locs[0,:])
                        max_x = np.max(locs[0,:])
                        min_y = np.min(locs[1,:])
                        max_y = np.max(locs[1,:])
                        bbs.append((min_x, max_x, min_y, max_y))
                        classes.append(objClass)

                    bbs_clean = np.array(bbs).copy()

                    #################################################
                    ##### Add noise to the BBs
                    #################################################
                    # # first do some mis-classification
                    # classes = misclassify(classes, CLASSES, CONFUSION_MATRIX)
                    # # now add some error to the BBs
                    # bbs = add_jitter(bbs, shape=im.shape, length_scale_fraction=0.05, center_error_fraction=0.05)
                    # bbs, classes = introduce_false_negatives(bbs, classes, p=0.01, min_size=4)
                    # bbs, classes = introduce_false_positives(bbs, CLASSES, classes, shape=im.shape, p=0.01)
                    # bbs, classes = merge_close_bbs(bbs, classes, area_similarity_factor=1.7, overlap_factor=0.5)
                    #################################################

                    quiet = False
                    if not quiet:
                        # Draw the images
                        # boxedTrue = draw_bbs_on_image(trueIm, bbs)
                        boxedSeg =  draw_bbs_on_image(im, bbs)

                        cv2.imshow('infrared:2, seg:3, i:'+str(i), boxedSeg)
                        cv2.waitKey(int(DELAY * 1000))
                        cv2.destroyAllWindows()

                        # display the images
                        # cv2.imshow('Scene + BBs', swapRB(np.flipud(boxedTrue)))
                        # cv2.imshow('Segmented + BBs', swapRB(np.flipud(boxedSeg)))
                        # cv2.waitKey(int(DELAY * 1000))

        pose = client.simGetVehiclePose()
        pp.pprint(pose)

        # print_pose(client)

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





    # import pprint
    # from decimal import *
    # def print_pose(client):
    #     # 1 vehicle pose
    #     pose = client.simGetVehiclePose()
    #     print("vehicle pose: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
    #     angles = airsim.to_eularian_angles(pose.orientation)
    #     print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
        # 2 multirotor pose
        # state = client.getMultirotorState()
        # # s = pprint.pformat(state)
        # # print("state: %s" % s)
        # print("multirotor pose: x={}, y={}, z={}".format(state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val))
        # angles = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        # print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
        # kinematics = client.simGetGroundTruthKinematics()
        # environment = client.simGetGroundTruthEnvironment()
        # print("Kinematics: %s\nEnvironemt %s" % (
        #     pprint.pformat(kinematics), pprint.pformat(environment)))
        # 3 imu pose
        # 4 camera pose
        # img_position = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])[0].camera_position
        # img_orientation = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])[0].camera_orientation
        # print("camera pose: x={}, y={}, z={}".format(img_position.x_val, img_position.y_val, img_position.z_val))
        # angles = airsim.to_eularian_angles(img_orientation)
        # print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))