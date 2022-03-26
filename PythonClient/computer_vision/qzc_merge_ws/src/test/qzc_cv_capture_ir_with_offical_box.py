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
    # "prp_trafficLight_Blueprint[\w]*",
    "prp_streetLight_Blueprint[\w]*",
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
    "BP_LightStudio[\w]*",


    "prp_chainlinkFence[\w]*",
    "prp_tarp[\w]*",
    "prp_vent[\w]*",
    "door_A[\w]*",
    "prp_ac[\w]*",
    "StoreSign[\w]*",
    "prp_pylong_Sml[\w]*",
    "flg_grass[\w]*",
    "prp_parkingMeter[\w]*",
    # "Flg_hedge_Short[\w]*",
    "Flg_hedge[\w]*",
    "prp_metalFence[\w]*",
    "prp_barrier_Lrg[\w]*",
    "TrashBag[\w]*",
    "prp_potSquare[\w]*",
    "prp_cementBarrier[\w]*",
    "flg_fern[\w]*",
    "prp_flag[\w]*",
    "Sign_1_mdl[\w]*"
]

# for detection, not support [\w]* ??
REGEX_OBJECTS_OF_DETECTION_INTEREST = [
    "Landscape*",
    # "prp_trafficLight_Blueprint*",
    "prp_streetLight_Blueprint*",
    "prp_manhole*",
    "prp_light*",
    "prp_garbageCan*",
    "prp_bench*",
    "prp_fireHydrant*",
    "prp_chair*",
    "prp_table*",
    "CameraActor*",
    "prp_sewerGrate*",
    "prp_tableUmbrella*",
    "prp_electricalBox*",
    "Awning_mdl*",
    "PowerLine*",
    "prp_crosswalkSign*",
    "Building*",
    "prp_metalPillar*",
    "flg_tree*",
    # "BP_LightStudio[\w]*",  # TODO:qzc, we close it, it should be "BP_LightStudio*"


    "prp_chainlinkFence*",
    "prp_tarp*",
    "prp_vent*",
    "door_A*",
    "prp_ac*",
    "StoreSign*",
    "prp_pylong_Sml*",
    "flg_grass*",
    "prp_parkingMeter*",
    # "Flg_hedge_Short*",
    "Flg_hedge*",  # TODO:qzc use this is correct, but slow
    "prp_metalFence*",
    "prp_barrier_Lrg*",
    "TrashBag*",
    "prp_potSquare*",
    "prp_cementBarrier*",
    "flg_fern*",
    "prp_flag*",
    "Sign_1_mdl*"
]

# Classes corresponding to the objects in the REGEX_OBJECTS_OF_INTEREST list above.
OBJECT_OF_INTEREST_CLASSES = [
    "ground",
    # "trafficLight",
    "streetLight",
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
    "sky",


    "chainlinkFence",
    "tarp",
    "vent",
    "door",
    "ac",
    "StoreSign",
    "pylong_Sml",
    "grass",
    "parkingMeter",
    # "hedge_Short",
    "hedge",
    "metalFence",
    "barrier_Lrg",
    "TrashBag",
    "potSquare",
    "cementBarrier",
    "fern",
    "flag",
    "sign2"
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

# https://tool.lu/color/
CLASS_COLORS_OUTDOOR = [
    (135, 169, 180), # 浅灰色
    (112, 105, 191), # 深紫色
    (89, 121, 72), # 深绿色
    (190, 225, 64), # 深绿色-2
    (206, 190, 59), # 深黄色-2
    (81, 13, 36), # 深粉色
    (115, 176, 195), # 深蓝色-2
    (161, 171, 27), #深绿色-3
    (153, 108, 6), # 深米黄色
    (29, 26, 199), # 深蓝色
    (102, 16, 239), # 深紫色
    (242, 107, 146), # 老红色
    (156, 198, 23), # 浅绿色
    (49, 89, 160), #浅蓝色+2
    (68, 218, 116), # 深绿色-3
    (196, 30, 8), # 红色
    (121, 67, 28), # 棕色
    (0, 53, 65), # 蓝绿色
    (11, 236, 9), # 绿色
    (54, 72, 205), # 蓝色
    (146, 52, 70), # 红色-3
    (226, 149, 143), # 红色-2
    (151, 126, 171), # 浅紫色-2
    (194, 39, 7), # 红色2
    (205, 120, 161), # 红色2-2
    (212, 51, 60), # 红色3
    (103, 252, 157), # 绿色2
    (211, 80, 208), # 粉色
    (195, 237, 132), # 绿色+3
    (189, 135, 188), # 粉色+3
    (124, 21, 123), # 紫色2
    (19, 132, 69), # 绿色3
    (94, 253, 175), # 绿色-3
    (90, 162, 242), # 蓝色-1
    (182, 251, 87), # 绿色+2
    (199, 29, 1), # 红色3
    (254, 12, 229) # 粉色-4
]

# deal with swapped RB values
# cp = MESH_COLS.copy()
# MESH_COLS[:,0] = cp[:,2]
# MESH_COLS[:,2] = cp[:,0]

id_color_dict={}
for key, val in zip(OBJECT_OF_INTEREST_CLASSES, range(len(CLASS_COLORS_OUTDOOR))):
    id_color_dict[val] = CLASS_COLORS_OUTDOOR[val]
id_color_dict[255]=(0,0,0)

map_label_to_color = np.vectorize(
    lambda x: id_color_dict.get(x, id_color_dict[255])
)


import sys
def status(length, percent, index=0, remain_time=0):
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



client = None

# set camera name and image type to request images and detections
detect_camera_name = "0"
detect_image_type = airsim.ImageType.Scene
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
        found = client.simSetSegmentationObjectID(key, val, True)
        print(' ' * (40 - len(msg)) + ('[SUCCESS]' if found else '[FAILED!]'))
        if key == "flg_tree[\w]*":
            key_new = "InstancedFoliageActor[\w]*"
            found = client.simSetSegmentationObjectID(key_new, val, True)
            msg_new = 'Setting %s to ID %d...' % (key_new, val)
            print(msg_new, end='')
            print(' ' * (40 - len(msg_new)) + ('[SUCCESS]' if found else '[FAILED!]'))
        if key == "prp_pylong_Sml[\w]*":
            key_new = "prp_pylon_Lrg[\w]*"
            found = client.simSetSegmentationObjectID(key_new, val, True)
            msg_new = 'Setting %s to ID %d...' % (key_new, val)
            print(msg_new, end='')
            print(' ' * (40 - len(msg_new)) + ('[SUCCESS]' if found else '[FAILED!]'))
        if key == "Landscape[\w]*":
            key_new = "Intersection[\w]*"
            found = client.simSetSegmentationObjectID(key_new, val, True)
            msg_new = 'Setting %s to ID %d...' % (key_new, val)
            print(msg_new, end='')
            print(' ' * (40 - len(msg_new)) + ('[SUCCESS]' if found else '[FAILED!]'))

            key_new = "GroundArea[\w]*"
            found = client.simSetSegmentationObjectID(key_new, val, True)
            msg_new = 'Setting %s to ID %d...' % (key_new, val)
            print(msg_new, end='')
            print(' ' * (40 - len(msg_new)) + ('[SUCCESS]' if found else '[FAILED!]'))

    # set 2d/3d detection
    # set detection radius in [cm]
    client.simSetDetectionFilterRadius(detect_camera_name, detect_image_type, 30 * 100) # 30m
    # add desired object name to detect in wild card/regex format
    for key, val in zip(REGEX_OBJECTS_OF_DETECTION_INTEREST, range(len(REGEX_OBJECTS_OF_DETECTION_INTEREST))):
        msg = 'Detection: add mesh name--> %s ..' % (key)
        print(msg, end='\n')
        client.simAddDetectionFilterMeshName(detect_camera_name, detect_image_type, key)

    open_extral_object = True
    if open_extral_object:
        key='InstancedFoliageActor*'
        msg = 'Detection: add mesh name--> %s ..' % (key)
        print(msg, end='\n')
        client.simAddDetectionFilterMeshName(detect_camera_name, detect_image_type, key)
    key='prp_pylon_Lrg*'
    msg = 'Detection: add mesh name--> %s ..' % (key)
    print(msg, end='\n')
    client.simAddDetectionFilterMeshName(detect_camera_name, detect_image_type, key)
    # client.simAddDetectionFilterMeshName(detect_camera_name, detect_image_type, "prp_cementBarrier*")


if __name__ == '__main__':
    file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_rec.txt" # sys.argv[1]
    # file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/airsim_rec_continue.txt" # sys.argv[1]
    # file_path = "/Users/aqiu/Documents/AirSim/airsim_rec_small1.txt" # sys.argv[1]
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
        for n in range(3):
            os.makedirs(os.path.join(tmp_dir, str(n)))
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    # for semantic rgb
    semantic_bgr_path = os.path.join(tmp_dir, str(20))
    if os.path.exists(semantic_bgr_path) == False:
        os.makedirs(semantic_bgr_path)
    # for semantic bbox
    semantic_bb_path = os.path.join(tmp_dir, str(21))
    if os.path.exists(semantic_bb_path) == False:
        os.makedirs(semantic_bb_path)

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
    # ---------------------------- for debug end --------------------------------

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

        print_log = False
        if print_log:
            print(">>>>>>>>>>simGetVehiclePose")
            print("vehicle pose: xyz={}, {},{}, qwqxqyqz={}, {},{},{}".format(pos_x, pos_y, pos_z, quat_w, quat_x,quat_y,quat_z))
            pose = client.simGetVehiclePose()
            # print("vehicle pose: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
            p =pp.pprint(pose)

            camera_info = client.simGetCameraInfo(str(0))
            # print("vehicle pose: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
            p = pp.pprint(camera_info)

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
            airsim.ImageRequest("0", airsim.ImageType.Infrared,False, True),
            # airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            # airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
            # airsim.ImageRequest("0", airsim.ImageType.Infrared,False, False),
            # airsim.ImageRequest("0", airsim.ImageType.Segmentation,False, False)
        ])

        for i, response in enumerate(responses):
            if response.pixels_as_float:
                # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir,  str(i), str(timestamp) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
            else:
                # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(timestamp) + "_" + str(i) + '.png')), response.image_data_uint8)

                # 方式一：根据label在图像上，进行box计算
                if  i == 100:
                    im = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                    im = im.reshape(response.height, response.width, 3)           # reshape array to 3channel image
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

                        # display the images
                        # cv2.imshow('Scene + BBs', swapRB(np.flipud(boxedTrue)))
                        # cv2.imshow('Segmented + BBs', swapRB(np.flipud(boxedSeg)))
                        # cv2.waitKey(int(DELAY * 1000))

                        cv2.imshow('infrared:2, seg:3, i:'+str(i), boxedSeg)
                        cv2.waitKey(int(DELAY * 1000))
                        cv2.destroyAllWindows()

                # 方式二：直接调用airsim官方的detection接口
                if i == 0: # airsim.ImageType.Scene
                    rawImage = client.simGetImage(detect_camera_name, detect_image_type)
                    # png = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                    # png = png.reshape(response.height, response.width, 3)           # reshape array to 4 channel image
                    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                    all_of_interests= client.simGetDetections(detect_camera_name, detect_image_type)

                    object_infos =[]
                    if all_of_interests:
                        for interest in all_of_interests:
                            # s = pprint.pformat(interest)
                            # print(" object: %s" % s)
                            label = client.simGetSegmentationObjectID(interest.name)
                            pose_in_w = client.simGetObjectPose(interest.name)

                            object_info="{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}"\
                            .format(interest.box2D.max.x_val, interest.box2D.max.y_val, interest.box2D.min.x_val, interest.box2D.min.y_val,
                            interest.box3D.max.x_val, interest.box3D.max.y_val, interest.box3D.max.z_val,
                            interest.box3D.min.x_val, interest.box3D.min.y_val, interest.box3D.min.z_val,
                            interest.geo_point.altitude, interest.geo_point.latitude, interest.geo_point.longitude,
                            interest.name,
                            interest.relative_pose.orientation.w_val, interest.relative_pose.orientation.x_val, interest.relative_pose.orientation.y_val, interest.relative_pose.orientation.z_val,
                            interest.relative_pose.position.x_val, interest.relative_pose.position.y_val, interest.relative_pose.position.z_val,
                            pose_in_w.position.x_val, pose_in_w.position.y_val, pose_in_w.position.z_val,
                            pose_in_w.orientation.w_val, pose_in_w.orientation.x_val, pose_in_w.orientation.y_val, pose_in_w.orientation.z_val,  label)
                            object_infos.append(object_info)


                            cv2.rectangle(png,(int(interest.box2D.min.x_val),int(interest.box2D.min.y_val)),(int(interest.box2D.max.x_val),int(interest.box2D.max.y_val)),(255,0,0),2)
                            cv2.putText(png, interest.name, (int(interest.box2D.min.x_val),int(interest.box2D.min.y_val + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))


                    bbs_file = os.path.normpath(os.path.join(semantic_bb_path, str(timestamp) + "_" + str(i) + '.txt'))
                    header = "box2D.max.x_val box2D.max.y_val box2D.min.x_val box2D.min.y_val \
                            box3D.max.x_val box3D.max.y_val box3D.max.z_val \
                            box3D.min.x_val box3D.min.y_val box3D.min.z_val \
                            geo_point.altitude geo_point.latitude geo_point.longitude \
                            name \
                            relative_pose.orientation.w_val relative_pose.orientation.x_val relative_pose.orientation.y_val relative_pose.orientation.z_val \
                            relative_pose.position.x_val relative_pose.position.y_val relative_pose.position.z_val \
                            pose_in_w.position.x_val pose_in_w.position.y_val pose_in_w.position.z_val\
                            pose_in_w.orientation.w_val pose_in_w.orientation.x_val pose_in_w.orientation.y_val pose_in_w.orientation.z_val label"
                    with open(bbs_file,'w') as f:    #设置文件对象
                            f.writelines(header+"\n")
                            for object_info in object_infos:
                                f.writelines(object_info+"\n")                 #将字符串写入文件中

                    save_box_image = True
                    if save_box_image:
                        # airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(3), str(timestamp) + "_" + str(i) + '.png')), png)
                        cv2.imwrite(os.path.normpath(os.path.join(semantic_bb_path, str(timestamp) + "_" + str(i) + '.png')), png)
                    else:
                        cv2.imshow("bbox", png)
                        cv2.waitKey(int(DELAY * 300))
                        cv2.destroyAllWindows()

                if i == 2:
                    im = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                    # im = im.reshape(response.height, response.width, 3)           # reshape array to 4 channel image
                    im = cv2.imdecode(airsim.string_to_uint8_array(im), cv2.IMREAD_UNCHANGED)
                    semantic_bgr = np.asarray(map_label_to_color(im[:, :, 0]))
                    semantic_bgr = np.stack((semantic_bgr[0], semantic_bgr[1], semantic_bgr[2]), axis=2)
                    semantic_bgr = semantic_bgr.astype(np.uint8)
                    semantic_bgr = cv2.cvtColor(semantic_bgr , cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.normpath(os.path.join(semantic_bgr_path, str(timestamp) + "_" + str(2) + '.png')),  semantic_bgr)

        if print_log:
            pose = client.simGetVehiclePose()
            pp.pprint(pose)

        index = index + 1
        percent = (index - start_index) / duration
        if percent - last_percent > interval:
            last_percent = percent

            now_time = time.time()
            remain_time = float(now_time - start_time) * (duration-index) / (index * 3600)
            status(40, percent, index, remain_time)

        # print_pose(client)
        # time.sleep(1)

        line = fin.readline().strip()

    status(40, 1, duration, 0)

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