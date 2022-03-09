#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper script to check the class -> color mapping in AirSim.

Usage:
    1. Set segmentation resolution to a low value in settings.json (10x10)
    2. Launch an Unreal scene and run this script
    3. Output `seg_colors.csv` will be written with a class -> RGB mapping
"""
from pydoc import cli
import airsim
import numpy as np
import csv
import os
import cv2

client = airsim.VehicleClient(timeout_value = 7200)
client.confirmConnection()

requests = airsim.ImageRequest("0", airsim.ImageType.Infrared, False, False)

object_list = ["prp_pylong_Sml14", "TemplateCube_Rounded_26", "ground", "prp_pylong_Sml11", "TemplateCube_Rounded_7"]
# cylinder_ok = client.simSetSegmentationObjectID("TemplateCube_Rounded[\w]*", 241, True)
# cylinder_ok=client.simGetSegmentationObjectID("prp_pylong_Sml14")
# print(cylinder_ok)
# cylinder_ok=client.simGetSegmentationObjectID("prp_pylong_Sml11")
# print(cylinder_ok)

# cylinder_ok=client.simGetSegmentationObjectID("prp_manhole5")
# print(cylinder_ok)
object_seg_ids = [client.simGetSegmentationObjectID(object_name) for object_name in object_list]
for object_name, object_seg_id in zip(object_list, object_seg_ids):
    print(f"object_name: {object_name}, object_seg_id: {object_seg_id}")

# object_list = sorted(client.simListSceneObjects())
# object_seg_ids = [client.simGetSegmentationObjectID(object_name) for object_name in object_list]
# for object_name, object_seg_id in zip(object_list, object_seg_ids):
#     print(f"object_name: {object_name}, object_seg_id: {object_seg_id}")

allList = client.simListSceneObjects('.*')
print("all size: ", len(allList))
i = 0
with open('ir_allobjs_qzc.txt','w') as f:    #设置文件对象
    for object_name in allList:
        i=i+1
        if i < 1000000:
            id = client.simGetSegmentationObjectID(object_name)
            f.writelines(str(object_name) + ' '+ str(id)+"\n")                 #将字符串写入文件中
        # else:
        #     f.writelines(obj+"\n")                 #将字符串写入文件中
print("all size: ", len(allList), " i ", i)

if 0:
    allList = client.simListSceneObjects('.*')
    print("all size: ", len(allList))
    with open('allobjs_qzc.txt','w') as f:    #设置文件对象
        for obj in allList:
            f.writelines(obj+"\n")                 #将字符串写入文件中

    # with open('all_objects.txt', 'w') as f:
    #     writer = csv.writer(f, delimiter=' ')
    #     for obj in allList:
    #         writer.writerow(obj)

    colors = {}
    file_path = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02-06/color_map/"
    for cls_id in range(256):
        # map every asset to cls_id and extract the single RGB value produced
        client.simSetSegmentationObjectID(".*", cls_id, is_name_regex=True)
        response = client.simGetImages([requests])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        cv2.imwrite(os.path.normpath(file_path + str(cls_id) + '.png'), img_rgb)


        color = tuple(np.unique(img_rgb.reshape(-1, img_rgb.shape[-1]), axis=0)[0])
        print(f"{cls_id}\t{color}")
        colors[cls_id] = color

    with open('qzc_seg_colors.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for k, v in colors.items():
            writer.writerow([k] + list(v))