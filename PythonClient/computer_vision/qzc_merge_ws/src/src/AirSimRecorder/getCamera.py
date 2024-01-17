import airsim
import os
import time
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()

# Set plane IDs
client.simSetSegmentationObjectID("backwall[\w]*", 40, True)
client.simSetSegmentationObjectID("frontwall[\w]*", 50, True)
client.simSetSegmentationObjectID("leftwall[\w]*", 60, True)
client.simSetSegmentationObjectID("rightwall[\w]*", 70, True)
client.simSetSegmentationObjectID("ground[\w]*", 80, True)
client.simSetSegmentationObjectID("ceiling[\w]*", 90, True)

# Ignore non-planes
client.simSetSegmentationObjectID("barrel[\w]*", 0, True)
client.simSetSegmentationObjectID("brick[\w]*", 0, True)
client.simSetSegmentationObjectID("tubelight[\w]*", 0, True)
client.simSetSegmentationObjectID("water_container[\w]*", 0, True)
client.simSetSegmentationObjectID("ladder[\w]*", 0, True)
client.simSetSegmentationObjectID("broken_pillar[\w]*", 0, True)
client.simSetSegmentationObjectID("door_piece[\w]*", 0, True)
client.simSetSegmentationObjectID("walk_in_circle[\w]*", 0, True)

time.sleep(1)

if not os.path.isdir("./captures"):
    os.system('mkdir captures')

time_file = open("./captures/timestamps.txt", 'w')

i = 0
while(client.isApiControlEnabled()):
    responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene), airsim.ImageRequest("front_center", airsim.ImageType.Segmentation)])
    timestamp = client.getMultirotorState().timestamp
    response_rgb = responses[0]
    response_seg = responses[1]

    if (response_rgb.height != 0 and response_rgb.width != 0 and response_seg.height != 0 and response_seg.width != 0):
        airsim.write_file(os.path.join('./captures/', (str(i) + '_rgb.png')), response_rgb.image_data_uint8)
        airsim.write_file(os.path.join('./captures/', (str(i) + '_seg.png')), response_seg.image_data_uint8)
        time_file.write(str(timestamp) + "\t" + str(i) + "\n")
        i = i+1

time_file.close()
print("camera reader disconnected...")
