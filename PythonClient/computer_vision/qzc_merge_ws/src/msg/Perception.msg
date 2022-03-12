Header header

# 0: main, 1: left, 2: right, 3: rear, 4: head
uint32 camera_id

# person recognition
Person target_person
Person[] persons

# movable space detection
# point attribute, 0: step, 1: door, 2: stair, 3: other
Keypoint[] movable_space

# 3D object detection
Object_3D[] obj_3d

# line keypoint detection
LKD line_keypoints

# door window detection
DWD[] door_windows