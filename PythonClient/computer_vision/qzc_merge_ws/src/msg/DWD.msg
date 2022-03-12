uint32 id
# 0: elevator, 1: vertical_door, 2: horizontal_door, 3: electrical_horizontal_door, 4: window
uint32 category
vision_msgs/BoundingBox2D bbox
# 8 keypoints, keypoint attribute: 0: occluded, 1: visible
Keypoint[] keypoints

float32 conf
float32 is_occluded
float32 is_cropped
# 0: open, 1: half-open, 2: close, 3: unknown
uint32  open_state