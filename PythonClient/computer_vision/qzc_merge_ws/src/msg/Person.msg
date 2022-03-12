# 0 means invalid, 1 means target person, >=2 means other person
uint64 id
vision_msgs/BoundingBox2D bbox
float32 conf
uint32 type
float32 is_occluded
float32 is_cropped
# ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#  "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
Keypoint[] keypoints
vision_msgs/BoundingBox2D left_hand_bbox
vision_msgs/BoundingBox2D right_hand_bbox

# UNKNOWN = 0,
# STAND,
# WALKING,
# SITTING,
# RIGHT_ARM_UP,     // command follow/stop
# FOREARM_OPEN,
# FOREARM_CROSS,
int32 pose

# UNKNOWN = 0,
# OK,
# ONE,
# YEAH,
# SPIDERMAN,
# FIST,
# FIVE,
# THREE,
# ROCK,
int32 left_hand_gesture
int32 right_hand_gesture

# UNKNOWN = 0,
# MOVE_FORWARD,
# MOVE_BACK,
# FOLLOW,
# STOP,
# COME_HEAR,
# TURN_AROUND,
# GET_DOWN,
# STAND_UP,
int32 command

# person toward direction
float32 body_heading
float32 face_heading

# compare init follow person feature and current feature, check whether this person is follow target
float32 check_master_score

# 2048 dim normalized vector
float32[] reid_feature
float32[] master_reid_feature