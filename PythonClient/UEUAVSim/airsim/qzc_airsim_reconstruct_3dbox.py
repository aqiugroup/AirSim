import argparse
import json
import os
import re

import airsim
import numpy as np
import open3d as o3d
import pandas as pd
import PIL.Image
from tqdm import tqdm

from scipy.spatial.transform import Rotation

def DepthConversion(PointDepth, f):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]
    i_c = float(H) / 2 - 1
    j_c = float(W) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
    return PlaneDepth.astype(np.float32)

# Parse command line arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-r', '--run', help='folder name of the run')
group.add_argument('-l', '--last', action='store_true', help='use last run')
parser.add_argument('-s', '--step', default=1, type=int, help='frame step')
parser.add_argument('-t', '--depth_trunc', default=10000, type=float, help='max distance of depth projection')
parser.add_argument('-w', '--write_frames', action='store_true', help='save a point cloud for each frame')
parser.add_argument('--seg', action='store_true', help='use segmentation colors')
parser.add_argument('--vis', action='store_true', help='show visualization')
# args = parser.parse_args()
# debug
# args = parser.parse_args(['-r','', '-w', '--vis'])
args = parser.parse_args(['-r','', '--vis'])

# Get the default directory for AirSim
airsim_path = os.path.join(os.path.expanduser('~'), 'Documents', 'AirSim')

# Load the settings file
with open(os.path.join(airsim_path, 'settings.json'), 'r') as fp:
    data = json.load(fp)

# Get the camera intrinsics
capture_settings = data['CameraDefaults']['CaptureSettings'][0]
img_width = capture_settings['Width']
img_height = capture_settings['Height']
img_fov = capture_settings['FOV_Degrees']

# Compute the focal length
fov_rad = img_fov * np.pi/180
fd = (img_width/2.0) / np.tan(fov_rad/2.0)

# Create the camera intrinsic object
intrinsic = o3d.camera.PinholeCameraIntrinsic()
# intrinsic.set_intrinsics(img_width, img_height, fd, fd, img_width/2 - 0.5, img_height/2 - 0.5)
intrinsic.set_intrinsics(img_width, img_height, fd, fd, img_width/2, img_height/2)

# Get the run name
if args.last:
    runs = []
    for f in os.listdir(airsim_path):
        if re.fullmatch('\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', f):
            runs.append(f)
    run = sorted(runs)[-1]
else:
    run = args.run

# Load the recording metadata
data_path = os.path.join(airsim_path, run)
df = pd.read_csv(os.path.join(data_path, 'airsim_rec_small.txt'), delimiter='\t')

# Create the output directory if needed
if args.write_frames:
    os.makedirs(os.path.join(data_path, 'points'), exist_ok=True)

# Initialize an empty point cloud and camera list
pcd = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
cams = []

data_folder = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_small/"
data_folder = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_ir_box/"
# Loop over all the frames
for frame in tqdm(range(0, df.shape[0], args.step)):

    # === Create the transformation matrix ===

    x, y, z = df.iloc[frame][['POS_X', 'POS_Y', 'POS_Z']]
    T = np.eye(4)
    T[:3,3] = [-y, -z, -x]

    qw, qx, qy, qz = df.iloc[frame][['Q_W', 'Q_X', 'Q_Y', 'Q_Z']]
    R = np.eye(4)
    R[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((qw, qy, qz, qx)) # TODO:(qzc) (qw,qx,qy,qz) ?

    C = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

    F = R.T @ T @ C

    # a  = np.array([[1, 2], [3, 4]])  # 初始化一个非奇异矩阵(数组)
    # print(a)
    # print(np.linalg.inv(a))  # 对应于MATLAB中 inv() 函数
    # # 矩阵对象可以通过 .I 更方便的求逆,但是需要不是奇异矩阵
    # A = np.matrix(a)
    # print(A.I)
    # === Load the images ===

    # rgb_filename, seg_filename, depth_filename = df.iloc[frame].ImageFile.split(';')
    rgb_path = data_folder+"0/" + str(df.iloc[frame].TimeStamp)+"_0.png"
    depth_path = data_folder+"1/" + str(df.iloc[frame].TimeStamp)+"_1.pfm"
    seg_path = data_folder+"2/" + str(df.iloc[frame].TimeStamp)+"_2.png"
    box_path = data_folder+"3/" + str(df.iloc[frame].TimeStamp)+"_0.txt"


    # rgb_path = os.path.join(data_path, 'images', rgb_filename)
    rgb = PIL.Image.open(rgb_path).convert('RGB')

    # seg_path = os.path.join(data_path, 'images', seg_filename)
    seg = PIL.Image.open(seg_path).convert('RGB')

    # depth_path = os.path.join(data_path, 'images', depth_filename)
    depth, _ = airsim.utils.read_pfm(depth_path)
    depth = DepthConversion(depth, fd)

    # === Create the point cloud ===

    color = seg if args.seg else rgb
    color_image = o3d.geometry.Image(np.asarray(color))
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
    rgbd_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic=F)
    pcd += rgbd_pc

    # Save the point cloud for this frame
    if args.write_frames:
        pcd_name = f'points_seg_{frame:06d}' if args.seg else f'points_rgb_{frame:06d}'
        pcd_path = os.path.join(data_path, 'points', pcd_name + '.pcd')
        o3d.io.write_point_cloud(pcd_path, rgbd_pc)

        cam_path = os.path.join(data_path, 'points', f'cam_{frame:06d}.json')
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = F
        o3d.io.write_pinhole_camera_parameters(cam_path, cam)
    print("open3d size : ", len(rgbd_pc.points))

    # test my algorithm
    F=np.linalg.inv(F)
    cnt = 0
    for i in range(img_height):
        for j in range(img_width):
            d = depth[i, j]
            if d > 0 and d < 10000:
                z = d
                x = (j - img_width/2) * z / fd
                y = (i - img_height/2) * z / fd



                p = F @ [x, y, z, 1.0]
                rgbd_pc.points[cnt] = p[:3]
                cnt= cnt + 1

                # if j > 669.6342163085938 and j <  719.166015625 and i >  19.220108032226562  and i < 273.0154113769531 :
                # if j > 669.6342163085938 and j <  719.166015625 and i >  149  and i < 151 :
                #     print("p_in_c" + str(x)+" "+str(y)+" "+str(z)+" p_in_w "+ str(p[0])+" "+str(p[1])+" "+str(p[2]))
    print(cnt)

    # rgbd_pc.points.clear()
    # rgbd_pc.points.append(new_point)
    pcd2+=rgbd_pc
    if args.write_frames:
        pcd_name = f'points_seg_{frame:06d}' if args.seg else f'points_rgb_{frame:06d}'
        pcd_path = os.path.join(data_path, 'points', pcd_name + '_qzc.pcd')
        o3d.io.write_point_cloud(pcd_path, rgbd_pc)

    # === Save the camera position ===
    cams.append(o3d.geometry.LineSet.create_camera_visualization(intrinsic, F))

    #################### begin: get 3d box ###################
    df = pd.read_csv(box_path, sep='\s+')
    for frame in (range(0, df.shape[0], 1)):
        max_x, max_y, min_x, min_y, name, label, max_x_3d, max_y_3d, max_z_3d, min_x_3d, min_y_3d, min_z_3d, qw, qx, qy, qz, x, y, z \
        = df.iloc[frame][[
            'box2D.max.x_val', 'box2D.max.y_val',
            'box2D.min.x_val', 'box2D.min.y_val',
            'name',
            'label',
            'box3D.max.x_val', 'box3D.max.y_val', 'box3D.max.z_val',
            'box3D.min.x_val', 'box3D.min.y_val', 'box3D.min.z_val',
            'relative_pose.orientation.w_val', 'relative_pose.orientation.x_val', 'relative_pose.orientation.y_val', 'relative_pose.orientation.z_val',
            'relative_pose.position.x_val','relative_pose.position.y_val','relative_pose.position.z_val',
        ]]
        if frame > 30 and label < 3:
            oriented_bounding_box = rgbd_pc.get_oriented_bounding_box()
            oriented_bounding_box.color = (1, 0, 0)

            # !!!!!!!!!!! notice : relative_pose.position.z_val not correct !!!!!!!!!!!!!!!!!!!!!
            x =(max_x_3d+min_x_3d)/2
            y =(max_y_3d+min_y_3d)/2
            z =(max_z_3d+min_z_3d)/2
            # === Create the transformation matrix ===
            T1 = [x, y, z, 1]

            R1 = np.eye(4)
            # get_rotation_matrix_from_quaternion (w,x,y,z)
            R1[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((qw, qx, qy, qz))

            C1 = np.array([
                [ 1,  0,  0,  0],
                [ 0,  0, -1,  0],
                [ 0,  1,  0,  0],
                [ 0,  0,  0,  1]
            ])
            C2 = np.array([
                [ 0,  1,  0,  0],
                [ 1,  0,  0,  0],
                [ 0,  0,  -1,  0],
                [ 0,  0,  0,  1]
            ])
            C3=C1 @ C2 # cam(ned) --> cam(右x下y前z)
            F11 = F @ C3
            new_p = F11 @ T1
            F11 = F11  @ R1
            F12 = F11[:3, :3]
            R12 = Rotation.from_matrix(F12)
            new_quat = R12.as_quat()

            oriented_bounding_box.center = [new_p[0], new_p[1], new_p[2]]
            oriented_bounding_box.R  = o3d.geometry.get_rotation_matrix_from_quaternion((new_quat[3], new_quat[0], new_quat[1], new_quat[2]))
            oriented_bounding_box.extent=[max_x_3d-min_x_3d,  max_y_3d-min_y_3d, abs(max_z_3d-min_z_3d)]
            o3d.visualization.draw([rgbd_pc, oriented_bounding_box])
            print("display 3d box")
    #################### end: get 3d box ###################


# Save the point cloud
pcd_name = 'points_seg' if args.seg else 'points_rgb'
pcd_path = os.path.join(data_path, pcd_name + '.pcd')
o3d.io.write_point_cloud(pcd_path, pcd)

# Visualize
if args.vis:
    geos = [pcd2]
    geos.extend(cams)
    o3d.visualization.draw_geometries(geos)
