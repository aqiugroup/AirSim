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
df = pd.read_csv(os.path.join(data_path, 'airsim_rec_small1.txt'), delimiter='\t')

# Create the output directory if needed
if args.write_frames:
    os.makedirs(os.path.join(data_path, 'points'), exist_ok=True)

# Initialize an empty point cloud and camera list
pcd = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
cams = []

data_folder = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_small/"
data_folder = "/Users/aqiu/Documents/AirSim/2022-03-07-02-02/airsim_drone_ir_box/"
data_folder = "/Users/aqiu/Documents/AirSim/airsim_drone/"
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
    # C = np.array([
    #     [ 1,  0,  0,  0],
    #     [ 0,  1, 0,  0],
    #     [ 0,  0,  1,  0],
    #     [ 0,  0,  0,  1]
    #     ])

    Tcw = R.T @ T # c was in cam(右x下y前z)
    Tcw1 = R.T @ T @ C

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
    rgbd_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic=Tcw1)
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
    Tw1c=np.linalg.inv(Tcw1)
    cnt = 0
    for i in range(img_height):
        for j in range(img_width):
            d = depth[i, j]
            if d > 0 and d < 10000:
                z = d
                x = (j - img_width/2) * z / fd
                y = (i - img_height/2) * z / fd



                p = Tw1c @ [x, y, z, 1.0]
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
    cams.append(o3d.geometry.LineSet.create_camera_visualization(intrinsic, Tcw1))

    #################### begin: get 3d box ###################
    df1 = pd.read_csv(box_path, sep='\s+')
    oriented_bounding_boxs=[rgbd_pc]
    for frame1 in (range(0, df1.shape[0], 1)):
        max_x, max_y, min_x, min_y, name, label, max_x_3d, max_y_3d, max_z_3d, min_x_3d, min_y_3d, min_z_3d, qw, qx, qy, qz, x, y, z, \
        o_x, o_y, o_z, o_qw, o_qx, o_qy, o_qz\
        = df1.iloc[frame1][[
            'box2D.max.x_val', 'box2D.max.y_val',
            'box2D.min.x_val', 'box2D.min.y_val',
            'name',
            'label',
            'box3D.max.x_val', 'box3D.max.y_val', 'box3D.max.z_val',
            'box3D.min.x_val', 'box3D.min.y_val', 'box3D.min.z_val',
            'relative_pose.orientation.w_val', 'relative_pose.orientation.x_val', 'relative_pose.orientation.y_val', 'relative_pose.orientation.z_val',
            'relative_pose.position.x_val','relative_pose.position.y_val','relative_pose.position.z_val',
            'pose_in_w.position.x_val', 'pose_in_w.position.y_val', 'pose_in_w.position.z_val',
            'pose_in_w.orientation.w_val', 'pose_in_w.orientation.x_val', 'pose_in_w.orientation.y_val', 'pose_in_w.orientation.z_val'
        ]]
        if label < 5:
        # if frame1 > 150 and label < 5:
            oriented_bounding_box = rgbd_pc.get_oriented_bounding_box()
            oriented_bounding_box.color = (1, 0, 0)

            ########################## box的中心和朝向 #######################################
            ########################## 方式一：用simGetDetections获取的3d box结果 #######################################
            # !!!!!!!!!!! notice : relative_pose.position.z_val not correct !!!!!!!!!!!!!!!!!!!!!
            # x =(max_x_3d+min_x_3d)/2
            # y =(max_y_3d+min_y_3d)/2
            # z =(max_z_3d+min_z_3d)/2
            # === Create the transformation matrix ===
            T1 = [x, y, z, 1]

            R1 = np.eye(4)
            # get_rotation_matrix_from_quaternion (w,x,y,z)
            R1[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((qw, qx, qy, qz))
            if label == 1:
                print("object: min={}, {},{}, max={},{},{}".format(min_x_3d, min_y_3d, min_z_3d, max_x_3d, max_y_3d, max_z_3d))
                print("object: xyz={}, {},{}, qwqxqyqz={}, {},{},{}".format(x, y, z, qw, qx, qy, qz))

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
            F11 = Tw1c @ C3
            new_p = F11 @ T1
            F11 = F11  @ R1
            F12 = F11[:3, :3]
            R12 = Rotation.from_matrix(F12)
            new_quat = R12.as_quat()

            # 将box位姿转到相机下 t_co
            t_co = C3 @T1
            r_co = C3 @ R1
            r_co1 = Rotation.from_matrix(r_co[:3,:3])
            q_co = r_co1.as_quat()
            if label == 1:
                print("object pose: xyz={}, {},{}, qwqxqyqz={}, {},{},{}".format(t_co[0], t_co[1], t_co[2], q_co[3], q_co[0], q_co[1], q_co[2]))

            # T2 = [o_x, o_y, o_z, 1]
            # R2 = np.eye(4)
            # R2[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((o_qw, o_qx, o_qy, o_qz))
            # new_p2 = F @ Tcw @ T2  @ C3
            # F21 =  F @ Tcw @ R2  @ C3
            # F22 = F21[:3, :3]
            # R22 = Rotation.from_matrix(F22)
            # new_quat2 = R22.as_quat()

            # T2 = np.eye(4)
            # T2[:3,3] = [-o_y, -o_z, -o_x]
            # R2 = np.eye(4)
            # R2[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((o_qw, o_qy, o_qz, o_qx))
            # O_F = R2.T @ T2 @ C
            # O_F=np.linalg.inv(O_F)
            # new_p2 = O_F[:3,3]
            # F22 = O_F[:3, :3]
            # R22 = Rotation.from_matrix(F22)
            # new_quat2 = R22.as_quat()

            ########################## 方式二：用 simGetObjectPose 获取的 box 结果 #######################################
            # 1 先转到世界坐标系下
            T2 = [o_x, o_y, o_z, 1]
            R2 = np.eye(4)
            R2[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((o_qw, o_qx, o_qy, o_qz))
            t_w1o1 = C.T @ C3 @ T2
            F21 = C.T@ C3 @ R2
            F22 = F21[:3, :3]
            R22 = Rotation.from_matrix(F22)
            q_w1o1 = R22.as_quat()
            new_quat = q_w1o1

            # 2 进行z方向的补偿（沿着box朝向）
            # 沿着box的z方向补偿一半的高度
            sign = 1
            if max_z_3d-min_z_3d < 0:
                sign = -1
            T_WO = np.eye(4)
            T_WO[:,3] =t_w1o1
            r_wo2 = F21
            r_wo2[:3, 3] = [0,  0,  sign * (max_z_3d-min_z_3d)/2]
            T_WO = T_WO @ r_wo2
            # 沿着box的z方向补偿一半的高度
            t_w1o2 = T_WO[:, 3]
            new_p = t_w1o2



            # 以下为将box位姿转到相机下 t_co
            # Tco = Tcw1 * Tw1o
            t_co = Tcw1 @ t_w1o2
            r_co = Tcw1 @ F21
            r_co1 = Rotation.from_matrix(r_co[:3,:3])
            q_co = r_co1.as_quat()
            if label == 1:
                print("object pose: xyz={}, {},{}, qwqxqyqz={}, {},{},{}".format(t_co[0], t_co[1], t_co[2], q_co[3], q_co[0], q_co[1], q_co[2]))

            # 补偿z方向的距离
            # sign = 1
            # if max_z_3d-min_z_3d < 0:
            #     sign = -1
            # T_CO = np.eye(4)
            # T_CO[:,3] =t_co
            # r_co2 = r_co
            # r_co2[:3, 3] = [0,   (max_z_3d-min_z_3d)/2, 0]
            # T_CO = T_CO @ r_co2
            # t_co = T_CO[:, 3]

            # 验证转换后的 t_co
            F31 =Tw1c @ r_co
            F32 = F31[:3, :3]
            R33 = Rotation.from_matrix(F32)
            new_quat = R33.as_quat()


            t_wo =  Tw1c @ t_co
            # 沿着box的z方向补偿一半的高度(前面补偿了，这里就不需要了)
            # sign = 1
            # if max_z_3d-min_z_3d < 0:
            #     sign = -1
            # T_CO = np.eye(4)
            # T_CO[:,3] =t_wo
            # r_co2 = F31
            # r_co2[:3, 3] = [0,  0,  sign * (max_z_3d-min_z_3d)/2]
            # T_CO = T_CO @ r_co2
            # t_wo = T_CO[:, 3]
            # 沿着box的z方向补偿一半的高度
            new_p = t_wo[:3]

            # # Tco = inv(Tw1c) * Tw1o
            # t_co = Tw1c @ C.T @ C3 @T2
            # r_co = Tw1c @ C.T @ C3 @ R2
            # r_co1 = Rotation.from_matrix(r_co[:3,:3])
            # q_co = r_co1.as_quat()
            # if label == 1:
            #     print("object pose: xyz={}, {},{}, qwqxqyqz={}, {},{},{}".format(t_co[0], t_co[1], t_co[2], q_co[3], q_co[0], q_co[1], q_co[2]))
            # t_o =  np.linalg.inv(Tw1c) @ t_co
            # new_p = t_o[:3]
            # F31 = np.linalg.inv(Tw1c)  @ r_co
            # F32 = F31[:3, :3]
            # R33 = Rotation.from_matrix(F32)
            # new_quat = R33.as_quat()
            # new_p =  [o_x, o_y, o_z]
            # new_quat = [o_qx, o_qy, o_qz, o_qw]
            if label == 1:
                print("vehicle pose: xyz={}, {},{}, qwqxqyqz={}, {},{},{}".format(new_p[0], new_p[1], new_p[2], new_quat[3], new_quat[0], new_quat[1], new_quat[2]))
                # print("vehicle pose: xyz2={}, {},{}, qwqxqyqz2={}, {},{},{}".format(new_p2[0], new_p2[1], new_p2[2], new_quat2[3], new_quat2[0], new_quat2[1], new_quat2[2]))

            # new_p = new_p2
            # new_quat=new_quat2

            oriented_bounding_box.center = [new_p[0], new_p[1], new_p[2]]
            oriented_bounding_box.R  = o3d.geometry.get_rotation_matrix_from_quaternion((new_quat[3], new_quat[0], new_quat[1], new_quat[2]))
            oriented_bounding_box.extent=[abs(max_x_3d-min_x_3d),  abs(max_y_3d-min_y_3d), abs(max_z_3d-min_z_3d)]

            oriented_bounding_boxs.append(oriented_bounding_box)
    # o3d.visualization.draw([rgbd_pc, oriented_bounding_box])
    o3d.visualization.draw(oriented_bounding_boxs)
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
