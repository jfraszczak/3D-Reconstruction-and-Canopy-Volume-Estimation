import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.rosbag2 import Reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
import os
import shutil
from ..gps_reconstruction import sensor_transformation, get_timestamp



def read_poses(dir_path: str) -> dict[int, np.ndarray]:
    file_path = os.path.join(dir_path, 'odometries.txt')
    poses = {}
    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split()

            timestamp = int(components[12])
            pose = np.array(components[0:12], dtype=float)

            pose = pose.reshape(3, 4)
            pose = np.vstack([pose, [0, 0, 0, 1]])
            poses[timestamp] = pose

    return poses

def read_point_cloud(dir_path: str) -> o3d.geometry.PointCloud:
    file_path = os.path.join(dir_path, 'custom_map.pcd')
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def get_camera_poses(poses: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    camera_poses = dict()
    for timestamp in poses.keys():
        camera_poses[timestamp] = poses[timestamp] @ sensor_transformation('d435i_link') @ np.linalg.inv(sensor_transformation('os_sensor'))
    return camera_poses

def get_rgb_images(bag_path: str, poses: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        images_timestamps = []
        images_collection = dict()

        with Reader(bag_path) as reader:
            for connection, _, rawdata in reader.messages():
                if connection.topic == '/d435i/color/image_raw':
                    msg = deserialize_message(rawdata, Image)
                    timestamp = get_timestamp(msg) * 10 ** 9

                    if not timestamp in images_collection:
                        images_collection[timestamp] = 1
                    else:
                        images_timestamps.append(timestamp)

                if connection.topic == '/d435i/aligned_depth_to_color/image_raw':
                    msg = deserialize_message(rawdata, Image)
                    timestamp = get_timestamp(msg) * 10 ** 9
                    
                    if not timestamp in images_collection:
                        images_collection[timestamp] = 1
                    else:
                        images_timestamps.append(timestamp)

        corresponding_timestamps = dict()
        for timestamp in poses.keys():
            nearest_timestamp = min(images_timestamps, key=lambda t: abs(t - timestamp))
            corresponding_timestamps[nearest_timestamp] = timestamp

        rgb_images = dict()
        depth_images = dict()
        with Reader(bag_path) as reader:
            for connection, _, rawdata in reader.messages():
                if connection.topic == '/d435i/color/image_raw':
                    msg = deserialize_message(rawdata, Image)
                    timestamp = get_timestamp(msg) * 10 ** 9

                    if timestamp in corresponding_timestamps:
                        bgr_img = CvBridge().imgmsg_to_cv2(img_msg=msg, desired_encoding="bgr8")
                        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                        rgb_img = o3d.geometry.Image(rgb_img)
                        rgb_images[corresponding_timestamps[timestamp]] = rgb_img

                if connection.topic == '/d435i/aligned_depth_to_color/image_raw':
                    msg = deserialize_message(rawdata, Image)
                    timestamp = get_timestamp(msg) * 10 ** 9

                    if timestamp in corresponding_timestamps:
                        depth_img = CvBridge().imgmsg_to_cv2(img_msg=msg, desired_encoding="passthrough")
                        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_img).astype(np.uint16))
                        depth_images[corresponding_timestamps[timestamp]] = depth_img

        return rgb_images, depth_images        

def get_calib(bag_path: str) -> dict:
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        projection_matrix = None
        img_width = 0
        img_height = 0

        with Reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/d435i/aligned_depth_to_color/camera_info':

                    if projection_matrix is None:
                        msg = deserialize_message(rawdata, CameraInfo)
                        fx = msg.k[0]
                        fy = msg.k[4]
                        cx = msg.k[2]
                        cy = msg.k[5]

                        intrinsic.set_intrinsics(msg.width, msg.height, fx, fy, cx, cy)
                        projection_matrix = np.reshape(np.array(msg.p), (3, 4))
                        img_width = msg.width
                        img_height = msg.height
        calib = {
            'camera_name': -1,
            'image_width': img_width,
            'image_height': img_height,
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': intrinsic.intrinsic_matrix.flatten().tolist()
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': projection_matrix.flatten().tolist()
            }
        }

        return calib

def transformation_matrix2line(t: np.ndarray, timestamp: int, id: int) -> list:
        x = t[0, 3]
        y = t[1, 3]
        z = t[2, 3]
        quat = Rotation.from_matrix(t[:3, :3]).as_quat()
        qx, qy, qz, qw = quat
        line = [timestamp, x, y, z, qx, qy, qz, qw, id]
        line = [str(val) for val in line]

        return line

def update_output_directory(dir_path: str, bag_path: str) -> None:
    poses = read_poses(dir_path)
    pcd = read_point_cloud(dir_path)
    o3d.visualization.draw_geometries([pcd])
    camera_poses = get_camera_poses(poses)
    rgb_images, depth_images = get_rgb_images(bag_path, poses)
    calib = get_calib(bag_path)

    if os.path.exists(os.path.join(dir_path, 'rgb')):
        shutil.rmtree(os.path.join(dir_path, 'rgb'))
    os.makedirs(os.path.join(dir_path, 'rgb'))

    if os.path.exists(os.path.join(dir_path, 'depth')):
        shutil.rmtree(os.path.join(dir_path, 'depth'))
    os.makedirs(os.path.join(dir_path, 'depth'))

    if os.path.exists(os.path.join(dir_path, 'calib')):
        shutil.rmtree(os.path.join(dir_path, 'calib'))
    os.makedirs(os.path.join(dir_path, 'calib'))

    file_sensor_poses = open(os.path.join(dir_path, "sensor_poses.txt"), "w")
    file_camera_poses = open(os.path.join(dir_path, "camera_poses.txt"), "w")
    file_sensor_poses.write('#timestamp x y z qx qy qz qw id\n')
    file_camera_poses.write('#timestamp x y z qx qy qz qw id\n')

    id = 1
    for timestamp in rgb_images.keys():
        line = transformation_matrix2line(camera_poses[timestamp], timestamp, id)
        file_camera_poses.write(' '.join(line) + '\n')

        line = transformation_matrix2line(poses[timestamp], timestamp, id)
        file_sensor_poses.write(' '.join(line) + '\n')

        calib['camera_name'] = id
        with open(os.path.join(dir_path, 'calib', '{}.yaml'.format(id)), 'w') as outfile:
            yaml.safe_dump(calib, outfile, default_flow_style=False, sort_keys=False)

        rgb_image = rgb_images[timestamp]
        depth_image = depth_images[timestamp]
        o3d.io.write_image(os.path.join(dir_path, 'rgb', "{}.jpg".format(id)), rgb_image)
        o3d.io.write_image(os.path.join(dir_path, 'depth', "{}.png".format(id)), depth_image)

        id += 1

    file_sensor_poses.close()
    file_camera_poses.close()
                                
if __name__ == '__main__':
    dir_path = '/media/jakub/Extreme SSD/ART-SLAM/outputs/row_1_09-01_11-36'
    bag_path = '/media/jakub/Extreme SSD/2022-07-28-11-29-56_filare_1_lungo'
    update_output_directory(dir_path, bag_path)
