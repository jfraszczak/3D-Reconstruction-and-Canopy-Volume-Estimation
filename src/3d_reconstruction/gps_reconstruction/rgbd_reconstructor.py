from rosbags.rosbag2 import Reader
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import open3d as o3d
from cv_bridge import CvBridge
import cv2
import yaml
import os
from .reconstructor import Reconstructor
from .point_cloud_preprocessor import remove_outliers, remove_points_below_ground
from .utils_reconstruction import sensor_transformation, get_timestamp

with open('src/3d_reconstruction/gps_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


class RGBDReconstructor(Reconstructor):

    def __init__(self, output_dir: str, test_mode: bool=False):
        """
        Initializes all the necessary attributes for the reconstructor object and reads data from the Bag file.

        Args:
            test_mode (bool): Indication whether to visualize intermediate computations or not.
        """
        super().__init__(output_dir=output_dir, test_mode=test_mode)
        self.sensor = config['tf_static']['camera_link']
    
    def _collect_point_clouds(self, time_start: float = 0, time_end: float = -1, every_k: int = 1) -> None:
        """
        Computes consecutive point clouds based on the depth maps, intrinsic matrix of the RGBD camera and color images.

        Args:
            time_start (float): Value specifying point cloud with the earliest time stamp which should be taken into account.
            time_start (float): Value specifying point cloud with the latest time stamp which should be taken into account.
            every_k (int): Value specifying that only every k-th point cloud will be considered.
        """
        os.makedirs(os.path.join(self.output_dir, 'rgb'))
        os.makedirs(os.path.join(self.output_dir, 'depth'))
        os.makedirs(os.path.join(self.output_dir, 'calib'))
        
        self.clouds = []
        i = -1
        id = 1
        with Reader(config['bag']['path']) as reader:
            image_pairs = dict()
            for connection, _, rawdata in reader.messages():
                image_found = False

                if connection.topic == config['bag']['topics']['color_img']:
                    msg = deserialize_message(rawdata, Image)
                    timestamp = get_timestamp(msg)
                    if timestamp not in image_pairs:
                        image_pairs[timestamp] = dict()
                    image_pairs[timestamp]['color'] = deserialize_message(rawdata, Image)
                    image_found = True

                if connection.topic == config['bag']['topics']['depth_img']:
                    msg = deserialize_message(rawdata, Image)
                    timestamp = get_timestamp(msg)
                    if timestamp not in image_pairs:
                        image_pairs[timestamp] = dict()
                    image_pairs[timestamp]['depth'] = deserialize_message(rawdata, Image)
                    image_found = True
                
                if image_found:
                    if 'color' in image_pairs[timestamp] and 'depth' in image_pairs[timestamp]:
                    
                        # Compute point cloud only when both depth and color image are collected
                        msg_depth = image_pairs[timestamp]["depth"]
                        msg_color = image_pairs[timestamp]["color"]
                        image_pairs.pop(timestamp)

                        # Disregard measurements outside of given time range
                        if time_end == -1.0:
                            if time_start > timestamp:
                                continue
                        else:
                            if not (time_start <= timestamp <= time_end):
                                continue
                        
                        # Take into account only every k-th measurement
                        i += 1
                        if i % every_k != 0:
                            continue
                        
                        # Reformat images
                        depth_img = CvBridge().imgmsg_to_cv2(img_msg=msg_depth, desired_encoding="passthrough")
                        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_img).astype(np.uint16))

                        bgr_img = CvBridge().imgmsg_to_cv2(img_msg=msg_color, desired_encoding="bgr8")
                        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                        rgb_img = o3d.geometry.Image(rgb_img)

                        # Create point cloud
                        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
                        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, self.intrinsic)

                        # Preprocess point cloud
                        cloud = cloud.voxel_down_sample(voxel_size=0.05)

                        self.clouds.append((timestamp, cloud))
                        o3d.io.write_image(os.path.join(self.output_dir, 'rgb', "{}.jpg".format(id)), rgb_img)
                        o3d.io.write_image(os.path.join(self.output_dir, 'depth', "{}.png".format(id)), depth_img)

                        calib = {
                            'camera_name': str(id),
                            'image_width': self.img_width,
                            'image_height': self.img_height,
                            'camera_matrix': {
                                'rows': 3,
                                'cols': 3,
                                'data': self.intrinsic.intrinsic_matrix.flatten().tolist()
                            },
                            'projection_matrix': {
                                'rows': 3,
                                'cols': 4,
                                'data': self.camera_projection_matrix.flatten().tolist()
                            }
                        }

                        with open(os.path.join(self.output_dir, 'calib', '{}.yaml'.format(id)), 'w') as outfile:
                            yaml.safe_dump(calib, outfile, default_flow_style=False, sort_keys=False)

                        id += 1

        timestamps = [timestamp for timestamp, _ in self.clouds]
        sorted_timestamps = sorted(timestamps)
        assert sorted_timestamps == timestamps
