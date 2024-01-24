import math
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import os
import yaml
import json

def quaternion_from_euler(ai: float, aj: float, ak: float) -> np.ndarray:
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


class StaticTransformPublisher(Node):

    def __init__(self):
        super().__init__('static_transform_publisher')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_transforms()

    
    def publish_transforms(self) -> None:

        path, _ = os.path.split(os.path.realpath(__file__))
        with open(os.path.join(path, 'tf_static_manual.json'), 'r') as file:
            tf_static_manual = json.load(file)

        with open(os.path.join(path, 'tf_static_d435i.yaml'), 'r') as file:
            tf_static_d435i = yaml.safe_load(file)

        transforms = []
        for transform in tf_static_manual:
            t = self.make_transform(transform['frame_id'], transform['child_frame_id'], transform['transform'], tranform_to_quaternion=True)
            transforms.append(t)
        
        for transform in tf_static_d435i['transforms']:
            t = self.make_zero_transform(transform['header']['frame_id'], transform['child_frame_id'])
            transforms.append(t)
            
        self.tf_static_broadcaster.sendTransform(transforms)

    def make_transform(self, frame_id: str, child_frame_id: str, transformation: dict, tranform_to_quaternion: bool=False) -> TransformStamped:
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        t.transform.translation.x = float(transformation['translation']['x'])
        t.transform.translation.y = float(transformation['translation']['y'])
        t.transform.translation.z = float(transformation['translation']['z'])

        if tranform_to_quaternion:
            quat = quaternion_from_euler(float(transformation['rotation']['roll']), float(transformation['rotation']['pitch']), float(transformation['rotation']['yaw']))
        else:
            quat = [float(transformation['rotation']['x']), float(transformation['rotation']['y']), float(transformation['rotation']['z']), float(transformation['rotation']['w'])]

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        return t
    
    def make_zero_transform(self, frame_id: str, child_frame_id: str) -> TransformStamped:
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        return t

    

def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticTransformPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
