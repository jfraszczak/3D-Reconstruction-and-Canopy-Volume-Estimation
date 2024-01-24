from rosbags.rosbag2 import Reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

def extract_images_from_bag(bag_path: str, images_topic: str, output_dir: str, time_start: float=0.0, time_end: float=-1.0, every_k: int=1) -> None:
    '''
    Extracts images from a specified topic of a bag file and saves in separate output directory.

    Args:
        bag_path (str): Path to the bag file from which images are to be extracted.
        images_topic (str): Name of a topic in which images are stored.
        output_dir (str): Path of output direcory into which extracted images are to be saved.
        time_start (float): Value specifying image with the earliest time stamp which should be taken into account.
        time_end (float): Value specifying image with the latest time stamp which should be taken into account.
        every_k (int): Value specifying that only every k-th image will be saved.
    '''
    i = 0
    with Reader(bag_path) as reader:
        for connection, _, rawdata in reader.messages():
            if connection.topic == images_topic:
                
                msg = deserialize_message(rawdata, Image)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 10 ** 9

                # Disregard measurements outside of given time range
                if time_end == -1.0:
                    if time_start > timestamp:
                        continue
                else:
                    if not (time_start <= timestamp <= time_end):
                        continue
                
                i += 1
                if i % every_k != 1:
                    continue

                bgr_img = CvBridge().imgmsg_to_cv2(img_msg=msg, desired_encoding="bgr8")
                bgr_img = cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                file_name = str(msg.header.stamp.sec) + '_' + str(msg.header.stamp.nanosec) + '.png'
                output_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_path, bgr_img)

if __name__ == "__main__":
    extract_images_from_bag('/media/jakub/Seagate Expansion Drive/2022-07-28-13-05-05_filare_4', 
                            '/media/jakub/Seagate Expansion Drive/canopy_images',
                            '/d435i/color/image_raw',
                            1659006306 + 50,
                            1659006306 + 250,
                            every_k=10)
