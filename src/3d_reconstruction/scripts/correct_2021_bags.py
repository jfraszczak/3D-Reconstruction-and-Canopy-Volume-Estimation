from rosbags.rosbag2 import Reader, Writer
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2, Imu

def correct_2021_bags(bag_path: str, bag_d435i_path: str, bag_output: str) -> None:

    connections = dict()

    with Writer(bag_output) as writer:

        with Reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                try:
                    conn = writer.add_connection(connection.topic, connection.msgtype, 'cdr', '')
                    connections[connection.topic] = conn
                except:
                    pass
    
                if connection.topic == '/os_cloud_node/points':
                    msg = deserialize_message(rawdata, PointCloud2)
                elif connection.topic == '/os_cloud_node/imu':
                    msg = deserialize_message(rawdata, Imu)
                else:
                    writer.write(connections[connection.topic], timestamp, rawdata)
                    continue
 
                msg.header.stamp.sec = int(timestamp / 10 ** 9)
                msg.header.stamp.nanosec = timestamp - int(timestamp / 10 ** 9) * 10 ** 9
                writer.write(connections[connection.topic], timestamp, serialize_message(msg))

        with Reader(bag_d435i_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic[:6] == '/d435i':
                    try:
                        conn = writer.add_connection(connection.topic, connection.msgtype, 'cdr', '')
                        connections[connection.topic] = conn
                    except:
                        pass
                    writer.write(connections[connection.topic], timestamp, rawdata)

                    
#correct_2021_bags('/media/jakub/Extreme SSD/2021/2021-07-29-13-41-16_quarta', '/media/jakub/Extreme SSD/2021/2021-07-29-13-41-10_quarta', '/media/jakub/Extreme SSD/2021/row_4')
#correct_2021_bags('/media/jakub/Extreme SSD/2021/2021-07-29-13-23-27_terza', '/media/jakub/Extreme SSD/2021/2021-07-29-13-23-20_terza', '/media/jakub/Extreme SSD/2021/row_3')
#correct_2021_bags('/media/jakub/Extreme SSD/2021/2021-07-29-13-05-31_seconda_corretta', '/media/jakub/Extreme SSD/2021/2021-07-29-13-05-25_seconda_corretta', '/media/jakub/Extreme SSD/2021/row_2')
correct_2021_bags('/media/jakub/Extreme SSD/2021/2021-07-29-12-15-14_prima', '/media/jakub/Extreme SSD/2021/2021-07-29-12-15-11_prima', '/media/jakub/Extreme SSD/2021/row_1')