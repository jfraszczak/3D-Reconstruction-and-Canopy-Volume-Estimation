
# Run static transform publisher
ros2 run reconstruction static_transform_publisher

# Play bag file
ros2 bag play /media/jakub/Extreme\ SSD/2022-07-28-13-05-05_filare_4 --clock --topics /d435i/color/image_raw /d435i/aligned_depth_to_color/image_raw /d435i/aligned_depth_to_color/camera_info /mavros/imu/data_raw /fix /d435i/imu /t265/odom/sample /os_cloud_node/points

# Run rgbd odometry node
ros2 launch rgbd_odom_launch.py

# Run icp odometry node
ros2 launch icp_odom_launch.py

# Run localization node
ros2 launch localization_launch.py

# Run rtabmap
ros2 launch rtabmap_launch.py

# Export rtabmap reconstruction
rtabmap-export --poses_camera --images_id --output_dir "/home/jakub/Desktop/rgbd" "/home/jakub/.ros/rtabmap.db"
rtabmap-export --poses_camera --poses_scan --scan --images_id --output_dir "/home/jakub/Desktop/scan" "/home/jakub/.ros/rtabmap.db"