from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    parameters=[{
          'rgb_topic': '/d435i/color/image_raw',
          'depth_topic': '/d435i/aligned_depth_to_color/image_raw',
          'camera_info_topic': '/d435i/aligned_depth_to_color/camera_info',
          'publish_tf': False,
          'publish_null_when_lost': False
    }]

    remappings=[
          ('rgb/image', '/d435i/color/image_raw'),
          ('rgb/camera_info', '/d435i/aligned_depth_to_color/camera_info'),
          ('depth/image', '/d435i/aligned_depth_to_color/image_raw'),
          ('gps/fix', '/fix'),
          ('odom', 'odom_rgbd')
    ]


    return LaunchDescription([

        #RGB-D odometry
        Node(
            package='rtabmap_odom', executable='rgbd_odometry', output="screen",
            parameters=parameters,
            remappings=remappings,
            arguments=['--Odom/ResetCountdown 1 --GFTT/MinDistance 10 --Odom/Strategy 1 --Odom/FilteringStrategy 1'])
    ])