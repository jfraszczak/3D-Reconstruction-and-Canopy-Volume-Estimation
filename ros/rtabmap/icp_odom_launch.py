from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    parameters=[{
          'publish_tf': False
    }]

    remappings=[
          ('gps/fix', '/fix'),
          ('imu', '/mavros/imu/data_raw'),
          ('/scan_cloud', '/os_cloud_node/points'),
          ('scan', 'no'),
          ('odom', '/odom_icp')
    ]


    return LaunchDescription([

        #ICP odometry
        Node(
            package='rtabmap_odom', executable='icp_odometry', name="icp_odometry", output="screen",
            parameters=parameters,
            remappings=remappings,
            arguments=['--Odom/ResetCountdown 1 publish_null_when_lost False'])
    ])