from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    parameters=[{
          'odom_topic': 'odometry/global',
          'subscribe_scan_cloud':True,
          #'subscribe_scan':True,
          'subscribe_rgbd':False,
          'approx_sync':True,
          'gps_topic': '/fix',
          'use_sim_time': True,
          'visual_odometry': False,
          'publish_tf_map': False,
          'publish_tf_odom': False,
          'pos_tracking_enabled': False
    }]

    remappings=[
          ('rgb/image', '/d435i/color/image_raw'),
          ('rgb/camera_info', '/d435i/aligned_depth_to_color/camera_info'),
          ('depth/image', '/d435i/aligned_depth_to_color/image_raw'),
          ('scan', '/scan'),
          ('/scan_cloud', '/os_cloud_node/points'),
          ('gps/fix', '/fix'),
          ('odom', 'odometry/global')
    ]

    return LaunchDescription([

        # Nodes to launch       
        Node(
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=parameters,
            remappings=remappings,
            arguments=['-d --Optimizer/PriorsIgnored false --RGBD/NeighborLinkRefining true --RGBD/ProximityBySpace true --RGBD/OptimizeFromGraphEnd false --Rtabmap/DetectionRate 1']),
            
        Node(
            package='rtabmap_util', executable='point_cloud_xyzrgb', output='screen',
            parameters=[{
                "voxel_size": 0.05
            }],
            remappings=remappings),

        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            parameters=parameters,
            remappings=remappings
        )
    ])
