import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('bluerov_core')
    urdf_path = os.path.join(pkg_share, 'urdf', 'bluerov2.urdf')

    with open(urdf_path, 'r') as f:
        robot_description_content = f.read()

    return LaunchDescription([
        # 1. Robot State Publisher (URDF visualization)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}]
        ),

        # 2. Foxglove Bridge for Web/App Telemetry & Visualization
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            parameters=[{
                'port': 8765,
                'send_buffer_limit': 1000000,
                'use_compression': False,
            }]
        )
    ])
