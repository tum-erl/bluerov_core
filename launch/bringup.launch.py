import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Βρίσκουμε αυτόματα το μονοπάτι του package σου
    pkg_share = get_package_share_directory('bluerov_core')
    urdf_path = os.path.join(pkg_share, 'urdf', 'bluerov2.urdf')

    # Διαβάζουμε το αρχεία URDF ως string
    with open(urdf_path, 'r') as f:
        robot_description_content = f.read()

    return LaunchDescription([
        # Ο κόμβος που δημοσιεύει το URDF στο topic /robot_description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}]
        )
    ])
