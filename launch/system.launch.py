import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = get_package_share_directory('bluerov_core')

    # Launch arguments
    cam1_port_arg = DeclareLaunchArgument('cam1_port', default_value='5600', description='Primary camera port/device')
    cam2_port_arg = DeclareLaunchArgument('cam2_port', default_value='5604', description='Secondary camera port/device')
    enable_cam2_arg = DeclareLaunchArgument('enable_cam2', default_value='true', description='Enable secondary camera')

    # 1. Core Vision & Localization Launch
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'localization.launch.py')),
        launch_arguments={
            'cam1_port': LaunchConfiguration('cam1_port'),
            'cam2_port': LaunchConfiguration('cam2_port'),
            'enable_cam2': LaunchConfiguration('enable_cam2'),
        }.items()
    )

    # 2. Visualization & Foxglove Bridge Launch
    visualization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'visualization.launch.py'))
    )

    return LaunchDescription([
        cam1_port_arg,
        cam2_port_arg,
        enable_cam2_arg,
        localization_launch,
        visualization_launch
    ])
