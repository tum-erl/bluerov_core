from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    cam1_port_arg = DeclareLaunchArgument('cam1_port', default_value='5600', description='Primary camera port/device')
    cam2_port_arg = DeclareLaunchArgument('cam2_port', default_value='5604', description='Secondary camera port/device')
    enable_cam2_arg = DeclareLaunchArgument('enable_cam2', default_value='true', description='Enable secondary camera')

    return LaunchDescription([
        cam1_port_arg,
        cam2_port_arg,
        enable_cam2_arg,

        # BlueROV2 Modular Multi-Node System Container (Zero-copy in-RAM pipeline)
        # Includes: Camera1, Camera2, Localization (AprilTags PnP), Annotated Publisher
        Node(
            package='bluerov_core',
            executable='bluerov_system',
            output='screen',
            parameters=[{
                'camera1_node': {
                    'cam1_port': LaunchConfiguration('cam1_port'),
                },
                'camera2_node': {
                    'cam2_port': LaunchConfiguration('cam2_port'),
                    'enable_cam2': LaunchConfiguration('enable_cam2'),
                },
                'annotated_publisher_node': {
                    'enable_cam2': LaunchConfiguration('enable_cam2'),
                },
            }]
        )
    ])
