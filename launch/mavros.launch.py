import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Προεπιλεγμένο MAVLink endpoint για BlueROV2 / BlueOS:
    # Το BlueOS στέλνει MAVLink μέσω UDP στο port 14550
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value='udp://:14550@',
        description='MAVLink URL to connect to the BlueROV autopilot (e.g. udp://:14550@)'
    )

    gcs_url_arg = DeclareLaunchArgument(
        'gcs_url',
        default_value='',
        description='Optional GCS bridge URL (e.g. udp://@localhost:14555)'
    )

    tgt_system_arg = DeclareLaunchArgument(
        'tgt_system',
        default_value='1',
        description='Target system ID of the vehicle (standard ArduSub ID is 1)'
    )

    tgt_component_arg = DeclareLaunchArgument(
        'tgt_component',
        default_value='1',
        description='Target component ID of the vehicle'
    )

    log_output_arg = DeclareLaunchArgument(
        'log_output',
        default_value='screen',
        description='Log output destination'
    )

    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        name='mavros',
        namespace='mavros',
        output=LaunchConfiguration('log_output'),
        parameters=[{
            'fcu_url': LaunchConfiguration('fcu_url'),
            'gcs_url': LaunchConfiguration('gcs_url'),
            'target_system_id': LaunchConfiguration('tgt_system'),
            'target_component_id': LaunchConfiguration('tgt_component'),
            'fcu_protocol': 'v2.0',
            # Ρυθμίσεις αναφοράς συστημάτων συντεταγμένων
            'conn': {
                'heartbeat_rate': 1.0,
                'timeout': 10.0,
            },
        }]
    )

    return LaunchDescription([
        fcu_url_arg,
        gcs_url_arg,
        tgt_system_arg,
        tgt_component_arg,
        log_output_arg,
        mavros_node
    ])
