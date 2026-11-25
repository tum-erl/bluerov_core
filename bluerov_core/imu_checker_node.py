#!/usr/bin/env python3
"""
Simple test node to verify IMU data stream from MAVROS (/mavros/imu/data).
Prints orientation, angular velocity and linear acceleration.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

def quaternion_to_rpy(qx, qy, qz, qw):
    """Converts quaternion to Euler angles (roll, pitch, yaw) in degrees."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

class ImuCheckerNode(Node):
    def __init__(self):
        super().__init__('imu_checker_node')

        self.sub_imu = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            10
        )

        self.msg_count = 0
        self.last_print_time = self.get_clock().now()
        self.get_logger().info("IMU Checker Node started. Waiting for /mavros/imu/data...")

    def imu_callback(self, msg: Imu):
        self.msg_count += 1
        now = self.get_clock().now()

        # Τυπώνουμε 1 φορά το δευτερόλεπτο για να μην πλημμυρίζει το τερματικό
        if (now - self.last_print_time).nanoseconds >= 1e9:
            self.last_print_time = now

            q = msg.orientation
            roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
            av = msg.angular_velocity
            la = msg.linear_acceleration
            # Τυπωνουμε τις βασικες πληροφοριες: orientation, angular vel, linear acc
            print(
                f"[IMU #{self.msg_count}] "
                f"RPY: R={roll:+6.1f}°, P={pitch:+6.1f}°, Y={yaw:+6.1f}° | "
                f"Gyro (rad/s): [{av.x:+.2f}, {av.y:+.2f}, {av.z:+.2f}] | "
                f"Acc (m/s²): [{la.x:+.2f}, {la.y:+.2f}, {la.z:+.2f}]"
            )

def main(args=None):
    rclpy.init(args=args)
    node = ImuCheckerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
