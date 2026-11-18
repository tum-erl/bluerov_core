#!/usr/bin/env python3
"""
Laptop Webcam Publisher ROS 2 Node
Reads local webcam feed (OpenCV V4L2) and publishes sensor_msgs/Image with Real-Time QoS
"""

import os
import threading
import time
import numpy as np

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class LaptopVideo:
    """Laptop webcam capture class — OpenCV V4L2 capture with dedicated reader thread."""
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.latest_frame_data = None  # (bgr_frame, arrival_time)
        self.running = True
        self.condition = threading.Condition()

        # Χρήση CAP_V4L2 για απευθείας πρόσβαση στην κάμερα σε Ubuntu
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)

        # MJPG format για υψηλά FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            recv_time = time.time()
            if ret and frame is not None:
                with self.condition:
                    self.latest_frame_data = (frame, recv_time)
                    self.condition.notify()
            else:
                time.sleep(0.01)

    def get_latest_frame_blocking(self):
        with self.condition:
            while self.latest_frame_data is None and self.running:
                self.condition.wait(timeout=0.1)
            if not self.running:
                return None, None
            frame, recv_time = self.latest_frame_data
            self.latest_frame_data = None
            return frame, recv_time

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()


class LaptopCameraPublisher(Node):
    def __init__(self):
        super().__init__("laptop_camera_publisher_node")
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        
        self.camera_id = self.get_parameter("camera_id").get_parameter_value().integer_value
        self.camera_frame_id = self.get_parameter("camera_frame_id").get_parameter_value().string_value
        
        self.cv_bridge = CvBridge()
        
        # Real-time QoS Profile (Best Effort, depth=1)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.image_pub = self.create_publisher(Image, "rov/camera/image_raw", sensor_qos)
        
        self.get_logger().info(f"Opening laptop webcam (device ID: {self.camera_id})...")
        self.video = LaptopVideo(camera_id=self.camera_id)
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.worker_thread.start()

    def publish_loop(self):
        first_frame = True
        while rclpy.ok() and self.is_running:
            frame, _ = self.video.get_latest_frame_blocking()
            if frame is None:
                continue
            
            if first_frame:
                h, w = frame.shape[:2]
                self.get_logger().info(f"Laptop webcam connected ({w}x{h})! Publishing started.")
                first_frame = False

            try:
                msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.camera_frame_id
                
                self.image_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing image: {e}")

    def destroy_node(self):
        self.is_running = False
        self.video.stop()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaptopCameraPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()