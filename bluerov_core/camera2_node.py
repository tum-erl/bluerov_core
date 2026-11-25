#!/usr/bin/env python3
"""
Camera 2 Ingestion Node (Camera2Node)
Handles GStreamer UDP ingestion for Secondary Camera (port 5604) and places frames into shared RAM buffer.
"""

import threading
import rclpy
from rclpy.node import Node
from bluerov_core.camera1_node import GstVideoReceiver


class Camera2Node(Node):
    """Camera 2 capture node."""
    def __init__(self, shared_buffer):
        super().__init__("camera2_node")
        self.shared_buffer = shared_buffer

        self.declare_parameter("cam2_port", 5604)
        self.declare_parameter("enable_cam2", True)
        self.cam2_port = self.get_parameter("cam2_port").value
        self.enable_cam2 = bool(self.get_parameter("enable_cam2").value)

        self.is_running = True
        self.video = None

        if self.enable_cam2:
            self.get_logger().info(f"Connecting Secondary Camera on port {self.cam2_port}...")
            self.video = GstVideoReceiver(port=self.cam2_port, name="cam2")
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            self.get_logger().info("Camera 2 ingestion thread started.")
        else:
            self.get_logger().info("Camera 2 is disabled via parameter.")

    def _capture_loop(self):
        while rclpy.ok() and self.is_running:
            if not self.video:
                break
            frame, recv_time = self.video.get_latest_frame_blocking(timeout=0.2)
            if frame is not None:
                self.shared_buffer.put(frame, recv_time)

    def destroy_node(self):
        self.is_running = False
        if hasattr(self, "video") and self.video:
            self.video.stop()
        super().destroy_node()
