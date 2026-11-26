#!/usr/bin/env python3
"""
Synchronized Data Recorder Node (record_data_node)
Records Camera 1 GStreamer UDP stream (port 5600) and MAVROS IMU topic (/mavros/imu/data)
simultaneously using a common monotonic clock, saving synchronized datasets for offline replay.

Output directory structure:
  <output_dir>/rec_<YYYYMMDD_HHMMSS>/
    ├── video.mp4      (Camera 1 video stream)
    ├── frames.csv     (frame_idx, t_rel_sec)
    └── imu.csv        (t_rel_sec, qx, qy, qz, qw, wx, wy, wz, ax, ay, az)
"""

import csv
import os
import threading
import time
from datetime import datetime
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu

from bluerov_core.camera1_node import GstVideoReceiver


class RecordDataNode(Node):
    def __init__(self):
        super().__init__("record_data_node")

        self.declare_parameter("cam1_port", 5600)
        self.declare_parameter("imu_topic", "/mavros/imu/data")
        self.declare_parameter("output_dir", "recordings")

        self.cam1_port = self.get_parameter("cam1_port").value
        self.imu_topic = self.get_parameter("imu_topic").value
        base_out_dir = self.get_parameter("output_dir").value

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_out_dir, f"rec_{timestamp_str}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.video_path = os.path.join(self.session_dir, "video.mp4")
        self.frames_csv_path = os.path.join(self.session_dir, "frames.csv")
        self.imu_csv_path = os.path.join(self.session_dir, "imu.csv")

        # Common monotonic time origin for exact synchronization
        self.t0 = time.monotonic()
        self.is_running = True

        # Counters & file handles
        self.frame_count = 0
        self.imu_count = 0
        self.video_writer = None
        self.imu_lock = threading.Lock()

        self.frames_file = open(self.frames_csv_path, "w", newline="")
        self.frames_writer = csv.writer(self.frames_file)
        self.frames_writer.writerow(["frame_idx", "t_rel_sec"])

        self.imu_file = open(self.imu_csv_path, "w", newline="")
        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            "t_rel_sec",
            "qx", "qy", "qz", "qw",
            "wx", "wy", "wz",
            "ax", "ay", "az"
        ])

        # Subscribe to IMU (BEST_EFFORT matches MAVROS sensor QoS, fallback compatible)
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self._imu_callback, imu_qos)

        # Start Camera 1 GStreamer receiver
        self.get_logger().info(f"Recording session directory: {self.session_dir}")
        self.get_logger().info(f"Listening for Camera 1 on port {self.cam1_port} and IMU on {self.imu_topic}...")
        self.video = GstVideoReceiver(port=self.cam1_port, name="rec_cam1")

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

        # 1 Hz status logger
        self.status_timer = self.create_timer(1.0, self._print_status)

    def _imu_callback(self, msg: Imu):
        if not self.is_running:
            return
        t_rel = time.monotonic() - self.t0
        q = msg.orientation
        w = msg.angular_velocity
        a = msg.linear_acceleration

        with self.imu_lock:
            if not self.imu_file.closed:
                self.imu_writer.writerow([
                    f"{t_rel:.6f}",
                    f"{q.x:.8f}", f"{q.y:.8f}", f"{q.z:.8f}", f"{q.w:.8f}",
                    f"{w.x:.8f}", f"{w.y:.8f}", f"{w.z:.8f}",
                    f"{a.x:.8f}", f"{a.y:.8f}", f"{a.z:.8f}",
                ])
                self.imu_count += 1

    def _video_loop(self):
        while rclpy.ok() and self.is_running:
            frame, _ = self.video.get_latest_frame_blocking(timeout=0.2)
            if frame is None:
                continue

            t_rel = time.monotonic() - self.t0

            if self.video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, (w, h))
                self.get_logger().info(f"Video stream detected ({w}x{h}). Recording started!")

            self.video_writer.write(frame)
            self.frames_writer.writerow([self.frame_count, f"{t_rel:.6f}"])
            self.frame_count += 1

    def _print_status(self):
        elapsed = time.monotonic() - self.t0
        print(f"[REC {elapsed:6.1f}s] Frames: {self.frame_count:5d} | IMU msgs: {self.imu_count:6d}", flush=True)

    def destroy_node(self):
        self.is_running = False
        if hasattr(self, "video") and self.video:
            self.video.stop()
        if hasattr(self, "video_thread") and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if self.video_writer is not None:
            self.video_writer.release()

        with self.imu_lock:
            if not self.imu_file.closed:
                self.imu_file.flush()
                self.imu_file.close()
        if not self.frames_file.closed:
            self.frames_file.flush()
            self.frames_file.close()

        self.get_logger().info(
            f"Recording saved to {self.session_dir} "
            f"({self.frame_count} frames, {self.imu_count} IMU samples)."
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecordDataNode()
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
