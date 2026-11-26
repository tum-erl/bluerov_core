#!/usr/bin/env python3
"""
Synchronized Data Replayer / Simulator Node (replay_data_node)
Replays recorded Camera 1 video and MAVROS IMU measurements synchronously on a shared timeline,
simulating the live BlueROV2 hardware for offline testing of the full ROS 2 pipeline.

Simulates:
  - Camera 1: Streams H.264 RTP over UDP to 127.0.0.1:<cam1_port> (consumed by Camera1Node)
  - IMU:      Publishes sensor_msgs/Imu to <imu_topic> (consumed by ekf_node & imu_checker)
"""

import csv
import glob
import os
import sys
import threading
import time
import cv2

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu


def find_latest_recording(base_dir="recordings"):
    """Finds the most recent rec_* directory inside base_dir."""
    pattern = os.path.join(base_dir, "rec_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not dirs:
        return None
    dirs.sort()
    return dirs[-1]


class ReplayDataNode(Node):
    def __init__(self):
        super().__init__("replay_data_node")

        self.declare_parameter("data_dir", "")
        self.declare_parameter("cam1_port", 5600)
        self.declare_parameter("imu_topic", "/mavros/imu/data")
        self.declare_parameter("imu_frame_id", "base_link")
        self.declare_parameter("loop", False)
        self.declare_parameter("playback_rate", 1.0)

        data_dir = self.get_parameter("data_dir").value
        self.cam1_port = int(self.get_parameter("cam1_port").value)
        self.imu_topic = self.get_parameter("imu_topic").value
        self.imu_frame_id = self.get_parameter("imu_frame_id").value
        self.loop = bool(self.get_parameter("loop").value) or ("--loop" in sys.argv) or ("-loop" in sys.argv)
        self.playback_rate = max(0.1, float(self.get_parameter("playback_rate").value))

        if not data_dir:
            data_dir = find_latest_recording("recordings")
            if not data_dir:
                raise RuntimeError("No recording found in 'recordings/'. Provide -p data_dir:=<path>.")

        self.data_dir = data_dir
        self.video_path = os.path.join(data_dir, "video.mp4")
        self.frames_csv_path = os.path.join(data_dir, "frames.csv")
        self.imu_csv_path = os.path.join(data_dir, "imu.csv")

        for p in (self.video_path, self.frames_csv_path, self.imu_csv_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required recording file: {p}")

        # Load timestamps & IMU records into memory for jitter-free replay
        self.frame_timestamps = self._load_frame_timestamps(self.frames_csv_path)
        self.imu_records = self._load_imu_records(self.imu_csv_path)

        # Inspect video dimensions
        cap_probe = cv2.VideoCapture(self.video_path)
        if not cap_probe.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        self.width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()

        self.get_logger().info(
            f"Loaded recording '{self.data_dir}': "
            f"{len(self.frame_timestamps)} video frames ({self.width}x{self.height}), "
            f"{len(self.imu_records)} IMU samples."
        )

        # RELIABLE QoS serves both RELIABLE (ekf_node) and BEST_EFFORT (imu_checker) subscribers
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.imu_pub = self.create_publisher(Imu, self.imu_topic, imu_qos)

        # Initialize GStreamer H.264 UDP sender
        Gst.init(None)
        self.gst_pipe = None
        self.appsrc = None
        self._init_gst_sender()

        self.is_running = True
        self.cycle_lock = threading.Lock()
        self.t0_wall = time.monotonic()
        self.start_barrier = threading.Barrier(2)

        self.video_thread = threading.Thread(target=self._video_replay_loop, daemon=True)
        self.imu_thread = threading.Thread(target=self._imu_replay_loop, daemon=True)
        self.video_thread.start()
        self.imu_thread.start()

    @staticmethod
    def _load_frame_timestamps(csv_path):
        timestamps = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["t_rel_sec"]))
        return timestamps

    @staticmethod
    def _load_imu_records(csv_path):
        records = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append((
                    float(row["t_rel_sec"]),
                    float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]),
                    float(row["wx"]), float(row["wy"]), float(row["wz"]),
                    float(row["ax"]), float(row["ay"]), float(row["az"]),
                ))
        return records

    def _init_gst_sender(self):
        # Try x264enc first (gstreamer1.0-plugins-ugly), fallback to avenc_h264 (gstreamer1.0-libav)
        encoders = [
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30",
            "avenc_h264",
        ]
        for enc in encoders:
            cmd = (
                f"appsrc name=src is-live=true format=time "
                f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate=30/1 ! "
                f"videoconvert ! video/x-raw,format=I420 ! "
                f"{enc} ! rtph264pay config-interval=1 pt=96 ! "
                f"udpsink host=127.0.0.1 port={self.cam1_port} sync=false"
            )
            try:
                pipe = Gst.parse_launch(cmd)
                src = pipe.get_by_name("src")
                ret = pipe.set_state(Gst.State.PLAYING)
                if ret != Gst.StateChangeReturn.FAILURE and src is not None:
                    self.gst_pipe = pipe
                    self.appsrc = src
                    self.get_logger().info(
                        f"Streaming simulated Camera 1 H.264 UDP to 127.0.0.1:{self.cam1_port} ({enc.split()[0]})"
                    )
                    return
                pipe.set_state(Gst.State.NULL)
            except Exception:
                pass

        raise RuntimeError(
            "Could not initialize GStreamer H.264 encoder (tried x264enc, avenc_h264). "
            "Ensure gstreamer1.0-plugins-ugly or gstreamer1.0-libav is installed."
        )

    def _sync_cycle_start(self, is_leader):
        """Synchronizes video and IMU threads at the start of each playback cycle."""
        try:
            self.start_barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            return False

        if is_leader:
            with self.cycle_lock:
                self.t0_wall = time.monotonic()

        try:
            self.start_barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            return False
        return True

    def _wait_until(self, t_rel_sec, pts_offset_sec=0.0):
        target = self.t0_wall + (t_rel_sec / self.playback_rate)
        while self.is_running and rclpy.ok():
            dt = target - time.monotonic()
            if dt <= 0.0:
                return True
            time.sleep(min(dt, 0.01))
        return False

    def _video_replay_loop(self):
        pts_base_ns = 0
        while rclpy.ok() and self.is_running:
            if not self._sync_cycle_start(is_leader=True):
                break

            cap = cv2.VideoCapture(self.video_path)
            frame_idx = 0
            last_t_rel = 0.0

            while rclpy.ok() and self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx < len(self.frame_timestamps):
                    t_rel = self.frame_timestamps[frame_idx]
                else:
                    t_rel = last_t_rel + (1.0 / 30.0)
                last_t_rel = t_rel

                if not self._wait_until(t_rel):
                    break

                if self.appsrc is not None:
                    data = frame.tobytes()
                    buf = Gst.Buffer.new_wrapped(data)
                    buf.pts = pts_base_ns + int((t_rel / self.playback_rate) * Gst.SECOND)
                    buf.duration = int((1.0 / 30.0 / self.playback_rate) * Gst.SECOND)
                    self.appsrc.emit("push-buffer", buf)

                frame_idx += 1

            cap.release()
            pts_base_ns += int(((last_t_rel + 0.05) / self.playback_rate) * Gst.SECOND)

            if not self.loop:
                self.get_logger().info("Video replay finished.")
                break

    def _imu_replay_loop(self):
        while rclpy.ok() and self.is_running:
            if not self._sync_cycle_start(is_leader=False):
                break

            for rec in self.imu_records:
                if not rclpy.ok() or not self.is_running:
                    break
                t_rel, qx, qy, qz, qw, wx, wy, wz, ax, ay, az = rec

                if not self._wait_until(t_rel):
                    break

                msg = Imu()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.imu_frame_id
                msg.orientation.x = qx
                msg.orientation.y = qy
                msg.orientation.z = qz
                msg.orientation.w = qw
                msg.angular_velocity.x = wx
                msg.angular_velocity.y = wy
                msg.angular_velocity.z = wz
                msg.linear_acceleration.x = ax
                msg.linear_acceleration.y = ay
                msg.linear_acceleration.z = az
                self.imu_pub.publish(msg)

            if not self.loop:
                self.get_logger().info("IMU replay finished.")
                break

    def destroy_node(self):
        self.is_running = False
        try:
            self.start_barrier.abort()
        except Exception:
            pass
        if self.gst_pipe is not None:
            self.gst_pipe.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ReplayDataNode()
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
