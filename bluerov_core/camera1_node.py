#!/usr/bin/env python3
"""
Camera 1 Ingestion Node (Camera1Node)
Handles GStreamer UDP ingestion for Primary Camera (port 5600) and places frames into shared RAM buffer.
"""

import sys
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GstVideoReceiver:
    """Robust zero-latency GStreamer UDP receiver for H.264 BlueROV streams."""
    def __init__(self, port=5600, name="cam"):
        Gst.init(None)
        self.port = port
        self.name = name
        self.latest_frame_data = None
        self.running = True
        self.condition = threading.Condition()
        self.cap = None

        # Check if local webcam is requested (e.g. port 0 or /dev/video0)
        self.is_v4l2 = str(port).startswith("/dev/video") or (str(port).isdigit() and int(port) < 100)
        if self.is_v4l2:
            device_idx = int(port) if str(port).isdigit() else port
            self.cap = cv2.VideoCapture(device_idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.thread = threading.Thread(target=self._v4l2_loop, daemon=True)
            self.thread.start()
        else:
            self.sink_name = f"sink_{self.name}_{self.port}"
            self.video_source = (
                f"udpsrc port={self.port} buffer-size=65536 "
                f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
                f"rtpjitterbuffer latency=50 drop-on-latency=true"
            )
            self.video_codec = "! rtph264depay ! h264parse ! avdec_h264 max-threads=2"
            self.video_decode = "! videoconvert ! video/x-raw, format=BGR"
            self.video_sink_conf = f"! appsink name={self.sink_name} emit-signals=false sync=false max-buffers=1 drop=true"

            self.video_pipe = None
            self.video_sink = None
            self.run_gst()

    def run_gst(self):
        command = f"{self.video_source} {self.video_codec} {self.video_decode} {self.video_sink_conf}"
        try:
            self.video_pipe = Gst.parse_launch(command)
            self.video_sink = self.video_pipe.get_by_name(self.sink_name)
            self.video_pipe.set_state(Gst.State.PLAYING)
        except Exception as e:
            print(f"[GstVideoReceiver] Error starting pipeline on port {self.port}: {e}", file=sys.stderr)

    def _v4l2_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                recv_time = time.time()
                with self.condition:
                    self.latest_frame_data = (frame, recv_time)
                    self.condition.notify()
            else:
                time.sleep(0.01)

    def get_latest_frame_blocking(self, timeout=0.2):
        if self.is_v4l2:
            with self.condition:
                while self.latest_frame_data is None and self.running:
                    self.condition.wait(timeout=timeout)
                if not self.running or self.latest_frame_data is None:
                    return None, None
                frame, recv_time = self.latest_frame_data
                self.latest_frame_data = None
                return frame, recv_time

        if not self.running or not self.video_sink:
            return None, None

        timeout_ns = int(timeout * 1e9)
        sample = self.video_sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return None, None

        # Drain any newer sample to guarantee the freshest real-time frame
        while True:
            newer = self.video_sink.emit("try-pull-sample", 0)
            if newer is None:
                break
            sample = newer

        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        h = caps_structure.get_value("height")
        w = caps_structure.get_value("width")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8).copy()
                recv_time = time.time()
                return frame, recv_time
            finally:
                buf.unmap(map_info)
        return None, None

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.cap:
            self.cap.release()
        if hasattr(self, "video_pipe") and self.video_pipe:
            self.video_pipe.set_state(Gst.State.NULL)


class Camera1Node(Node):
    """Camera 1 capture node."""
    def __init__(self, shared_buffer):
        super().__init__("camera1_node")
        self.shared_buffer = shared_buffer

        self.declare_parameter("cam1_port", 5600)
        self.cam1_port = self.get_parameter("cam1_port").value

        self.is_running = True
        self.get_logger().info(f"Connecting Primary Camera on port {self.cam1_port}...")
        self.video = GstVideoReceiver(port=self.cam1_port, name="cam1")

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Camera 1 ingestion thread started.")

    def _capture_loop(self):
        while rclpy.ok() and self.is_running:
            frame, recv_time = self.video.get_latest_frame_blocking(timeout=0.2)
            if frame is not None:
                self.shared_buffer.put(frame, recv_time)

    def destroy_node(self):
        self.is_running = False
        if hasattr(self, "video") and self.video:
            self.video.stop()
        super().destroy_node()
