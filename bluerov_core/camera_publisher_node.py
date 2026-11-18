#!/usr/bin/env python3
"""
BlueROV2 Camera Publisher ROS 2 Node
Reads GStreamer UDP feed and publishes raw sensor_msgs/Image with Real-Time QoS
"""

import os
import threading
import time
import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Εισαγωγή ρυθμίσεων QoS για real-time sensor streams (αποφυγή backlog)
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class Video:
    """BlueROV video capture class — BGR GStreamer pipeline for true color display."""
    def __init__(self, port=5600):
        Gst.init(None)
        self.port = port
        self.latest_frame_data = None
        self.running = True
        self.condition = threading.Condition()

        self.video_source = (
            f"udpsrc port={self.port} "
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            f"rtpjitterbuffer latency=0 drop-on-latency=true"
        )
        self.video_codec = "! rtph264depay ! h264parse ! avdec_h264 max-threads=2"
        self.video_decode = "! videoconvert ! video/x-raw, format=BGR"
        self.video_sink_conf = "! appsink name=appsink0 emit-signals=true sync=false max-buffers=1 drop=true"

        self.video_pipe = None
        self.video_sink = None
        self.run()

    def start_gst(self, config=None):
        if not config:
            config = ["videotestsrc", "! videoconvert ! video/x-raw, format=BGR", "! appsink name=appsink0"]
        command = " ".join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_sink = self.video_pipe.get_by_name("appsink0")
        self.video_pipe.set_state(Gst.State.PLAYING)

    @staticmethod
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        h = caps_structure.get_value("height")
        w = caps_structure.get_value("width")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8).copy()
            return frame
        finally:
            buf.unmap(map_info)

    def run(self):
        self.start_gst([self.video_source, self.video_codec, self.video_decode, self.video_sink_conf])
        self.video_sink.connect("new-sample", self.callback)

    def callback(self, sink):
        recv_time = time.time()
        sample = sink.emit("pull-sample")
        if sample:
            frame = self.gst_to_opencv(sample)
            if frame is not None:
                with self.condition:
                    self.latest_frame_data = (frame, recv_time)
                    self.condition.notify()
        return Gst.FlowReturn.OK

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
        if self.video_pipe:
            self.video_pipe.set_state(Gst.State.NULL)


class BlueROVCameraPublisher(Node):
    def __init__(self):
        super().__init__("camera_publisher_node")
        self.declare_parameter("udp_port", 5600)
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        
        self.port = self.get_parameter("udp_port").get_parameter_value().integer_value
        self.camera_frame_id = self.get_parameter("camera_frame_id").get_parameter_value().string_value
        
        self.cv_bridge = CvBridge()
        
        # QoS Profile: BEST_EFFORT, KEEP_LAST (Βάθος=1)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publisher με Best Effort QoS
        self.image_pub = self.create_publisher(Image, "rov/camera/image_raw", sensor_qos)
        
        self.get_logger().info(f"Starting GStreamer capture on port {self.port}...")
        self.video = Video(port=self.port)
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.worker_thread.start()

    def publish_loop(self):
        while rclpy.ok() and self.is_running:
            frame, _ = self.video.get_latest_frame_blocking()
            if frame is None:
                continue
            
            try:
                msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                # Time syncing
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
    node = BlueROVCameraPublisher()
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