#!/usr/bin/env python3
"""
Annotated Publisher Node (AnnotatedPublisherNode)
Monitors Foxglove subscriptions for Camera 1 and Camera 2 annotated streams.
Only when a subscriber is actively listening, it reads frames & tag detections from RAM,
renders overlays, encodes to lightweight JPEG (~100-150KB), and publishes CompressedImage.
"""

import threading
import time
import numpy as np
import cv2
import pupil_apriltags as apriltag

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

from bluerov_core.common import TAG_FAMILY, draw_tag_overlay


def make_compressed_msg(frame, frame_id, stamp, quality=75):
    """Encodes BGR OpenCV frame directly into a lightweight JPEG CompressedImage message."""
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        msg.data = buf.tobytes()
        return msg
    return None


class AnnotatedPublisherNode(Node):
    def __init__(self, cam1_buffer, cam1_detections, cam2_buffer=None):
        super().__init__("annotated_publisher_node")
        self.cam1_buffer = cam1_buffer
        self.cam1_detections = cam1_detections
        self.cam2_buffer = cam2_buffer

        # Parameters
        self.declare_parameter("output_width", 960)
        self.declare_parameter("jpeg_quality", 75)
        self.declare_parameter("enable_cam2", True)
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        self.declare_parameter("camera2_frame_id", "camera2_optical_frame")

        self.output_width = int(self.get_parameter("output_width").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.enable_cam2 = bool(self.get_parameter("enable_cam2").value)
        self.camera_frame = self.get_parameter("camera_frame_id").value
        self.camera2_frame = self.get_parameter("camera2_frame_id").value

        # Real-time QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.pub_annotated_cam1 = self.create_publisher(CompressedImage, "camera/image_annotated", sensor_qos)
        if self.enable_cam2:
            self.pub_cam2_annotated = self.create_publisher(CompressedImage, "camera2/image_annotated", sensor_qos)
            self.detector_cam2 = apriltag.Detector(
                families=TAG_FAMILY,
                nthreads=2,
                quad_decimate=4.0,
                quad_sigma=0.0,
                refine_edges=0
            )

        self.is_running = True

        # Workers
        self.cam1_worker = threading.Thread(target=self._cam1_publish_loop, daemon=True)
        self.cam1_worker.start()

        if self.enable_cam2 and self.cam2_buffer is not None:
            self.cam2_worker = threading.Thread(target=self._cam2_publish_loop, daemon=True)
            self.cam2_worker.start()

        self.get_logger().info("Annotated publisher worker loops started.")

    def _cam1_publish_loop(self):
        last_seq = -1
        while rclpy.ok() and self.is_running:
            # Check subscription count first to save CPU if Foxglove is closed
            if self.pub_annotated_cam1.get_subscription_count() == 0:
                time.sleep(0.05)
                continue

            frame, recv_time, seq = self.cam1_buffer.get_latest(last_seen_seq=last_seq, timeout=0.1)
            if frame is None or seq == last_seq:
                continue
            last_seq = seq

            stamp = self.get_clock().now().to_msg()
            h, w = frame.shape[:2]
            new_w = self.output_width
            new_h = int(h * (new_w / float(w)))

            small_cam1 = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_x = float(new_w) / float(w)
            scale_y = float(new_h) / float(h)

            detected_tags = self.cam1_detections.get_for_seq(seq)
            for tag_id, corners in detected_tags.items():
                scaled_corners = corners.copy()
                scaled_corners[:, 0] *= scale_x
                scaled_corners[:, 1] *= scale_y
                draw_tag_overlay(small_cam1, tag_id, scaled_corners)

            comp_annotated = make_compressed_msg(
                small_cam1, self.camera_frame, stamp, quality=self.jpeg_quality
            )
            if comp_annotated:
                self.pub_annotated_cam1.publish(comp_annotated)

    def _cam2_publish_loop(self):
        last_seq = -1
        while rclpy.ok() and self.is_running:
            if self.pub_cam2_annotated.get_subscription_count() == 0:
                time.sleep(0.05)
                continue

            frame, recv_time, seq = self.cam2_buffer.get_latest(last_seen_seq=last_seq, timeout=0.1)
            if frame is None or seq == last_seq:
                continue
            last_seq = seq

            stamp = self.get_clock().now().to_msg()
            h, w = frame.shape[:2]
            new_w = self.output_width
            new_h = int(h * (new_w / float(w)))

            small_cam2 = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gray2 = cv2.cvtColor(small_cam2, cv2.COLOR_BGR2GRAY)
            detections2 = self.detector_cam2.detect(gray2)

            for r in detections2:
                draw_tag_overlay(small_cam2, r.tag_id, r.corners, color=(0, 255, 0))

            comp_annotated = make_compressed_msg(
                small_cam2, self.camera2_frame, stamp, quality=self.jpeg_quality
            )
            if comp_annotated:
                self.pub_cam2_annotated.publish(comp_annotated)

    def destroy_node(self):
        self.is_running = False
        super().destroy_node()
