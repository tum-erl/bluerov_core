#!/usr/bin/env python3
"""
BlueROV2 All-in-One Autonomous System Node (bluerov_system_node)

Consolidates the complete BlueROV2 visual pipeline into a single high-performance node:
  1. Internal GStreamer UDP ingestion for Primary Camera (port 5600) and Secondary Camera (port 5604).
  2. In-memory zero-copy AprilTag detection (quad_decimate=2.0 for rock-solid 30+ FPS).
  3. Textbook Dual-Engine PnP:
       - Multi-Tag (>= 2 tags): SQPNP with Levenberg-Marquardt refinement.
       - Single-Tag (1 tag or fallback): IPPE_SQUARE with rotation continuity (eliminates planar flip ambiguity).
  4. Robust EMA pose filtering & ROS FLU coordinate transformation.
  5. Full ROS TF broadcast (world -> rov, rov -> camera_optical_frame, rov -> camera2_optical_frame, world -> tag_X).
  6. Bandwidth-Optimized Compressed Streams (JPEG ~100-150KB instead of 6.2MB uncompressed):
       - /camera/image_annotated (CompressedImage with tag bounding boxes)
       - /camera2/image_raw (CompressedImage for auxiliary camera)
       - /rov/pose (PoseStamped in FLU coordinates)
       - /rov/tags_markers (Clean visual tag cubes with DELETEALL)
"""

import os
import sys
import threading
import time
import cv2
import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pupil_apriltags as apriltag

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# ==============================================================================
# 1. GLOBAL CONFIGURATION & TAG MAP
# ==============================================================================
REFERENCE_TAG_ID = 2
TAG_FAMILY = "tagStandard41h12"
SMALL_TAG_SIZE = 0.05
BIG_TAG_SIZE = 0.126

TAG_MAP = {
    2: {"pos": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "size": BIG_TAG_SIZE},
    1: {"pos": [0.255, 0.285, 0.0], "rpy": [0.0, 0.0, 0.0], "size": BIG_TAG_SIZE},
    3: {"pos": [-0.30, 0.13, 0.0], "rpy": [0.0, 90.0, 0.0], "size": BIG_TAG_SIZE},
    9: {"pos": [0.0, 0.575, 0.0], "rpy": [270.0, 0.0, 0.0], "size": BIG_TAG_SIZE},

    1001: {"pos": [0.130, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "size": SMALL_TAG_SIZE},
    1003: {"pos": [0.285, 0.098, 0.0], "rpy": [0.0, 270.0, 0.0], "size": SMALL_TAG_SIZE}
}

# ROV Camera calibration parameters in water
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FX, FY, CX, CY = 1055.3052, 1054.4018, 977.0207, 600.9562
K = np.array([[FX, 0.0, CX],[0.0, FY, CY],[0.0, 0.0, 1.0]], dtype=np.float64)
D = np.array([-0.00458762, 0.06154082, 0.00437917, 0.00265318, -0.06060720], dtype=np.float64)


# Camera offset relative to ROV body center (in meters, optical frame)
CAMERA_TO_BODY_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Rotation matrix: Optical Camera Frame (X right, Y down, Z forward) -> FLU Body Frame (X fwd, Y left, Z up)
R_OPTICAL_TO_BODY = np.array([
    [ 0.0,  0.0,  1.0], 
    [-1.0,  0.0,  0.0], 
    [ 0.0, -1.0,  0.0], 
], dtype=np.float64)

TAG_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), 
    (255, 0, 255), (255, 255, 0), (128, 0, 128), (255, 165, 0)
]

# ==============================================================================
# 2. MATH HELPERS & TRANSFORMATIONS
# ==============================================================================
def rpy_to_rotation_matrix(rpy_deg):
    r_rad, p_rad, y_rad = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r_rad), -np.sin(r_rad)], [0, np.sin(r_rad), np.cos(r_rad)]])
    Ry = np.array([[np.cos(p_rad), 0, np.sin(p_rad)], [0, 1, 0], [-np.sin(p_rad), 0, np.cos(p_rad)]])
    Rz = np.array([[np.cos(y_rad), -np.sin(y_rad), 0], [np.sin(y_rad), np.cos(y_rad), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def rotation_matrix_to_rpy(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])])
    else:
        return np.degrees([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])

def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        return (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S, 0.25 * S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S, (R[2, 1] - R[1, 2]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S, (R[0, 2] - R[2, 0]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        return (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S, (R[1, 0] - R[0, 1]) / S

def quaternion_nlerp(q1, q2, alpha=0.5):
    """Normalized linear quaternion interpolation with antipodal flip handling."""
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    q = (1.0 - alpha) * q1 + alpha * q2
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else q1

def build_tag_tables(tag_map):
    global_map, local_map, trans_map = {}, {}, {}
    for tag_id, transform in tag_map.items():
        s = transform["size"] / 2.0
        # Local corners in tag frame: top-left, top-right, bottom-right, bottom-left
        local_corners = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)
        R_tw = rpy_to_rotation_matrix(transform["rpy"])
        t_tw = np.array(transform["pos"], dtype=np.float64).reshape(3, 1)
        global_map[tag_id] = (R_tw @ local_corners.T + t_tw).T.astype(np.float64)
        local_map[tag_id] = local_corners
        trans_map[tag_id] = (R_tw, t_tw)
    return global_map, local_map, trans_map

GLOBAL_TAG_CORNERS, LOCAL_TAG_CORNERS, TAG_TRANSFORMS = build_tag_tables(TAG_MAP)

def ippe_solution_to_world_pos(rvec, tvec, R_tw, t_tw):
    """Converts single-tag local PnP pose to camera and body pose in world optical frame."""
    R_ct, _ = cv2.Rodrigues(rvec)
    R_wc = R_tw @ R_ct.T
    t_wc = R_tw @ (-R_ct.T @ tvec.reshape(3, 1)) + t_tw
    t_wb = t_wc + R_wc @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)
    return R_wc, t_wb, t_wc

def draw_tag_overlay(display_frame, tag_id, corners_2d, color=None):
    """Clean standard visualization overlay for detected tags."""
    if color is None:
        color = TAG_COLORS[hash(tag_id) % len(TAG_COLORS)]
    cv2.polylines(display_frame, [corners_2d.astype(np.int32).reshape((-1, 1, 2))], True, color, 3)
    cx, cy = int(np.mean(corners_2d[:, 0])), int(np.mean(corners_2d[:, 1]))
    cv2.putText(display_frame, f"ID {tag_id}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

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

# ==============================================================================
# 3. GSTREAMER / OPENCV VIDEO INGESTION CLASS
# ==============================================================================
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


# ==============================================================================
# 4. ALL-IN-ONE BLUEROV SYSTEM NODE
# ==============================================================================
class BlueROVSystemNode(Node):
    def __init__(self):
        super().__init__("bluerov_system_node")

        # Parameters
        self.declare_parameter("cam1_port", 5600)
        self.declare_parameter("cam2_port", 5604)
        self.declare_parameter("enable_cam2", True)
        self.declare_parameter("jpeg_quality", 75)
        self.declare_parameter("output_width", 960) # Downscale to 960x540 for lightning-fast JPEG encoding & Foxglove display
        self.declare_parameter("quad_decimate", 2.0) # 2.0 for 30+ FPS speed with subpixel accuracy
        self.declare_parameter("cam2_quad_decimate", 4.0) # 4.0 for Cam2 gives ~2-3ms detection without localization overhead
        self.declare_parameter("filter_alpha", 0.50)
        self.declare_parameter("max_xy_bound", 2.5)
        self.declare_parameter("max_z_bound", 0.50)
        self.declare_parameter("max_lost_frames", 30)

        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("rov_frame_id", "rov")
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        self.declare_parameter("camera2_frame_id", "camera2_optical_frame")

        self.cam1_port = self.get_parameter("cam1_port").value
        self.cam2_port = self.get_parameter("cam2_port").value
        self.enable_cam2 = bool(self.get_parameter("enable_cam2").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.output_width = int(self.get_parameter("output_width").value)
        self.quad_decimate = float(self.get_parameter("quad_decimate").value)
        self.cam2_quad_decimate = float(self.get_parameter("cam2_quad_decimate").value)
        self.filter_alpha = float(self.get_parameter("filter_alpha").value)
        self.max_xy_bound = float(self.get_parameter("max_xy_bound").value)
        self.max_z_bound = float(self.get_parameter("max_z_bound").value)
        self.max_lost_frames = int(self.get_parameter("max_lost_frames").value)

        self.world_frame = self.get_parameter("world_frame_id").value
        self.rov_frame = self.get_parameter("rov_frame_id").value
        self.camera_frame = self.get_parameter("camera_frame_id").value
        self.camera2_frame = self.get_parameter("camera2_frame_id").value

        self.cv_bridge = CvBridge()
        self.is_running = True

        # Real-time QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ----------------------------------------------------------------------
        # Publishers (Clean & Minimal: Only topics Foxglove actually uses)
        # ----------------------------------------------------------------------
        # Camera 1 (Annotated stream for Navigation): CompressedImage
        self.pub_annotated_cam1 = self.create_publisher(CompressedImage, "camera/image_annotated", sensor_qos)

        # Camera 2 (Annotated inspection stream): CompressedImage
        if self.enable_cam2:
            self.pub_cam2_annotated = self.create_publisher(CompressedImage, "camera2/image_annotated", sensor_qos)

        # Navigation & TF
        self.pose_pub = self.create_publisher(PoseStamped, "rov/pose", 10)
        self.human_pub = self.create_publisher(String, "rov/pose_human_readable", 10)
        self.tag_marker_pub = self.create_publisher(MarkerArray, "rov/tags_markers", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Publish static tags and camera frames once
        self.publish_static_transforms()
        self.publish_tag_markers()
        self.marker_timer = self.create_timer(3.0, self.publish_tag_markers)

        # Initialize AprilTag Detectors
        self.detector = apriltag.Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=self.quad_decimate,
            quad_sigma=0.0,
            refine_edges=1
        )
        if self.enable_cam2:
            self.detector_cam2 = apriltag.Detector(
                families=TAG_FAMILY,
                nthreads=2,
                quad_decimate=4.0,
                quad_sigma=0.0,
                refine_edges=0
            )

        # Tracking & FPS State
        self.frame_count = 0
        self.lost_frames_count = 0
        self.prev_world_pos = None
        self.prev_R_wc = None
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        self.filtered_pos = None
        self.filtered_quat = None
        self.filtered_cam_pos = None
        self.filtered_cam_quat = None
        self.last_R_wb_flu = None

        # Start GStreamer Ingestion
        self.get_logger().info(f"Connecting Primary Camera on port {self.cam1_port}...")
        self.video1 = GstVideoReceiver(port=self.cam1_port, name="cam1")

        if self.enable_cam2:
            self.get_logger().info(f"Connecting Secondary Camera on port {self.cam2_port}...")
            self.video2 = GstVideoReceiver(port=self.cam2_port, name="cam2")
            self.cam2_thread = threading.Thread(target=self.cam2_worker_loop, daemon=True)
            self.cam2_thread.start()

        # Start Camera 1 Localization Loop
        self.cam1_thread = threading.Thread(target=self.cam1_worker_loop, daemon=True)
        self.cam1_thread.start()

        self.get_logger().info("BlueROV All-in-One System Node successfully started.")

    def publish_static_transforms(self):
        """Broadcasts permanent static transforms for camera frames and all tags in TAG_MAP."""
        static_transforms = []
        now = self.get_clock().now().to_msg()

        # Primary Camera: rov -> camera_optical_frame
        t_cam1 = TransformStamped()
        t_cam1.header.stamp = now
        t_cam1.header.frame_id = self.rov_frame
        t_cam1.child_frame_id = self.camera_frame
        cam_off_flu = R_OPTICAL_TO_BODY @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)
        t_cam1.transform.translation.x = float(cam_off_flu[0])
        t_cam1.transform.translation.y = float(cam_off_flu[1])
        t_cam1.transform.translation.z = float(cam_off_flu[2])
        qx_c, qy_c, qz_c, qw_c = rotation_matrix_to_quaternion(R_OPTICAL_TO_BODY)
        t_cam1.transform.rotation.x = float(qx_c)
        t_cam1.transform.rotation.y = float(qy_c)
        t_cam1.transform.rotation.z = float(qz_c)
        t_cam1.transform.rotation.w = float(qw_c)
        static_transforms.append(t_cam1)

        # Secondary Camera: rov -> camera2_optical_frame
        if self.enable_cam2:
            t_cam2 = TransformStamped()
            t_cam2.header.stamp = now
            t_cam2.header.frame_id = self.rov_frame
            t_cam2.child_frame_id = self.camera2_frame
            t_cam2.transform.translation.x = 0.0
            t_cam2.transform.translation.y = 0.0
            t_cam2.transform.translation.z = 0.0
            t_cam2.transform.rotation.x = float(qx_c)
            t_cam2.transform.rotation.y = float(qy_c)
            t_cam2.transform.rotation.z = float(qz_c)
            t_cam2.transform.rotation.w = float(qw_c)
            static_transforms.append(t_cam2)

        # Static Tags: world -> tag_X
        for tag_id, tag_data in TAG_MAP.items():
            pos = tag_data["pos"]
            rpy = tag_data["rpy"]
            pos_flu = R_OPTICAL_TO_BODY @ np.array(pos, dtype=np.float64).reshape(3, 1)
            R_tw = rpy_to_rotation_matrix(rpy)
            R_tw_flu = R_OPTICAL_TO_BODY @ R_tw @ R_OPTICAL_TO_BODY.T
            qx_t, qy_t, qz_t, qw_t = rotation_matrix_to_quaternion(R_tw_flu)

            t_tag = TransformStamped()
            t_tag.header.stamp = now
            t_tag.header.frame_id = self.world_frame
            t_tag.child_frame_id = f"tag_{tag_id}"
            t_tag.transform.translation.x = float(pos_flu[0])
            t_tag.transform.translation.y = float(pos_flu[1])
            t_tag.transform.translation.z = float(pos_flu[2])
            t_tag.transform.rotation.x = float(qx_t)
            t_tag.transform.rotation.y = float(qy_t)
            t_tag.transform.rotation.z = float(qz_t)
            t_tag.transform.rotation.w = float(qw_t)
            static_transforms.append(t_tag)

        self.static_tf_broadcaster.sendTransform(static_transforms)

    def publish_tag_markers(self):
        """Publishes clean 3D cubes for all mapped tags, with DELETEALL header."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.header.stamp = now
        clear_marker.header.frame_id = self.world_frame
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for tag_id, tag_data in TAG_MAP.items():
            pos = tag_data["pos"]
            rpy = tag_data["rpy"]
            size = tag_data.get("size", BIG_TAG_SIZE)

            pos_flu = R_OPTICAL_TO_BODY @ np.array(pos, dtype=np.float64).reshape(3, 1)
            R_tw = rpy_to_rotation_matrix(rpy)
            R_tw_flu = R_OPTICAL_TO_BODY @ R_tw @ R_OPTICAL_TO_BODY.T
            qx_t, qy_t, qz_t, qw_t = rotation_matrix_to_quaternion(R_tw_flu)

            square_marker = Marker()
            square_marker.header.stamp = now
            square_marker.header.frame_id = self.world_frame
            square_marker.ns = "apriltag_squares"
            square_marker.id = int(tag_id)
            square_marker.type = Marker.CUBE
            square_marker.action = Marker.ADD

            square_marker.pose.position.x = float(pos_flu[0])
            square_marker.pose.position.y = float(pos_flu[1])
            square_marker.pose.position.z = float(pos_flu[2])
            square_marker.pose.orientation.x = float(qx_t)
            square_marker.pose.orientation.y = float(qy_t)
            square_marker.pose.orientation.z = float(qz_t)
            square_marker.pose.orientation.w = float(qw_t)

            square_marker.scale.x = 0.005
            square_marker.scale.y = float(size)
            square_marker.scale.z = float(size)

            square_marker.color.r = 0.1
            square_marker.color.g = 0.1
            square_marker.color.b = 0.1
            square_marker.color.a = 0.9

            marker_array.markers.append(square_marker)

        self.tag_marker_pub.publish(marker_array)

    def is_pose_physically_valid(self, t_wb, R_wc=None):
        x = float(t_wb[0])
        y = float(t_wb[1])
        z = float(t_wb[2])

        if z > self.max_z_bound:
            return False
        if abs(x) > self.max_xy_bound or abs(y) > self.max_xy_bound:
            return False
        return True

    # --------------------------------------------------------------------------
    # SECONDARY CAMERA WORKER (Auxiliary Stream with Tag Detection)
    # --------------------------------------------------------------------------
    def cam2_worker_loop(self):
        while rclpy.ok() and self.is_running:
            frame, _ = self.video2.get_latest_frame_blocking(timeout=0.2)
            if frame is None:
                continue

            stamp = self.get_clock().now().to_msg()

            # Process & publish only if Foxglove is subscribed to camera2/image_annotated
            if self.pub_cam2_annotated.get_subscription_count() > 0:
                h, w = frame.shape[:2]
                new_w = self.output_width
                new_h = int(h * (new_w / float(w)))

                # Downscale directly: 4x faster detection, zero 6MB full-frame copying
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

    # --------------------------------------------------------------------------
    # PRIMARY CAMERA WORKER (Localization & Tag Overlay)
    # --------------------------------------------------------------------------
    def cam1_worker_loop(self):
        while rclpy.ok() and self.is_running:
            bgr_frame, recv_time = self.video1.get_latest_frame_blocking(timeout=0.2)
            if bgr_frame is None:
                continue

            self.frame_count += 1
            now_stamp = self.get_clock().now().to_msg()

            # 1. AprilTag Detection
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            detected_tags = {}
            for r in self.detector.detect(gray):
                if r.tag_id in GLOBAL_TAG_CORNERS:
                    detected_tags[r.tag_id] = r.corners

            # 2. Draw Visual Overlay & Publish Compressed Image directly to Foxglove
            if self.pub_annotated_cam1.get_subscription_count() > 0:
                h, w = bgr_frame.shape[:2]
                new_w = self.output_width
                new_h = int(h * (new_w / float(w)))

                small_cam1 = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_x = float(new_w) / float(w)
                scale_y = float(new_h) / float(h)
                for tag_id, corners in detected_tags.items():
                    scaled_corners = corners.copy()
                    scaled_corners[:, 0] *= scale_x
                    scaled_corners[:, 1] *= scale_y
                    draw_tag_overlay(small_cam1, tag_id, scaled_corners)

                comp_annotated = make_compressed_msg(
                    small_cam1, self.camera_frame, now_stamp, quality=self.jpeg_quality
                )
                if comp_annotated:
                    self.pub_annotated_cam1.publish(comp_annotated)

            # 3. Dual-Engine PnP Solver
            num_tags = len(detected_tags)
            solved_new_pose = False
            R_wc_candidate = None
            t_wb_candidate = None
            t_wc_candidate = None
            mode_str = "NONE"

            # Engine A: Multi-Tag PnP (>= 2 tags)
            if num_tags >= 2:
                obj_pts = np.vstack([GLOBAL_TAG_CORNERS[tid] for tid in detected_tags.keys()]).astype(np.float64)
                img_pts = np.vstack([detected_tags[tid] for tid in detected_tags.keys()]).astype(np.float64)

                ok, rvec, tvec = False, None, None
                try:
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_SQPNP)
                except Exception:
                    ok = False

                if not ok:
                    try:
                        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_EPNP)
                    except Exception:
                        ok = False

                if ok and rvec is not None and tvec is not None:
                    try:
                        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)
                    except Exception:
                        pass

                    R_cw, _ = cv2.Rodrigues(rvec)
                    R_wc = R_cw.T
                    t_wc = -R_wc @ tvec.reshape(3, 1)
                    t_wb = t_wc + R_wc @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)

                    if self.is_pose_physically_valid(t_wb, R_wc):
                        R_wc_candidate = R_wc
                        t_wb_candidate = t_wb
                        t_wc_candidate = t_wc
                        solved_new_pose = True
                        mode_str = f"MULTI (tags {sorted(list(detected_tags.keys()))})"

            # Engine B: Single-Tag PnP (1 tag or fallback)
            if not solved_new_pose and num_tags >= 1:
                tag_id = REFERENCE_TAG_ID if REFERENCE_TAG_ID in detected_tags else list(detected_tags.keys())[0]
                corners = detected_tags[tag_id]
                R_tw, t_tw = TAG_TRANSFORMS[tag_id]

                candidate_solutions = []
                try:
                    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                        LOCAL_TAG_CORNERS[tag_id], corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if retval > 0 and len(rvecs) > 0:
                        for rv, tv in zip(rvecs, tvecs):
                            candidate_solutions.append((rv, tv))
                except Exception:
                    pass

                if not candidate_solutions:
                    try:
                        ok, rv, tv = cv2.solvePnP(LOCAL_TAG_CORNERS[tag_id], corners, K, D, flags=cv2.SOLVEPNP_IPPE)
                        if ok:
                            candidate_solutions.append((rv, tv))
                    except Exception:
                        pass

                if candidate_solutions:
                    valid_candidates = []
                    for rv, tv in candidate_solutions:
                        R_wc, t_wb, t_wc = ippe_solution_to_world_pos(rv, tv, R_tw, t_tw)
                        if self.is_pose_physically_valid(t_wb, R_wc):
                            # Position continuity
                            pos_cost = float(np.linalg.norm(t_wb - self.prev_world_pos)) if self.prev_world_pos is not None else 0.0

                            # Orientation continuity: eliminates planar flip ambiguity
                            rot_cost = float(np.linalg.norm(R_wc - self.prev_R_wc)) if self.prev_R_wc is not None else -float(R_wc[2, 2])

                            total_cost = pos_cost + 0.35 * rot_cost
                            valid_candidates.append((total_cost, R_wc, t_wb, t_wc))

                    if valid_candidates:
                        valid_candidates.sort(key=lambda item: item[0])
                        _, R_wc_candidate, t_wb_candidate, t_wc_candidate = valid_candidates[0]
                        solved_new_pose = True
                        mode_str = f"SINGLE (tag {tag_id})"

            # 4. Filtering & Output
            if solved_new_pose and t_wb_candidate is not None:
                self.lost_frames_count = 0
                self.prev_world_pos = t_wb_candidate.copy()
                self.prev_R_wc = R_wc_candidate.copy()

                # Optical -> ROS FLU
                t_wb_flu = R_OPTICAL_TO_BODY @ t_wb_candidate
                t_wc_flu = R_OPTICAL_TO_BODY @ t_wc_candidate
                R_wb_flu = R_OPTICAL_TO_BODY @ R_wc_candidate @ R_OPTICAL_TO_BODY.T
                R_wc_flu = R_OPTICAL_TO_BODY @ R_wc_candidate

                qx_r, qy_r, qz_r, qw_r = rotation_matrix_to_quaternion(R_wb_flu)
                qx_c, qy_c, qz_c, qw_c = rotation_matrix_to_quaternion(R_wc_flu)

                raw_pos = np.array([float(t_wb_flu[0]), float(t_wb_flu[1]), float(t_wb_flu[2])])
                raw_quat = [float(qx_r), float(qy_r), float(qz_r), float(qw_r)]
                raw_cam_pos = np.array([float(t_wc_flu[0]), float(t_wc_flu[1]), float(t_wc_flu[2])])
                raw_cam_quat = [float(qx_c), float(qy_c), float(qz_c), float(qw_c)]

                if self.filtered_pos is None:
                    self.filtered_pos = raw_pos
                    self.filtered_quat = raw_quat
                    self.filtered_cam_pos = raw_cam_pos
                    self.filtered_cam_quat = raw_cam_quat
                else:
                    alpha = self.filter_alpha
                    self.filtered_pos = alpha * raw_pos + (1.0 - alpha) * self.filtered_pos
                    self.filtered_quat = list(quaternion_nlerp(self.filtered_quat, raw_quat, alpha=alpha))
                    self.filtered_cam_pos = alpha * raw_cam_pos + (1.0 - alpha) * self.filtered_cam_pos
                    self.filtered_cam_quat = list(quaternion_nlerp(self.filtered_cam_quat, raw_cam_quat, alpha=alpha))

                self.last_R_wb_flu = R_wb_flu
            else:
                self.lost_frames_count += 1
                if self.lost_frames_count > self.max_lost_frames:
                    self.filtered_pos = None
                    self.filtered_quat = None
                    self.prev_world_pos = None
                    self.prev_R_wc = None

            # 5. Broadcast & Publish Pose
            if self.filtered_pos is not None:
                x_b, y_b, z_b = self.filtered_pos
                qx_r, qy_r, qz_r, qw_r = self.filtered_quat
                x_c, y_c, z_c = self.filtered_cam_pos
                qx_wc, qy_wc, qz_wc, qw_wc = self.filtered_cam_quat

                # ROV Pose (FLU)
                pose_msg = PoseStamped()
                pose_msg.header.stamp = now_stamp
                pose_msg.header.frame_id = self.world_frame
                pose_msg.pose.position.x = float(x_b)
                pose_msg.pose.position.y = float(y_b)
                pose_msg.pose.position.z = float(z_b)
                pose_msg.pose.orientation.x = float(qx_r)
                pose_msg.pose.orientation.y = float(qy_r)
                pose_msg.pose.orientation.z = float(qz_r)
                pose_msg.pose.orientation.w = float(qw_r)
                self.pose_pub.publish(pose_msg)

                # TF: world -> rov
                t_stamped_rov = TransformStamped()
                t_stamped_rov.header.stamp = now_stamp
                t_stamped_rov.header.frame_id = self.world_frame
                t_stamped_rov.child_frame_id = self.rov_frame
                t_stamped_rov.transform.translation.x = float(x_b)
                t_stamped_rov.transform.translation.y = float(y_b)
                t_stamped_rov.transform.translation.z = float(z_b)
                t_stamped_rov.transform.rotation.x = float(qx_r)
                t_stamped_rov.transform.rotation.y = float(qy_r)
                t_stamped_rov.transform.rotation.z = float(qz_r)
                t_stamped_rov.transform.rotation.w = float(qw_r)

                # TF: rov -> camera_optical_frame
                qx_c_cam, qy_c_cam, qz_c_cam, qw_c_cam = rotation_matrix_to_quaternion(R_OPTICAL_TO_BODY)
                cam_off_flu = R_OPTICAL_TO_BODY @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)
                t_stamped_cam = TransformStamped()
                t_stamped_cam.header.stamp = now_stamp
                t_stamped_cam.header.frame_id = self.rov_frame
                t_stamped_cam.child_frame_id = self.camera_frame
                t_stamped_cam.transform.translation.x = float(cam_off_flu[0])
                t_stamped_cam.transform.translation.y = float(cam_off_flu[1])
                t_stamped_cam.transform.translation.z = float(cam_off_flu[2])
                t_stamped_cam.transform.rotation.x = float(qx_c_cam)
                t_stamped_cam.transform.rotation.y = float(qy_c_cam)
                t_stamped_cam.transform.rotation.z = float(qz_c_cam)
                t_stamped_cam.transform.rotation.w = float(qw_c_cam)

                self.tf_broadcaster.sendTransform([t_stamped_rov, t_stamped_cam])

                if self.last_R_wb_flu is not None:
                    roll, pitch, yaw = rotation_matrix_to_rpy(self.last_R_wb_flu)
                    rpy_str = f"RPY: R={roll:+.1f}° P={pitch:+.1f}° Y={yaw:+.1f}°"
                else:
                    rpy_str = ""

                status_text = "TRACKING" if self.lost_frames_count == 0 else f"HOLDING ({self.lost_frames_count}/{self.max_lost_frames})"
                msg_str = f"[{status_text} | {mode_str}] ROV: X={x_b:+.3f}, Y={y_b:+.3f}, Z={z_b:+.3f} | {rpy_str}"
                self.human_pub.publish(String(data=msg_str))

            # Periodic terminal print & Live FPS measurement (1 Hz for zero terminal overhead)
            self.fps_frame_count += 1
            now_time = time.time()
            dt = now_time - self.last_fps_time
            if dt >= 1.0:
                self.current_fps = self.fps_frame_count / dt
                self.fps_frame_count = 0
                self.last_fps_time = now_time
                if self.filtered_pos is not None:
                    x_b, y_b, z_b = self.filtered_pos
                    print(f"[{self.current_fps:4.1f} Hz] ROV FLU: X={x_b:+.3f}, Y={y_b:+.3f}, Z={z_b:+.3f} | {mode_str}")

    def destroy_node(self):
        self.is_running = False
        if hasattr(self, "video1") and self.video1:
            self.video1.stop()
        if hasattr(self, "video2") and self.video2:
            self.video2.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVSystemNode()
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
