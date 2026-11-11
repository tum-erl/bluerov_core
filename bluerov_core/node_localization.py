#!/usr/bin/env python3
"""
BlueROV2 AprilTag Camera Localization ROS 2 Node (Jazzy)
Node: rov_localization_node
Features:
- Event-Driven Processing with dedicated worker thread (Condition variable)
- Low-latency direct GRAY8 GStreamer pipeline (avdec_h264 for ARM/RPi CPU)
- Adaptive Multi-ROI Tracking with Periodic Full Scan for new tag discovery
- Dictionary-based AprilTag deduplication for robust PnP solving
- Hardware timestamp propagation from GStreamer arrival to ROS 2 header
- Clean thread join & shutdown cleanup
Topics Published:
- /rov/pose (geometry_msgs/msg/PoseStamped)
- /rov/pose_human_readable (std_msgs/msg/String)
"""

import threading
import time
import cv2
import gi
import numpy as np
import pupil_apriltags as apriltag

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

# ==============================================================================
# 1. GLOBAL CONFIGURATION & TAG MAP
# ==============================================================================
REFERENCE_TAG_ID = 2  # Defines origin of coordinate system (0,0,0)
TAG_FAMILY = "tagStandard41h12"

# Μεγέθη tags (σε μέτρα)
SMALL_TAG_SIZE = 0.05
BIG_TAG_SIZE = 0.125

TAG_MAP = {
    # 3: {
    #     "pos": [-0.30, -0.13, 0.0], 
    #     "rpy": [0.0, 270.0, 0.0],
    #     "size": BIG_TAG_SIZE,
    # },
    # 1: {
    #     "pos": [0.263, -0.283, 0.0],
    #     "rpy": [0.0, 0.0, 0.0],
    #     "size": BIG_TAG_SIZE,
    # },
    2: {
        "pos": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
        "size": BIG_TAG_SIZE,
    }

    # 1000: {
    #     "pos": [-0.28, -0.575, 0.0],
    #     "rpy": [0.0, 0.0, 0.0],
    #     "size": SMALL_TAG_SIZE,
    # },

    # 1001: {
    #     "pos": [0.254, -0.575, 0.0],
    #     "rpy": [0.0, 0.0, 0.0],
    #     "size": SMALL_TAG_SIZE,
    # },

    # 1002: {
    #     "pos": [-0.235, 0.2, 0.0],
    #     "rpy": [270.0, 180.0, 0.0],
    #     "size": SMALL_TAG_SIZE,
    # },

    # 1003: {
    #     "pos": [0.285, -0.098, 0.0],
    #     "rpy": [0.0, 90.0, 0.0],
    #     "size": SMALL_TAG_SIZE,
    # }
}

# ==============================================================================
# 2. CAMERA INTRINSICS & RIGID OFFSETS
# ==============================================================================
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

FX = 1031.7610
FY = 1031.2645
CX = 958.9953
CY = 594.2563

K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

D = np.array([-0.03125212, 0.12170703, -0.00131824, -0.00245868, -0.10696144], dtype=np.float64)


CAMERA_TO_BODY_OFFSET = np.array([[0.0], [0.0], [-0.00]])


# ==============================================================================
# 3. BLUEROV2 VIDEO CAPTURE CLASS (DIRECT GRAY8 & RPI OPTIMIZED)
# ==============================================================================
class Video:
    """BlueROV video capture class με βελτιστοποιημένο GRAY8 GStreamer pipeline."""

    def __init__(self, port=5600):
        Gst.init(None)

        self.port = port
        self.latest_frame_data = None  # (frame, arrival_time)
        self.running = True
        self.condition = threading.Condition()

        self.video_source = (
            f"udpsrc port={self.port} "
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            f"rtpjitterbuffer latency=0 drop-on-latency=true"
        )
        self.video_codec = "! rtph264depay ! h264parse ! avdec_h264 max-threads=2"
        self.video_decode = "! videoconvert ! video/x-raw, format=GRAY8"
        self.video_sink_conf = (
            "! appsink name=appsink0 emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        if not config:
            config = [
                "videotestsrc",
                "! videoconvert ! video/x-raw, format=GRAY8",
                "! appsink name=appsink0",
            ]

        command = " ".join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_sink = self.video_pipe.get_by_name("appsink0")
        self.video_pipe.set_state(Gst.State.PLAYING)

    @staticmethod
    def gst_to_opencv(sample):
        """Εξάγει απευθείας 2D uint8 NumPy array (H, W) για Grayscale επεξεργασία."""
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        h = caps_structure.get_value("height")
        w = caps_structure.get_value("width")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            frame = np.ndarray((h, w), buffer=map_info.data, dtype=np.uint8).copy()
            return frame
        finally:
            buf.unmap(map_info)

    def run(self):
        self.start_gst([
            self.video_source,
            self.video_codec,
            self.video_decode,
            self.video_sink_conf,
        ])
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
        """Μπλοκάρει αποδοτικά μέχρι να φτάσει νέο frame από το GStreamer."""
        with self.condition:
            while self.latest_frame_data is None and self.running:
                self.condition.wait(timeout=0.1)

            if not self.running:
                return None, None

            frame, recv_time = self.latest_frame_data
            self.latest_frame_data = None  # Consume frame
            return frame, recv_time

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.video_pipe:
            self.video_pipe.set_state(Gst.State.NULL)


# ==============================================================================
# 4. HELPER FUNCTIONS: ROTATIONS & 3D CORNER GENERATION
# ==============================================================================
def rpy_to_rotation_matrix(rpy_deg):
    r_rad, p_rad, y_rad = np.radians(rpy_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r_rad), -np.sin(r_rad)],
        [0, np.sin(r_rad), np.cos(r_rad)],
    ])
    Ry = np.array([
        [np.cos(p_rad), 0, np.sin(p_rad)],
        [0, 1, 0],
        [-np.sin(p_rad), 0, np.cos(p_rad)],
    ])
    Rz = np.array([
        [np.cos(y_rad), -np.sin(y_rad), 0],
        [np.sin(y_rad), np.cos(y_rad), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def rotation_matrix_to_rpy(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return qx, qy, qz, qw


def build_global_tag_corners(tag_map):
    global_corners_map = {}
    for tag_id, transform in tag_map.items():
        s = transform["size"] / 2.0
        # Corner order must match pupil_apriltags detector output:
        # 0: lower-left, 1: lower-right, 2: upper-right, 3: upper-left
        local_corners = np.array(
            [
                [-s, -s, 0.0],
                [s, -s, 0.0],
                [s, s, 0.0],
                [-s, s, 0.0],
            ],
            dtype=np.float32,
        )

        R = rpy_to_rotation_matrix(transform["rpy"])
        t = np.array(transform["pos"], dtype=np.float32).reshape(3, 1)

        corners_global = (R @ local_corners.T + t).T
        global_corners_map[tag_id] = corners_global

    return global_corners_map


GLOBAL_TAG_CORNERS = build_global_tag_corners(TAG_MAP)


# ==============================================================================
# 5. ROI TRACKING DATA STRUCTURE
# ==============================================================================
class TrackedTag:
    """Διατηρεί κατάσταση εντοπισμού και υπολογίζει adaptive ROI γύρω από το tag."""

    def __init__(self, tag_id, corners):
        self.tag_id = tag_id
        self.lost_frames = 0
        self.update(corners)

    def update(self, corners):
        self.corners = corners.copy()
        self.lost_frames = 0
        xs = corners[:, 0]
        ys = corners[:, 1]
        self.x_min, self.x_max = np.min(xs), np.max(xs)
        self.y_min, self.y_max = np.min(ys), np.max(ys)

    def get_roi_crop(self, frame_w, frame_h, margin_factor=2.0, min_size=80):
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0

        w = max(self.x_max - self.x_min, min_size) * margin_factor
        h = max(self.y_max - self.y_min, min_size) * margin_factor

        x1 = max(0, int(cx - w / 2.0))
        y1 = max(0, int(cy - h / 2.0))
        x2 = min(frame_w, int(cx + w / 2.0))
        y2 = min(frame_h, int(cy + h / 2.0))
        return x1, y1, x2, y2


# ==============================================================================
# 6. ROS 2 NODE CLASS
# ==============================================================================
class BlueROVLocalizationNode(Node):

    def __init__(self):
        super().__init__("localization_node")

        self.declare_parameter("udp_port", 5600)
        self.declare_parameter("world_frame_id", "world")
        self.port = self.get_parameter("udp_port").get_parameter_value().integer_value
        self.world_frame = (
            self.get_parameter("world_frame_id").get_parameter_value().string_value
        )

        self.pose_pub = self.create_publisher(PoseStamped, "rov/pose", 10)
        self.human_pub = self.create_publisher(String, "rov/pose_human_readable", 10)

        self.detector = apriltag.Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0, # 1.0 = full resolution (increase for speed at cost of accuracy)
            quad_sigma=0.0, # disable Gaussian blur for sharper edges
            refine_edges=1, # disable edge refinement for speed
        )

        # ROI Tracking παραμέτροι
        self.tracked_tags = {}
        self.frame_count = 0
        self.FULL_SCAN_INTERVAL = 30  # Πλήρες scan ανά 30 frames (ή αν δεν υπάρχουν tags)
        self.MAX_LOST_FRAMES = 5      # Διαγραφή από το tracking μετά από 5 χαμένα frames

        self.get_logger().info(f"Connecting to UDP stream on port {self.port}...")
        self.video = Video(port=self.port)

        self.is_running = True
        self.worker_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker_thread.start()
        self.get_logger().info("Localization worker thread started.")

    def processing_loop(self):
        """Εκτελεί το pipeline μόλις φτάσει νέο frame από το GStreamer."""
        first_frame = True
        while rclpy.ok() and self.is_running:
            frame, recv_time = self.video.get_latest_frame_blocking()
            if frame is None:
                continue

            if first_frame:
                self.get_logger().info("Video stream connected! Processing started.")
                first_frame = False

            self.process_pipeline(frame, recv_time)

    def process_pipeline(self, frame, recv_time):
        self.frame_count += 1
        gray = frame
        detected_tags_in_frame = {}

        # 1. APRILTAG DETECTION (ADAPTIVE ROI TRACKING)
        periodic_full_scan = (self.frame_count % self.FULL_SCAN_INTERVAL == 0)
        run_full_scan = (len(self.tracked_tags) == 0) or periodic_full_scan

        if run_full_scan:
            results = self.detector.detect(gray)
            for r in results:
                detected_tags_in_frame[r.tag_id] = r.corners
        else:
            for tag_id, track in list(self.tracked_tags.items()):
                x1, y1, x2, y2 = track.get_roi_crop(gray.shape[1], gray.shape[0])
                roi_gray = gray[y1:y2, x1:x2]

                roi_results = self.detector.detect(roi_gray)
                matched = False
                for r in roi_results:
                    global_corners = r.corners + np.array([x1, y1])
                    if r.tag_id == tag_id:
                        detected_tags_in_frame[tag_id] = global_corners
                        matched = True
                    elif r.tag_id not in detected_tags_in_frame:
                        # Pick up new tags discovered inside ROI crops
                        detected_tags_in_frame[r.tag_id] = global_corners

                if not matched:
                    track.lost_frames += 1

            # Fallback: if ROI scans found nothing, do a full scan immediately
            if not detected_tags_in_frame:
                results = self.detector.detect(gray)
                for r in results:
                    detected_tags_in_frame[r.tag_id] = r.corners

        # Ενημέρωση/Δημιουργία trackers
        for tag_id, corners in detected_tags_in_frame.items():
            if tag_id in self.tracked_tags:
                self.tracked_tags[tag_id].update(corners)
            else:
                self.tracked_tags[tag_id] = TrackedTag(tag_id, corners)

        # Καθαρισμός tags που χάθηκαν
        tags_to_remove = [
            tag_id
            for tag_id, track in self.tracked_tags.items()
            if track.lost_frames > self.MAX_LOST_FRAMES
        ]
        for tag_id in tags_to_remove:
            del self.tracked_tags[tag_id]

        # 2. SOLVE PNP & PUBLISH
        all_obj_pts = []
        all_img_pts = []
        detected_known_ids = []

        for tag_id, corners in detected_tags_in_frame.items():
            if tag_id in GLOBAL_TAG_CORNERS:
                all_obj_pts.append(GLOBAL_TAG_CORNERS[tag_id])
                all_img_pts.append(corners)
                detected_known_ids.append(tag_id)

        if len(detected_known_ids) > 0:
            obj_pts = np.vstack(all_obj_pts).astype(np.float64)
            img_pts = np.vstack(all_img_pts).astype(np.float64)

            if len(detected_known_ids) >= 2:
                success, rvec, tvec, _ = cv2.solvePnPRansac(
                    obj_pts,
                    img_pts,
                    K,
                    D,
                    reprojectionError=30.0,  # how sensitive RANSAC is to outliers
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_SQPNP
                )

            if success:
                R_cw, _ = cv2.Rodrigues(rvec)
                t_cw = tvec.reshape(3, 1)

                R_wc = R_cw.T
                t_wc = -R_wc @ t_cw
                t_wb = t_wc + R_wc @ CAMERA_TO_BODY_OFFSET

                rov_x, rov_y, rov_z = t_wb.flatten()
                rov_roll, rov_pitch, rov_yaw = rotation_matrix_to_rpy(R_wc)
                qx, qy, qz, qw = rotation_matrix_to_quaternion(R_wc)

                # Χρήση του χρόνου άφιξης του πακέτου στο GStreamer για ακριβές ROS Timestamp
                pose_msg = PoseStamped()
                pose_msg.header.stamp.sec = int(recv_time)
                pose_msg.header.stamp.nanosec = int((recv_time - int(recv_time)) * 1e9)
                pose_msg.header.frame_id = self.world_frame
                pose_msg.pose.position.x = float(rov_x)
                pose_msg.pose.position.y = float(rov_y)
                pose_msg.pose.position.z = float(rov_z)
                pose_msg.pose.orientation.x = float(qx)
                pose_msg.pose.orientation.y = float(qy)
                pose_msg.pose.orientation.z = float(qz)
                pose_msg.pose.orientation.w = float(qw)
                self.pose_pub.publish(pose_msg)

                human_msg = String()
                human_msg.data = (
                    f"Position [m]: X={rov_x:+.3f}, Y={rov_y:+.3f}, Z={rov_z:+.3f} | "
                    f"Orientation [deg]: Roll={rov_roll:+.1f}, Pitch={rov_pitch:+.1f}, Yaw={rov_yaw:+.1f}"
                )
                self.human_pub.publish(human_msg)

    def destroy_node(self):
        self.is_running = False
        self.video.stop()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
        super().destroy_node()


# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = BlueROVLocalizationNode()

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