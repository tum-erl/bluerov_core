#!/usr/bin/env python3
"""
BlueROV2 AprilTag Camera Localization ROS 2 Node (Jazzy)
Node: rov_localization_node
Features:
- Event-Driven Processing with dedicated worker thread (Condition variable)
- Low-latency direct GRAY8 GStreamer pipeline (avdec_h264 for ARM/RPi CPU)
- GStreamer H.264 parse-to-sample decode probe profiling
- Dictionary-based AprilTag deduplication for robust PnP solving
- Modular Pipeline Profiler with execution metrics & horizontal summary chart on exit
- Clean thread join & shutdown cleanup
Topics Published:
- /rov/pose (geometry_msgs/msg/PoseStamped)
- /rov/pose_human_readable (std_msgs/msg/String)
"""

import threading
import time
from collections import defaultdict
import cv2
import gi
import matplotlib.pyplot as plt
import numpy as np
import pupil_apriltags as apriltag

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


# ==============================================================================
# 0. MODULAR MULTI-THREAD PIPELINE PROFILER
# ==============================================================================
class PipelineProfiler:
    """Καταγράφει χρόνους εκτέλεσης ανά στάδιο και παράγει δομημένο γράφημα

    διαχωρισμένο ανά Hardware Thread (Stream Rate, Thread A, Thread B).
    """

    def __init__(self):
        self.stage_times = defaultdict(list)
        self.lock = threading.Lock()

    def add_timing(self, stage_name, duration_ms):
        with self.lock:
            self.stage_times[stage_name].append(duration_ms)

    def time_block(self, stage_name):
        class _TimerContext:

            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
                with self.profiler.lock:
                    self.profiler.stage_times[self.name].append(elapsed_ms)

        return _TimerContext(self, stage_name)

    def plot_summary(self):
        with self.lock:
            if not self.stage_times:
                print("[Profiler] Δεν καταγράφηκαν δεδομένα χρονισμού.")
                return
            stage_data = {k: list(v) for k, v in self.stage_times.items()}

        # 1. Διαχωρισμός σταδίων ανά αρχιτεκτονικό ρόλο / Thread
        stream_stages = ["0_inter_frame_interval"]

        thread_a_stages = [
            s
            for s in ["0_gst_h264_decode", "0_gst_buffer_to_numpy"]
            if s in stage_data
        ]

        thread_b_stages = [
            s
            for s in [
                "1_apriltag_detect",
                "2_solve_pnp_and_publish",
                "3_gui_events",
            ]
            if s in stage_data
        ]

        # Υπολογισμοί μετρικών
        avg_interval_ms = (
            np.mean(stage_data["0_inter_frame_interval"])
            if "0_inter_frame_interval" in stage_data
            else 0.0
        )
        stream_fps = (
            1000.0 / avg_interval_ms
            if avg_interval_ms > 0
            else 0.0
        )

        thread_a_total_ms = sum(
            np.mean(stage_data[s]) for s in thread_a_stages
        )
        thread_b_total_ms = sum(
            np.mean(stage_data[s]) for s in thread_b_stages
        )
        worker_max_hz = (
            1000.0 / thread_b_total_ms
            if thread_b_total_ms > 0
            else 0.0
        )

        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11, 8.5),
            gridspec_kw={"height_ratios": [1, 1.3, 1.8]},
        )

        # ----------------------------------------------------------------------
        # PANEL 1: Stream Delivery Rate (Callback Triggers)
        # ----------------------------------------------------------------------
        ax0 = axes[0]
        ax0.barh(
            [0],
            [avg_interval_ms],
            color="#2980b9",
            edgecolor="black",
            height=0.45,
            alpha=0.85,
        )
        ax0.set_yticks([0])
        ax0.set_yticklabels(
            ["Inter-Frame Arrival"], fontsize=10, fontweight="bold"
        )
        ax0.set_xlabel("Χρόνος ανάμεσα σε διαδοχικά callbacks (ms)", fontsize=9.5)
        ax0.set_title(
            f"1. Inbound Video Stream (Συχνότητα εκτέλεσης Callback)\n"
            f"Μέσος ρυθμός: {avg_interval_ms:.2f} ms ανά frame  |  Πραγματικός Ρυθμός Κάμερας: ~{stream_fps:.1f} FPS",
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=8,
        )
        ax0.set_xlim(0, max(avg_interval_ms * 1.35, 40))
        ax0.text(
            avg_interval_ms + 1.0,
            0,
            f"{avg_interval_ms:.2f} ms (~{stream_fps:.1f} FPS)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1b4f72",
        )

        # ----------------------------------------------------------------------
        # PANEL 2: Thread A (GStreamer C-Thread & Decode Pipeline)
        # ----------------------------------------------------------------------
        ax1 = axes[1]
        y_pos_a = np.arange(len(thread_a_stages))
        avg_a = [np.mean(stage_data[s]) for s in thread_a_stages]
        colors_a = ["#e67e22", "#d35400"]

        bars_a = ax1.barh(
            y_pos_a,
            avg_a,
            color=colors_a[: len(thread_a_stages)],
            edgecolor="black",
            height=0.5,
            alpha=0.85,
        )
        ax1.set_yticks(y_pos_a)
        ax1.set_yticklabels(thread_a_stages, fontsize=10, fontweight="bold")
        ax1.invert_yaxis()
        ax1.set_xlabel("Χρόνος εκτέλεσης στη C (ms)", fontsize=9.5)
        ax1.set_title(
            f"2. Thread A: Ingestion & Decode (GStreamer C-Thread)\n"
            f"Συνολικό Overhead στο Callback: {thread_a_total_ms:.2f} ms (Παράλληλο με το Worker Thread)",
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=8,
        )
        max_a = max(avg_a) if avg_a else 10.0
        ax1.set_xlim(0, max_a * 1.35)

        for bar, avg_t in zip(bars_a, avg_a):
            ax1.text(
                bar.get_width() + (max_a * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{avg_t:.2f} ms",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
            )

        # ----------------------------------------------------------------------
        # PANEL 3: Thread B (Python Worker Thread - Vision & Pose)
        # ----------------------------------------------------------------------
        ax2 = axes[2]
        y_pos_b = np.arange(len(thread_b_stages))
        avg_b = [np.mean(stage_data[s]) for s in thread_b_stages]
        colors_b = ["#16a085", "#c0392b", "#27ae60", "#8e44ad"]

        bars_b = ax2.barh(
            y_pos_b,
            avg_b,
            color=colors_b[: len(thread_b_stages)],
            edgecolor="black",
            height=0.5,
            alpha=0.85,
        )
        ax2.set_yticks(y_pos_b)
        ax2.set_yticklabels(thread_b_stages, fontsize=10, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_xlabel("Χρόνος εκτέλεσης στην Python (ms)", fontsize=9.5)
        ax2.set_title(
            f"3. Thread B: Vision Pipeline (Dedicated Worker Thread)\n"
            f"Καθαρός Χρόνος Επεξεργασίας Worker: {thread_b_total_ms:.2f} ms  |  Μέγιστη Ικανότητα Worker: {worker_max_hz:.1f} Hz",
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=8,
        )
        max_b = max(avg_b) if avg_b else 10.0
        ax2.set_xlim(0, max_b * 1.35)

        for bar, avg_t in zip(bars_b, avg_b):
            pct = (
                (avg_t / thread_b_total_ms) * 100.0
                if thread_b_total_ms > 0
                else 0
            )
            ax2.text(
                bar.get_width() + (max_b * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{avg_t:.2f} ms ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
            )

        plt.tight_layout()

        # Terminal Text Report
        print("\n" + "=" * 65)
        print(" ARCHITECTURAL PROFILING REPORT (MULTI-THREAD BREAKDOWN)")
        print("=" * 65)
        print(f" [STREAM ARRIVAL]")
        print(
            f"  * 0_inter_frame_interval     : {avg_interval_ms:6.2f} ms (Camera Rate: ~{stream_fps:.1f} FPS)"
        )
        print(f"\n [THREAD A: GStreamer & Ingestion (C-Level)]")
        for s in thread_a_stages:
            print(f"  - {s:<27}: {np.mean(stage_data[s]):6.2f} ms")
        print(f"  > Total Ingestion Latency    : {thread_a_total_ms:6.2f} ms")
        print(f"\n [THREAD B: Vision & PnP Solver (Worker Thread)]")
        for s in thread_b_stages:
            avg_t = np.mean(stage_data[s])
            pct = (avg_t / thread_b_total_ms) * 100.0 if thread_b_total_ms > 0 else 0
            print(f"  - {s:<27}: {avg_t:6.2f} ms ({pct:5.1f}%)")
        print(f"  > Total Worker Processing    : {thread_b_total_ms:6.2f} ms")
        print("-" * 65)
        print(f" Real Inbound Camera Rate      : {stream_fps:6.1f} FPS")
        print(f" Worker Max Throughput Limit   : {worker_max_hz:6.1f} Hz")
        print("=" * 65 + "\n")

        plt.show()

# ==============================================================================
# 1. GLOBAL CONFIGURATION & TAG MAP
# ==============================================================================
REFERENCE_TAG_ID = 0  # Defines origin of coordinate system (0,0,0)
TAG_SIZE_METERS = 0.06  # Tag black square side length in meters
TAG_FAMILY = "tagStandard41h12"

TAG_MAP = {
    0: {
        "pos": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
    },
    1: {
        "pos": [0.21, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
    },
    2: {
        "pos": [0.0, 0.40, 0.0],
        "rpy": [0.0, 0.0, 0.0],
    },
}

# ==============================================================================
# 2. CAMERA INTRINSICS & RIGID OFFSETS
# ==============================================================================
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

FX = 1200.0
FY = 1200.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0

K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.zeros(5, dtype=np.float64)

CAMERA_TO_BODY_OFFSET = np.array([[0.0], [0.0], [-0.00]])


# ==============================================================================
# 3. BLUEROV2 VIDEO CAPTURE CLASS (DIRECT GRAY8 & RPI OPTIMIZED)
# ==============================================================================
class Video:
    """BlueROV video capture class με βελτιστοποιημένο GRAY8 GStreamer pipeline και profiling."""

    def __init__(self, port=5600, profiler=None):
        Gst.init(None)

        self.port = port
        self.profiler = profiler

        self.latest_frame = None
        self.running = True
        self.condition = threading.Condition()

        self.last_frame_arrival_time = None
        self.current_h264_packet_time = None

        # Βελτιστοποιημένα στάδια για Raspberry Pi CPU:
        # - rtpjitterbuffer latency=0 για μηδενικό buffering καθυστέρησης
        # - avdec_h264 με NEON acceleration αντί του openh264dec
        # - format=GRAY8 για απευθείας απομόνωση του Y plane χωρίς υπολογιστικό κόστος
        self.video_source = (
            f"udpsrc port={self.port} "
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            f"rtpjitterbuffer latency=0 drop-on-latency=true"
        )
        self.video_codec = "! rtph264depay ! h264parse name=my_parser ! avdec_h264 max-threads=2"
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

        # Probe στο src pad του h264parse για μέτρηση decode latency
        parser = self.video_pipe.get_by_name("my_parser")
        if parser and self.profiler:
            src_pad = parser.get_static_pad("src")
            if src_pad:
                src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_h264_buffer_in)

        self.video_pipe.set_state(Gst.State.PLAYING)

    def _on_h264_buffer_in(self, pad, info):
        self.current_h264_packet_time = time.perf_counter()
        return Gst.PadProbeReturn.OK

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
            # Δημιουργία 2D πίνακα (H, W) και άμεση αντιγραφή ώστε να αποδεσμευτεί το GstBuffer
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
        recv_time = time.perf_counter()

        if self.profiler:
            if self.last_frame_arrival_time is not None:
                inter_frame_ms = (recv_time - self.last_frame_arrival_time) * 1000.0
                self.profiler.add_timing("0_inter_frame_interval", inter_frame_ms)
            self.last_frame_arrival_time = recv_time

            if self.current_h264_packet_time is not None:
                decode_ms = (recv_time - self.current_h264_packet_time) * 1000.0
                self.profiler.add_timing("0_gst_h264_decode", decode_ms)

        sample = sink.emit("pull-sample")
        if sample:
            if self.profiler:
                t0 = time.perf_counter()
                frame = self.gst_to_opencv(sample)
                t_buf_ms = (time.perf_counter() - t0) * 1000.0
                self.profiler.add_timing("0_gst_buffer_to_numpy", t_buf_ms)
            else:
                frame = self.gst_to_opencv(sample)

            with self.condition:
                self.latest_frame = frame
                self.condition.notify()

        return Gst.FlowReturn.OK

    def get_latest_frame_blocking(self):
        """Μπλοκάρει αποδοτικά μέχρι να φτάσει νέο frame από το GStreamer."""
        with self.condition:
            while self.latest_frame is None and self.running:
                self.condition.wait(timeout=0.1)

            if not self.running:
                return None

            frame = self.latest_frame
            self.latest_frame = None  # Consume frame
            return frame

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


def build_global_tag_corners(tag_map, tag_size):
    s = tag_size / 2.0
    local_corners = np.array(
        [
            [-s, s, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=np.float32,
    )

    global_corners_map = {}
    for tag_id, transform in tag_map.items():
        R = rpy_to_rotation_matrix(transform["rpy"])
        t = np.array(transform["pos"], dtype=np.float32).reshape(3, 1)

        corners_global = (R @ local_corners.T + t).T
        global_corners_map[tag_id] = corners_global

    return global_corners_map


GLOBAL_TAG_CORNERS = build_global_tag_corners(TAG_MAP, TAG_SIZE_METERS)


# ==============================================================================
# 5. ROS 2 NODE CLASS
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

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "rov/pose", 10)
        self.human_pub = self.create_publisher(String, "rov/pose_human_readable", 10)

        # Profiler instance
        self.profiler = PipelineProfiler()

        # Detector configuration
        self.detector = apriltag.Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=2.0, # Downsample factor for faster detection (1.0 = no downsampling)
            quad_sigma=0.0, # Gaussian blur applied to the image (0.0 = no blur)
            refine_edges=0, # Whether to refine the edges of detected tags (0 = no refinement, 1 = refine)
        )

        self.get_logger().info(f"Connecting to UDP stream on port {self.port}...")
        self.video = Video(port=self.port, profiler=self.profiler)

        # Worker thread
        self.is_running = True
        self.worker_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker_thread.start()
        self.get_logger().info("Localization worker thread started.")

    def processing_loop(self):
        """Εκτελεί το pipeline μόλις φτάσει νέο frame από το GStreamer."""
        first_frame = True
        while rclpy.ok() and self.is_running:
            frame = self.video.get_latest_frame_blocking()
            if frame is None:
                continue

            if first_frame:
                self.get_logger().info("Video stream connected! Processing started.")
                first_frame = False

            self.process_pipeline(frame)

    def process_pipeline(self, frame):
        
        gray = frame

        # 1. APRILTAG DETECT
        with self.profiler.time_block("1_apriltag_detect"):
            results = self.detector.detect(gray)
            detected_tags_in_frame = {}
            for r in results:
                detected_tags_in_frame[r.tag_id] = r.corners

        # 3. SOLVE PNP & PUBLISH
        with self.profiler.time_block("2_solve_pnp_and_publish"):
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
                        reprojectionError=8.0,
                        flags=cv2.SOLVEPNP_SQPNP,
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
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

                    stamp = self.get_clock().now().to_msg()

                    # Publisher 1: PoseStamped (Quaternions)
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = stamp
                    pose_msg.header.frame_id = self.world_frame
                    pose_msg.pose.position.x = float(rov_x)
                    pose_msg.pose.position.y = float(rov_y)
                    pose_msg.pose.position.z = float(rov_z)
                    pose_msg.pose.orientation.x = float(qx)
                    pose_msg.pose.orientation.y = float(qy)
                    pose_msg.pose.orientation.z = float(qz)
                    pose_msg.pose.orientation.w = float(qw)
                    self.pose_pub.publish(pose_msg)

                    # Publisher 2: Human-readable String format
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
        cv2.destroyAllWindows()
        super().destroy_node()


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = BlueROVLocalizationNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        profiler = node.profiler
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        profiler.plot_summary()


if __name__ == "__main__":
    main()