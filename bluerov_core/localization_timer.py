#!/usr/bin/env python3
"""
BlueROV2 AprilTag Camera Localization ROS 2 Node (Jazzy)
Node: rov_localization_node
Features:
- Pure Event-Driven Processing with GStreamer Decode Profiling
- FULL FRAME SCAN on EVERY step (No ROI logic)
- Execution Benchmarking & Summary Profiling on exit ('q')
Topics Published:
- /rov/pose (geometry_msgs/msg/PoseStamped)
- /rov/pose_human_readable (std_msgs/msg/String)
"""

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
# 0. PROFILER CLASS
# ==============================================================================
class PipelineProfiler:
    """Καταγράφει χρόνους εκτέλεσης ανά block και παράγει γράφημα μέσων όρων."""

    def __init__(self):
        self.stage_times = defaultdict(list)

    def add_timing(self, stage_name, duration_ms):
        self.stage_times[stage_name].append(duration_ms)

    def time_block(self, stage_name):
        class _TimerContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
                self.profiler.stage_times[self.name].append(elapsed_ms)

        return _TimerContext(self, stage_name)

    def plot_summary(self):
        if not self.stage_times:
            print("[Profiler] Δεν καταγράφηκαν δεδομένα.")
            return

        stages = list(self.stage_times.keys())
        avg_times = [np.mean(self.stage_times[s]) for s in stages]

        compute_stages = [s for s in stages if "inter_frame" not in s]
        compute_avg_ms = sum(np.mean(self.stage_times[s]) for s in compute_stages)
        max_possible_fps = 1000.0 / compute_avg_ms if compute_avg_ms > 0 else 0.0

        if "0_inter_frame_interval" in self.stage_times:
            avg_interval_ms = np.mean(self.stage_times["0_inter_frame_interval"])
            actual_camera_fps = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0.0
        else:
            actual_camera_fps = max_possible_fps

        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )
        fig, ax = plt.subplots(figsize=(11, 6))

        colors = ["#c0392b", "#d35400", "#2980b9", "#8e44ad", "#16a085", "#27ae60", "#7f8c8d"]
        bar_colors = [colors[i % len(colors)] for i in range(len(stages))]

        y_positions = np.arange(len(stages))
        bars = ax.barh(
            y_positions,
            avg_times,
            color=bar_colors,
            edgecolor="black",
            alpha=0.85,
            height=0.55,
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(stages, fontsize=9.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Μέσος Χρόνος (ms)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Pipeline Profiling (FULL FRAME SCAN - No ROI)\n"
            f"Συνολικό Latency Επεξεργασίας: {compute_avg_ms:.2f} ms | Πραγματικό Stream: ~{actual_camera_fps:.1f} FPS",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )

        max_val = max(avg_times) if avg_times else 1.0
        ax.set_xlim(0, max_val * 1.28)

        for bar, avg_t, stage_name in zip(bars, avg_times, stages):
            if "inter_frame" in stage_name:
                label_txt = f"{avg_t:.2f} ms (~{actual_camera_fps:.1f} FPS stream)"
            else:
                pct = (avg_t / compute_avg_ms) * 100.0 if compute_avg_ms > 0 else 0
                label_txt = f"{avg_t:.2f} ms ({pct:.1f}%)"

            ax.text(
                bar.get_width() + (max_val * 0.015),
                bar.get_y() + bar.get_height() / 2,
                label_txt,
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
            )

        plt.tight_layout()
        print("\n" + "=" * 60)
        print(" PIPELINE & GSTREAMER DECODE PROFILING REPORT")
        print("=" * 60)
        for s, avg_t in zip(stages, avg_times):
            if "inter_frame" in s:
                print(f" * {s:<28}: {avg_t:6.2f} ms (Camera Frame Rate: ~{actual_camera_fps:.1f} FPS)")
            else:
                pct = (avg_t / compute_avg_ms) * 100.0 if compute_avg_ms > 0 else 0
                print(f" - {s:<28}: {avg_t:6.2f} ms ({pct:5.1f}%)")
        print("-" * 60)
        print(f" Total Per-Frame Latency     : {compute_avg_ms:6.2f} ms")
        print(f" Max Theoretical Compute Rate: {max_possible_fps:6.1f} Hz")
        print(f" Actual Inbound Camera Rate  : {actual_camera_fps:6.1f} FPS")
        print("=" * 60 + "\n")

        plt.show()


# ==============================================================================
# 1. GLOBAL CONFIGURATION & TAG MAP
# ==============================================================================
REFERENCE_TAG_ID = 0
TAG_SIZE_METERS = 0.06
TAG_FAMILY = "tagStandard41h12"

TAG_MAP = {
    0: {"pos": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    1: {"pos": [0.21, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    2: {"pos": [0.0, 0.40, 0.0], "rpy": [0.0, 0.0, 0.0]},
}

# ==============================================================================
# 2. BLUEROV2 CAMERA INTRINSICS
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
# 3. BLUEROV2 EVENT-DRIVEN GSTREAMER VIDEO CAPTURE (PROFILED)
# ==============================================================================
class EventDrivenVideo:
    """GStreamer capture με probes μέτρησης χρόνου H.264 αποκωδικοποίησης."""

    def __init__(self, port=5600, frame_callback=None, profiler=None):
        Gst.init(None)

        self.port = port
        self.frame_callback = frame_callback
        self.profiler = profiler

        self.last_frame_arrival_time = None
        self.current_h264_packet_time = None

        self.video_pipe = None
        self.video_sink = None
        self.run()

    def run(self):
        pipeline_desc = (
            f"udpsrc port={self.port} ! "
            f"application/x-rtp, payload=96 ! rtph264depay ! "
            f"h264parse name=my_parser ! "
            f"openh264dec ! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert ! "
            f"appsink name=my_sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.video_pipe = Gst.parse_launch(pipeline_desc)
        self.video_sink = self.video_pipe.get_by_name("my_sink")
        parser = self.video_pipe.get_by_name("my_parser")

        if parser:
            src_pad = parser.get_static_pad("src")
            if src_pad:
                src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_h264_buffer_in)

        self.video_sink.connect("new-sample", self.callback)
        self.video_pipe.set_state(Gst.State.PLAYING)

    def _on_h264_buffer_in(self, pad, info):
        self.current_h264_packet_time = time.perf_counter()
        return Gst.PadProbeReturn.OK

    @staticmethod
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        array = np.ndarray(
            (
                caps_structure.get_value("height"),
                caps_structure.get_value("width"),
                3,
            ),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8,
        )
        return array

    def callback(self, sink):
        recv_time = time.perf_counter()

        if self.last_frame_arrival_time is not None and self.profiler:
            inter_frame_ms = (recv_time - self.last_frame_arrival_time) * 1000.0
            self.profiler.add_timing("0_inter_frame_interval", inter_frame_ms)
        self.last_frame_arrival_time = recv_time

        if self.current_h264_packet_time is not None and self.profiler:
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

            if self.frame_callback:
                self.frame_callback(frame)

        return Gst.FlowReturn.OK

    def stop(self):
        if self.video_pipe:
            self.video_pipe.set_state(Gst.State.NULL)


# ==============================================================================
# 4. ROTATIONS & 3D CORNER GENERATION
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
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
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
# 5. ROS 2 NODE CLASS (FULL SCAN EVERY STEP)
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

        self.profiler = PipelineProfiler()

        # Detector ρυθμίσεις (σταθερές για σύγκριση)
        self.detector = apriltag.Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=0,
        )

        self.get_logger().info(f"Connecting to UDP stream on port {self.port}...")
        self.video = EventDrivenVideo(
            port=self.port,
            frame_callback=self.on_new_frame,
            profiler=self.profiler,
        )
        self.get_logger().info("Video receiver initialized (FULL FRAME SCAN mode).")

    def on_new_frame(self, frame):
        """Εκτελείται απευθείας από το GStreamer thread μόλις παραδοθεί νέο frame."""
        if not rclpy.ok() or frame is None:
            return

        # 1. PREPROCESS
        with self.profiler.time_block("1_preprocess"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. APRILTAG DETECT (FULL FRAME SCAN ΣΕ ΚΑΘΕ ΒΗΜΑ)
        with self.profiler.time_block("2_apriltag_detect"):
            detected_tags_in_frame = {}
            results = self.detector.detect(gray)
            for r in results:
                detected_tags_in_frame[r.tag_id] = r.corners

        # 3. SOLVE PNP & PUBLISH
        with self.profiler.time_block("3_solve_pnp_and_publish"):
            all_obj_pts = []
            all_img_pts = []
            detected_known_ids = []

            for tag_id, corners in detected_tags_in_frame.items():
                if tag_id in GLOBAL_TAG_CORNERS:
                    all_obj_pts.append(GLOBAL_TAG_CORNERS[tag_id])
                    all_img_pts.append(corners)
                    detected_known_ids.append(tag_id)

            success = False
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

                    human_msg = String()
                    human_msg.data = (
                        f"Position [m]: X={rov_x:+.3f}, Y={rov_y:+.3f}, Z={rov_z:+.3f} | "
                        f"Orientation [deg]: Roll={rov_roll:+.1f}, Pitch={rov_pitch:+.1f}, Yaw={rov_yaw:+.1f}"
                    )
                    self.human_pub.publish(human_msg)

                    status_text = (
                        f"Tags used ({len(detected_known_ids)}): {detected_known_ids}"
                    )
                    pos_text = (
                        f"ROV Pos (m)  : X={rov_x:+.2f}, Y={rov_y:+.2f}, Z={rov_z:+.2f}"
                    )
                    rot_text = (
                        f"ROV Ori (deg): R={rov_roll:+.0f}, P={rov_pitch:+.0f},"
                        f" Y={rov_yaw:+.0f}"
                    )

                    cv2.putText(
                        frame,
                        status_text,
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        pos_text,
                        (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        rot_text,
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "PnP Solver Failed",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No known tags in view",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("BlueROV2 Localization (Global Multi-Tag PnP)", frame)

        # 4. EVENT POLLING & EXIT HANDLING
        with self.profiler.time_block("4_gui_events"):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Exit key 'q' pressed. Shutting down...")
                rclpy.shutdown()

    def destroy_node(self):
        self.video.stop()
        cv2.destroyAllWindows()
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
        profiler = node.profiler
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        profiler.plot_summary()


if __name__ == "__main__":
    main()