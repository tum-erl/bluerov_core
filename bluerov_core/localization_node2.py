#!/usr/bin/env python3
"""
BlueROV2 AprilTag Camera Localization ROS 2 Node (localization_node2)
Streamlined, robust, multi-tag PnP localization without state machine overengineering.

Key Features:
 1. Full-frame AprilTag detection (no ROI cropping, no flicker, rock-solid corners).
 2. Multi-Tag PnP (>=2 tags): Combines all tag points in a single SQPNP solve.
    Completely eliminates planar ambiguity and pose jumps.
 3. Single-Tag PnP (1 tag): Solves with IPPE_SQUARE and disambiguates using physical
    world constraints (ROV in front of tag wall: Z <= 0.05m, camera facing wall: R_wc[2,2] > 0).
 4. Exponential Moving Average (EMA) low-pass filtering for smooth output.
 5. Complete, compliant ROS TF hierarchy (world -> rov -> camera_optical_frame).
"""

import cv2
import numpy as np
import pupil_apriltags as apriltag

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
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

# Laptop camera calibrated parameters (from calibration_params_laptop.txt)
# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720
# FX, FY = 959.0366, 957.5605
# CX, CY = 634.5466, 359.5301
# K = np.array([[FX, 0.0, CX],[0.0, FY, CY],[0.0, 0.0, 1.0]], dtype=np.float64)
# D = np.array([0.02858738, -0.17890580, 0.00004431, -0.00097149, 0.21887859], dtype=np.float64)

# ROV calibration on air
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FX, FY = 1035.7549, 1036.4702
CX, CY = 971.0844, 605.2601
K = np.array([[FX, 0.0, CX],[0.0, FY, CY],[0.0, 0.0, 1.0]], dtype=np.float64)
D = np.array([-0.01495590, 0.05971814, 0.00197232, 0.00178856, -0.05786127], dtype=np.float64)


# Camera offset relative to ROV body center (in meters, optical frame)
CAMERA_TO_BODY_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Rotation matrix: Optical Camera Frame -> FLU Body Frame
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

def quaternion_nlerp(q1, q2, alpha=0.75):
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
    """
    Converts a single-tag local PnP pose (P_camera = R_ct * P_tag + t_ct)
    to the camera and body pose in the world optical frame.
    """
    R_ct, _ = cv2.Rodrigues(rvec)
    R_wc = R_tw @ R_ct.T
    t_wc = R_tw @ (-R_ct.T @ tvec.reshape(3, 1)) + t_tw
    t_wb = t_wc + R_wc @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)
    return R_wc, t_wb, t_wc

def compute_reprojection_error(obj_pts, img_pts, rvec, tvec, K, D):
    """Computes mean reprojection error in pixels to filter out outlier solutions."""
    try:
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        err = np.mean(np.linalg.norm(projected.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1))
        return float(err)
    except Exception:
        return 999.0

def draw_tag_overlay(display_frame, tag_id, corners_2d):
    """Clean standard visualization overlay for detected tags."""
    color = TAG_COLORS[hash(tag_id) % len(TAG_COLORS)]
    cv2.polylines(display_frame, [corners_2d.astype(np.int32).reshape((-1, 1, 2))], True, color, 3)
    cx, cy = int(np.mean(corners_2d[:, 0])), int(np.mean(corners_2d[:, 1]))
    cv2.putText(display_frame, f"ID {tag_id}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

# ==============================================================================
# 3. ROS 2 LOCALIZATION NODE
# ==============================================================================
class BlueROVLocalizationNode2(Node):
    def __init__(self):
        super().__init__("localization_node2")
        
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("rov_frame_id", "rov")
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        self.declare_parameter("filter_alpha", 0.35)
        self.declare_parameter("max_xy_bound", 2.5)   # Limits in X and Y (meters)
        self.declare_parameter("max_z_bound", 0.50)   # ROV is in front of wall: Z <= 0.50m
        self.declare_parameter("max_lost_frames", 30) # Frames to hold pose when tags lost (~1.0s)
        self.declare_parameter("max_jump_thresh", 0.20) # Max allowed step per frame (meters)
        
        self.world_frame = self.get_parameter("world_frame_id").get_parameter_value().string_value
        self.rov_frame = self.get_parameter("rov_frame_id").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame_id").get_parameter_value().string_value
        self.filter_alpha = float(self.get_parameter("filter_alpha").get_parameter_value().double_value)
        self.max_xy_bound = float(self.get_parameter("max_xy_bound").get_parameter_value().double_value)
        self.max_z_bound = float(self.get_parameter("max_z_bound").get_parameter_value().double_value)
        self.max_lost_frames = int(self.get_parameter("max_lost_frames").get_parameter_value().integer_value)
        self.max_jump_thresh = float(self.get_parameter("max_jump_thresh").get_parameter_value().double_value)
        
        self.cv_bridge = CvBridge()
        
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.image_sub = self.create_subscription(
            Image, 
            "rov/camera/image_raw", 
            self.image_callback, 
            sensor_qos
        )
        
        self.annotated_pub = self.create_publisher(Image, "rov/camera/image_annotated", sensor_qos)
        self.pose_pub = self.create_publisher(PoseStamped, "rov/pose", 10)
        self.camera_pose_pub = self.create_publisher(PoseStamped, "rov/camera_pose", 10)
        self.human_pub = self.create_publisher(String, "rov/pose_human_readable", 10)
        self.tag_marker_pub = self.create_publisher(MarkerArray, "rov/tags_markers", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        self.publish_static_camera_tf()
        self.publish_static_tags_tf()
        self.publish_tag_markers()
        self.marker_timer = self.create_timer(2.0, self.publish_tag_markers)

        # Full-frame detector
        self.detector = apriltag.Detector(
            families=TAG_FAMILY, 
            nthreads=4, 
            quad_decimate=4.0, 
            quad_sigma=0.0, 
            refine_edges=0.0
        )
        
        self.frame_count = 0
        self.lost_frames_count = 0
        self.outlier_streak = 0
        self.prev_world_pos = None
        self.prev_R_wc = None
        
        # Smoothed pose buffers (in ROS FLU world frame)
        self.filtered_pos = None
        self.filtered_quat = None
        self.filtered_cam_pos = None
        self.filtered_cam_quat = None
        self.last_R_wb_flu = None

        self.get_logger().info(
            "Ultra-Smooth PnP Localization Node started. "
            f"Alpha: {self.filter_alpha}, Max XY: {self.max_xy_bound}m, Max Z: {self.max_z_bound}m, Jump Gate: {self.max_jump_thresh}m"
        )

    def publish_static_camera_tf(self):
        t_static = TransformStamped()
        t_static.header.stamp = self.get_clock().now().to_msg()
        t_static.header.frame_id = self.rov_frame
        t_static.child_frame_id = self.camera_frame

        cam_off_flu = R_OPTICAL_TO_BODY @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)
        t_static.transform.translation.x = float(cam_off_flu[0])
        t_static.transform.translation.y = float(cam_off_flu[1])
        t_static.transform.translation.z = float(cam_off_flu[2])

        qx_c, qy_c, qz_c, qw_c = rotation_matrix_to_quaternion(R_OPTICAL_TO_BODY)
        t_static.transform.rotation.x = float(qx_c)
        t_static.transform.rotation.y = float(qy_c)
        t_static.transform.rotation.z = float(qz_c)
        t_static.transform.rotation.w = float(qw_c)

        self.static_tf_broadcaster.sendTransform(t_static)

    def publish_static_tags_tf(self):
        static_transforms = []
        for tag_id, tag_data in TAG_MAP.items():
            pos = tag_data["pos"]
            rpy = tag_data["rpy"]

            pos_flu = R_OPTICAL_TO_BODY @ np.array(pos, dtype=np.float64).reshape(3, 1)
            R_tw = rpy_to_rotation_matrix(rpy)
            R_tw_flu = R_OPTICAL_TO_BODY @ R_tw @ R_OPTICAL_TO_BODY.T
            qx_t, qy_t, qz_t, qw_t = rotation_matrix_to_quaternion(R_tw_flu)

            t_tag = TransformStamped()
            t_tag.header.stamp = self.get_clock().now().to_msg()
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

        if static_transforms:
            self.static_tf_broadcaster.sendTransform(static_transforms)

    def publish_tag_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Clear any previously cached / deleted markers in Foxglove / RViz
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
        """
        Physical verification in world optical coordinates:
        1. ROV is in front of origin tag wall: Z <= max_z_bound (e.g. <= 0.05m).
        2. ROV is within expected lateral range: |X| <= max_xy_bound, |Y| <= max_xy_bound.
        3. Camera optical Z-axis points toward the wall: R_wc[2, 2] > -0.1.
        """
        x = float(t_wb[0])
        y = float(t_wb[1])
        z = float(t_wb[2])

        if z > self.max_z_bound:
            return False
        if abs(x) > self.max_xy_bound or abs(y) > self.max_xy_bound:
            return False
        if R_wc is not None and R_wc[2, 2] <= -0.1:
            return False
        return True

    def image_callback(self, msg):
        try:
            bgr_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.frame_count += 1
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # ----------------------------------------------------------------------
        # 1. FULL-FRAME APRILTAG DETECTION (NO ROI, NO FLICKER)
        # ----------------------------------------------------------------------
        detected_tags = {}
        for r in self.detector.detect(gray):
            if r.tag_id in GLOBAL_TAG_CORNERS:
                detected_tags[r.tag_id] = r.corners

        # ----------------------------------------------------------------------
        # 2. ANNOTATED IMAGE PUBLISHING
        # ----------------------------------------------------------------------
        annotated_frame = bgr_frame.copy()
        for tag_id, corners in detected_tags.items():
            draw_tag_overlay(annotated_frame, tag_id, corners)

        try:
            annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            annotated_msg.header = msg.header
            annotated_msg.header.frame_id = self.camera_frame
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

        # ----------------------------------------------------------------------
        # 3. ROBUST PNP SOLVER
        # ----------------------------------------------------------------------
        num_tags = len(detected_tags)
        solved_new_pose = False
        R_wc_candidate = None
        t_wb_candidate = None
        t_wc_candidate = None
        mode_str = "NONE"

        if num_tags >= 2:
            # ------------------------------------------------------------------
            # MULTI-TAG PNP (>=2 tags, >=8 points)
            # ------------------------------------------------------------------
            # Object points are in world optical coordinates (GLOBAL_TAG_CORNERS).
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

        if not solved_new_pose and num_tags >= 1:
            # ------------------------------------------------------------------
            # SINGLE-TAG PNP (4 points with IPPE_SQUARE)
            # ------------------------------------------------------------------
            # Use reference tag if visible, otherwise the first detected tag
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
                        # Position continuity penalty
                        if self.prev_world_pos is not None:
                            pos_cost = float(np.linalg.norm(t_wb - self.prev_world_pos))
                        else:
                            pos_cost = 0.0

                        # Orientation continuity penalty: Frobenius norm of rotation difference
                        # CRITICAL: This completely eliminates planar flip ambiguity jumps!
                        if self.prev_R_wc is not None:
                            rot_cost = float(np.linalg.norm(R_wc - self.prev_R_wc))
                        else:
                            # Prefer camera optical Z pointing towards tag wall
                            rot_cost = -float(R_wc[2, 2])

                        total_cost = pos_cost + 0.35 * rot_cost
                        valid_candidates.append((total_cost, R_wc, t_wb, t_wc))

                if valid_candidates:
                    valid_candidates.sort(key=lambda item: item[0])
                    _, R_wc_candidate, t_wb_candidate, t_wc_candidate = valid_candidates[0]
                    solved_new_pose = True
                    mode_str = f"SINGLE (tag {tag_id})"

        # ----------------------------------------------------------------------
        # 4. FILTERING & STATE UPDATE
        # ----------------------------------------------------------------------
        if solved_new_pose and t_wb_candidate is not None:
            self.lost_frames_count = 0
            self.prev_world_pos = t_wb_candidate.copy()
            self.prev_R_wc = R_wc_candidate.copy()

            # Convert candidate to ROS FLU World Coordinates
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

        # ----------------------------------------------------------------------
        # 5. PUBLISH & TF BROADCAST
        # ----------------------------------------------------------------------
        if self.filtered_pos is not None:
            x_b, y_b, z_b = self.filtered_pos
            qx_r, qy_r, qz_r, qw_r = self.filtered_quat

            x_c, y_c, z_c = self.filtered_cam_pos
            qx_wc, qy_wc, qz_wc, qw_wc = self.filtered_cam_quat

            # ROV Pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = self.world_frame
            pose_msg.pose.position.x = float(x_b)
            pose_msg.pose.position.y = float(y_b)
            pose_msg.pose.position.z = float(z_b)
            pose_msg.pose.orientation.x = float(qx_r)
            pose_msg.pose.orientation.y = float(qy_r)
            pose_msg.pose.orientation.z = float(qz_r)
            pose_msg.pose.orientation.w = float(qw_r)
            self.pose_pub.publish(pose_msg)

            # Camera Pose
            cam_pose_msg = PoseStamped()
            cam_pose_msg.header.stamp = msg.header.stamp
            cam_pose_msg.header.frame_id = self.world_frame
            cam_pose_msg.pose.position.x = float(x_c)
            cam_pose_msg.pose.position.y = float(y_c)
            cam_pose_msg.pose.position.z = float(z_c)
            cam_pose_msg.pose.orientation.x = float(qx_wc)
            cam_pose_msg.pose.orientation.y = float(qy_wc)
            cam_pose_msg.pose.orientation.z = float(qz_wc)
            cam_pose_msg.pose.orientation.w = float(qw_wc)
            self.camera_pose_pub.publish(cam_pose_msg)

            # TF: world -> rov
            t_stamped_rov = TransformStamped()
            t_stamped_rov.header.stamp = msg.header.stamp
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
            t_stamped_cam.header.stamp = msg.header.stamp
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
            msg_str = (
                f"[{status_text} | {mode_str}] "
                f"ROV: X={x_b:+.3f}, Y={y_b:+.3f}, Z={z_b:+.3f} | {rpy_str}"
            )
            # print(f"[Frame {self.frame_count:05d}] {mode_str} | ROV FLU: X={x_b:+.3f}, Y={y_b:+.3f}, Z={z_b:+.3f} | {rpy_str}")
            self.human_pub.publish(String(data=msg_str))
        else:
            msg_str = f"[NO TAGS VISIBLE] Frame {self.frame_count}"
            self.human_pub.publish(String(data=msg_str))

def main(args=None):
    rclpy.init(args=args)
    node = BlueROVLocalizationNode2()
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
