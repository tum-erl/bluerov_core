#!/usr/bin/env python3
"""
BlueROV2 AprilTag Camera Localization ROS 2 Node — Corrected TF Tree
Subscribes to image_raw, processes AprilTags, publishes Pose, Annotated Image, 
and broadcasts correct TF hierarchy (world -> rov -> camera_optical_frame).
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
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# ==============================================================================
# 1. GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================
REFERENCE_TAG_ID = 2
TAG_FAMILY = "tagStandard41h12"
SMALL_TAG_SIZE = 0.05
BIG_TAG_SIZE = 0.126

TAG_MAP = {
    2: {"pos": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "size": BIG_TAG_SIZE},
    1: {"pos": [0.255, 0.285, 0.0], "rpy": [0.0, 0.0, 0.0], "size": BIG_TAG_SIZE},
    3: {"pos": [-0.30, -0.13, 0.0], "rpy": [0.0, 270.0, 0.0], "size": BIG_TAG_SIZE},

    1000: {"pos": [-0.28, -0.575, 0.0],"rpy": [0.0, 0.0, 0.0],"size": SMALL_TAG_SIZE},
    1001: {"pos": [0.254, -0.575, 0.0],"rpy": [0.0, 0.0, 0.0],"size": SMALL_TAG_SIZE},
    1002: {"pos": [-0.235, 0.2, 0.0],"rpy": [270.0, 180.0, 0.0],"size": SMALL_TAG_SIZE},
    1003: {"pos": [0.285, -0.098, 0.0],"rpy": [0.0, 90.0, 0.0],"size": SMALL_TAG_SIZE}
}

# Laptop camera parameters
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FX, FY = 960.2436, 959.9177
CX, CY = 638.2691, 365.7544
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.array([0.03387992, -0.23298166, -0.00091282, -0.00095538, 0.35325257], dtype=np.float64)

# Το Offset της κάμερας σε σχέση με το κέντρο του ROV (σε μέτρα)
CAMERA_TO_BODY_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Πίνακας Περιστροφής (Optical Camera Frame -> FLU Robot Frame)
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
# 2. MATH HELPERS
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

def build_tag_tables(tag_map):
    global_map, local_map, trans_map = {}, {}, {}
    for tag_id, transform in tag_map.items():
        s = transform["size"] / 2.0
        local_corners = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)
        R_tw = rpy_to_rotation_matrix(transform["rpy"])
        t_tw = np.array(transform["pos"], dtype=np.float64).reshape(3, 1)
        global_map[tag_id] = (R_tw @ local_corners.T + t_tw).T.astype(np.float64)
        local_map[tag_id] = local_corners
        trans_map[tag_id] = (R_tw, t_tw)
    return global_map, local_map, trans_map

GLOBAL_TAG_CORNERS, LOCAL_TAG_CORNERS, TAG_TRANSFORMS = build_tag_tables(TAG_MAP)

def ippe_solution_to_world_pos(rvec, tvec, R_tw, t_tw):
    R_ct, _ = cv2.Rodrigues(rvec)
    R_wc = R_tw @ R_ct.T
    t_wc = R_tw @ (-R_ct.T @ tvec.reshape(3, 1)) + t_tw
    
    t_wb = t_wc + R_wc @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)
    return R_wc, t_wb, t_wc

# ==============================================================================
# 3. ROI & DRAWING HELPERS
# ==============================================================================
class TrackedTag:
    def __init__(self, tag_id, corners):
        self.tag_id = tag_id
        self.lost_frames = 0
        self.update(corners)
    def update(self, corners):
        self.corners = corners.copy()
        self.lost_frames = 0
        xs, ys = corners[:, 0], corners[:, 1]
        self.x_min, self.x_max, self.y_min, self.y_max = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    def get_roi_crop(self, frame_w, frame_h, margin_factor=2.0, min_size=80):
        cx, cy = (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0
        w, h = max(self.x_max - self.x_min, min_size) * margin_factor, max(self.y_max - self.y_min, min_size) * margin_factor
        return (max(0, int(cx - w / 2.0)), max(0, int(cy - h / 2.0)), 
                min(frame_w, int(cx + w / 2.0)), min(frame_h, int(cy + h / 2.0)))

def draw_tag_overlay(display_frame, tag_id, corners_2d):
    color = TAG_COLORS[hash(tag_id) % len(TAG_COLORS)]  # Safe hashing αντί για απλό modulo
    cv2.polylines(display_frame, [corners_2d.astype(np.int32).reshape((-1, 1, 2))], True, color, 3)
    cx, cy = int(np.mean(corners_2d[:, 0])), int(np.mean(corners_2d[:, 1]))
    cv2.putText(display_frame, f"ID {tag_id}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

# ==============================================================================
# 4. ROS 2 HEADLESS SUBSCRIBER NODE WITH FIXED TF HIERARCHY
# ==============================================================================
class BlueROVLocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("rov_frame_id", "rov")
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        
        self.world_frame = self.get_parameter("world_frame_id").get_parameter_value().string_value
        self.rov_frame = self.get_parameter("rov_frame_id").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame_id").get_parameter_value().string_value

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
        self.static_tf_broadcaster = TransformBroadcaster(self) # ή StaticTransformBroadcaster
        
        self.publish_static_camera_tf()
        self.publish_static_tags_tf()
        
        # Δημοσίευση των markers μία φορά στην αρχή (ή μέσω timer αν θες να ανανεώνονται)
        self.publish_tag_markers()
        self.marker_timer = self.create_timer(2.0, self.publish_tag_markers) # Refresh κάθε 2 δευτερόλεπτα

        self.detector = apriltag.Detector(families=TAG_FAMILY, nthreads=4, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1)
        self.tracked_tags = {}
        self.frame_count = 0
        self.FULL_SCAN_INTERVAL = 30
        self.MAX_LOST_FRAMES = 5
        self.prev_world_pos = None
        self.lost_all_tags_count = 0

        self.get_logger().info("Localization node with fixed TF hierarchy started successfully.")

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

        self.tf_broadcaster.sendTransform(t_static)

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
            self.tf_broadcaster.sendTransform(static_transforms)

    def publish_tag_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

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

            text_marker = Marker()
            text_marker.header.stamp = now
            text_marker.header.frame_id = self.world_frame
            text_marker.ns = "apriltag_labels"
            text_marker.id = int(tag_id) + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(pos_flu[0])
            text_marker.pose.position.y = float(pos_flu[1])
            text_marker.pose.position.z = float(pos_flu[2]) + (size / 2.0) + 0.04
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.05
            text_marker.text = f"Tag {tag_id}"

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            marker_array.markers.append(text_marker)

        self.tag_marker_pub.publish(marker_array)

    def image_callback(self, msg):
        try:
            bgr_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.frame_count += 1
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        detected_tags_in_frame = {}

        run_full_scan = (len(self.tracked_tags) == 0) or (self.frame_count % self.FULL_SCAN_INTERVAL == 0)
        if run_full_scan:
            results = self.detector.detect(gray)
            for r in results: detected_tags_in_frame[r.tag_id] = r.corners
        else:
            for tag_id, track in list(self.tracked_tags.items()):
                x1, y1, x2, y2 = track.get_roi_crop(gray.shape[1], gray.shape[0])
                roi_results = self.detector.detect(gray[y1:y2, x1:x2])
                matched = False
                for r in roi_results:
                    if r.tag_id == tag_id:
                        detected_tags_in_frame[tag_id] = r.corners + np.array([x1, y1])
                        matched = True
                    elif r.tag_id not in detected_tags_in_frame:
                        detected_tags_in_frame[tag_id] = r.corners + np.array([x1, y1])
                if not matched: track.lost_frames += 1

            if not detected_tags_in_frame:
                for r in self.detector.detect(gray): detected_tags_in_frame[r.tag_id] = r.corners

        for tag_id, corners in detected_tags_in_frame.items():
            if tag_id in self.tracked_tags: self.tracked_tags[tag_id].update(corners)
            else: self.tracked_tags[tag_id] = TrackedTag(tag_id, corners)

        self.tracked_tags = {k: v for k, v in self.tracked_tags.items() if v.lost_frames <= self.MAX_LOST_FRAMES}

        annotated_frame = bgr_frame.copy()
        for tag_id, corners in detected_tags_in_frame.items(): 
            draw_tag_overlay(annotated_frame, tag_id, corners)
        
        try:
            annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            annotated_msg.header = msg.header
            annotated_msg.header.frame_id = self.camera_frame 
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

        all_obj_pts, all_img_pts, detected_ids = [], [], []
        for tag_id, corners in detected_tags_in_frame.items():
            if tag_id in GLOBAL_TAG_CORNERS:
                all_obj_pts.append(GLOBAL_TAG_CORNERS[tag_id])
                all_img_pts.append(corners)
                detected_ids.append(tag_id)

        if detected_ids:
            R_wc, t_wb, t_wc, success = None, None, None, False
            if len(detected_ids) == 1:
                tag_id = detected_ids[0]
                R_tw, t_tw = TAG_TRANSFORMS[tag_id]
                try:
                    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(LOCAL_TAG_CORNERS[tag_id], all_img_pts[0], K, D, flags=cv2.SOLVEPNP_IPPE)
                    if retval > 0 and len(rvecs) > 0:
                        success = True
                        if len(rvecs) == 1 or self.prev_world_pos is None:
                            R_wc, t_wb, t_wc = ippe_solution_to_world_pos(rvecs[0], tvecs[0], R_tw, t_tw)
                        else:
                            R0, t0, tc0 = ippe_solution_to_world_pos(rvecs[0], tvecs[0], R_tw, t_tw)
                            R1, t1, tc1 = ippe_solution_to_world_pos(rvecs[1], tvecs[1], R_tw, t_tw)
                            if np.linalg.norm(t0 - self.prev_world_pos) <= np.linalg.norm(t1 - self.prev_world_pos):
                                R_wc, t_wb, t_wc = R0, t0, tc0
                            else:
                                R_wc, t_wb, t_wc = R1, t1, tc1
                except Exception:
                    ok, rvec_fb, tvec_fb = cv2.solvePnP(LOCAL_TAG_CORNERS[tag_id], all_img_pts[0], K, D, flags=cv2.SOLVEPNP_IPPE)
                    if ok:
                        success = True
                        R_wc, t_wb, t_wc = ippe_solution_to_world_pos(rvec_fb, tvec_fb, R_tw, t_tw)
            else:
                first_id = detected_ids[0]
                R_tw0, t_tw0 = TAG_TRANSFORMS[first_id]
                ok_init = False
                rvec_seed, tvec_seed = None, None
                try:
                    retval0, rvecs0, tvecs0, _ = cv2.solvePnPGeneric(LOCAL_TAG_CORNERS[first_id], all_img_pts[0], K, D, flags=cv2.SOLVEPNP_IPPE)
                    if retval0 > 0:
                        if self.prev_world_pos is None or retval0 == 1:
                            rvec_seed, tvec_seed = rvecs0[0], tvecs0[0]
                        else:
                            _, t0s, _ = ippe_solution_to_world_pos(rvecs0[0], tvecs0[0], R_tw0, t_tw0)
                            _, t1s, _ = ippe_solution_to_world_pos(rvecs0[1], tvecs0[1], R_tw0, t_tw0)
                            if np.linalg.norm(t0s - self.prev_world_pos) <= np.linalg.norm(t1s - self.prev_world_pos):
                                rvec_seed, tvec_seed = rvecs0[0], tvecs0[0]
                            else:
                                rvec_seed, tvec_seed = rvecs0[1], tvecs0[1]
                        ok_init = True
                except Exception:
                    ok_init = False

                obj_pts = np.vstack(all_obj_pts).astype(np.float64)
                img_pts = np.vstack(all_img_pts).astype(np.float64)
                if ok_init:
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, D,
                        rvec=rvec_seed, tvec=tvec_seed,
                        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                else:
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_SQPNP)
                    if not ok:
                        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    success = True
                    R_cw, _ = cv2.Rodrigues(rvec)
                    R_wc = R_cw.T
                    t_cw = tvec.reshape(3, 1)
                    t_wc = -R_wc @ t_cw
                    t_wb = t_wc + R_wc @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)

            if success and R_wc is not None:
                self.lost_all_tags_count = 0
                self.prev_world_pos = t_wb.copy()

                t_wb_flu = R_OPTICAL_TO_BODY @ t_wb
                t_wc_flu = R_OPTICAL_TO_BODY @ t_wc
                R_wb_flu = R_OPTICAL_TO_BODY @ R_wc @ R_OPTICAL_TO_BODY.T
                qx_r, qy_r, qz_r, qw_r = rotation_matrix_to_quaternion(R_wb_flu)
                x_b, y_b, z_b = float(t_wb_flu[0]), float(t_wb_flu[1]), float(t_wb_flu[2])

                x_c, y_c, z_c = float(t_wc_flu[0]), float(t_wc_flu[1]), float(t_wc_flu[2])

                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = self.world_frame
                pose_msg.pose.position.x = x_b
                pose_msg.pose.position.y = y_b
                pose_msg.pose.position.z = z_b
                pose_msg.pose.orientation.x = float(qx_r)
                pose_msg.pose.orientation.y = float(qy_r)
                pose_msg.pose.orientation.z = float(qz_r)
                pose_msg.pose.orientation.w = float(qw_r)
                self.pose_pub.publish(pose_msg)

                cam_pose_msg = PoseStamped()
                cam_pose_msg.header.stamp = msg.header.stamp
                cam_pose_msg.header.frame_id = self.world_frame
                cam_pose_msg.pose.position.x = x_c
                cam_pose_msg.pose.position.y = y_c
                cam_pose_msg.pose.position.z = z_c
                R_wc_flu = R_OPTICAL_TO_BODY @ R_wc
                qx_wc, qy_wc, qz_wc, qw_wc = rotation_matrix_to_quaternion(R_wc_flu)
                cam_pose_msg.pose.orientation.x = float(qx_wc)
                cam_pose_msg.pose.orientation.y = float(qy_wc)
                cam_pose_msg.pose.orientation.z = float(qz_wc)
                cam_pose_msg.pose.orientation.w = float(qw_wc)
                self.camera_pose_pub.publish(cam_pose_msg)

                t_stamped_rov = TransformStamped()
                t_stamped_rov.header.stamp = msg.header.stamp
                t_stamped_rov.header.frame_id = self.world_frame
                t_stamped_rov.child_frame_id = self.rov_frame
                t_stamped_rov.transform.translation.x = x_b
                t_stamped_rov.transform.translation.y = y_b
                t_stamped_rov.transform.translation.z = z_b
                t_stamped_rov.transform.rotation.x = float(qx_r)
                t_stamped_rov.transform.rotation.y = float(qy_r)
                t_stamped_rov.transform.rotation.z = float(qz_r)
                t_stamped_rov.transform.rotation.w = float(qw_r)

                qx_c, qy_c, qz_c, qw_c = rotation_matrix_to_quaternion(R_OPTICAL_TO_BODY)
                cam_off_flu = R_OPTICAL_TO_BODY @ CAMERA_TO_BODY_OFFSET.reshape(3, 1)

                t_stamped_cam = TransformStamped()
                t_stamped_cam.header.stamp = msg.header.stamp
                t_stamped_cam.header.frame_id = self.rov_frame
                t_stamped_cam.child_frame_id = self.camera_frame
                t_stamped_cam.transform.translation.x = float(cam_off_flu[0])
                t_stamped_cam.transform.translation.y = float(cam_off_flu[1])
                t_stamped_cam.transform.translation.z = float(cam_off_flu[2])
                t_stamped_cam.transform.rotation.x = float(qx_c)
                t_stamped_cam.transform.rotation.y = float(qy_c)
                t_stamped_cam.transform.rotation.z = float(qz_c)
                t_stamped_cam.transform.rotation.w = float(qw_c)

                self.tf_broadcaster.sendTransform([t_stamped_rov, t_stamped_cam])

                roll, pitch, yaw = rotation_matrix_to_rpy(R_wb_flu)
                self.human_pub.publish(String(data=(
                    f"ROV FLU: X={x_b:+.3f}, Y={y_b:+.3f}, Z={z_b:+.3f} | "
                    f"Cam FLU: X={x_c:+.3f}, Y={y_c:+.3f}, Z={z_c:+.3f} | "
                    f"RPY: R={roll:+.1f} P={pitch:+.1f} Y={yaw:+.1f} | Tags: {detected_ids}")))
        else:
            self.lost_all_tags_count += 1
            if self.lost_all_tags_count > 15: self.prev_world_pos = None

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