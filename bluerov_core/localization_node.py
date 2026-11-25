#!/usr/bin/env python3
"""
Localization Node (LocalizationNode)
Consumes raw frames from Camera 1 via shared RAM buffer, detects AprilTags,
solves Dual-Engine PnP (Multi-Tag SQPNP & Single-Tag IPPE_SQUARE),
publishes /rov/pose, TFs, and updates shared detection metadata for the annotator.
"""

import threading
import time
import numpy as np
import cv2
import pupil_apriltags as apriltag

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from bluerov_core.common import (
    REFERENCE_TAG_ID, TAG_FAMILY, BIG_TAG_SIZE,
    TAG_MAP, K, D, CAMERA_TO_BODY_OFFSET, R_OPTICAL_TO_BODY,
    GLOBAL_TAG_CORNERS, LOCAL_TAG_CORNERS, TAG_TRANSFORMS,
    rpy_to_rotation_matrix, rotation_matrix_to_rpy,
    rotation_matrix_to_quaternion, quaternion_nlerp,
    ippe_solution_to_world_pos
)


class LocalizationNode(Node):
    def __init__(self, cam1_buffer, cam1_detections):
        super().__init__("localization_node")
        self.cam1_buffer = cam1_buffer
        self.cam1_detections = cam1_detections

        # Parameters
        self.declare_parameter("quad_decimate", 2.0)
        self.declare_parameter("filter_alpha", 0.50)
        self.declare_parameter("max_xy_bound", 2.5)
        self.declare_parameter("max_z_bound", 0.50)
        self.declare_parameter("max_lost_frames", 30)

        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("rov_frame_id", "rov")
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        self.declare_parameter("camera2_frame_id", "camera2_optical_frame")

        self.quad_decimate = float(self.get_parameter("quad_decimate").value)
        self.filter_alpha = float(self.get_parameter("filter_alpha").value)
        self.max_xy_bound = float(self.get_parameter("max_xy_bound").value)
        self.max_z_bound = float(self.get_parameter("max_z_bound").value)
        self.max_lost_frames = int(self.get_parameter("max_lost_frames").value)

        self.world_frame = self.get_parameter("world_frame_id").value
        self.rov_frame = self.get_parameter("rov_frame_id").value
        self.camera_frame = self.get_parameter("camera_frame_id").value
        self.camera2_frame = self.get_parameter("camera2_frame_id").value

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "rov/pose", 10)
        self.human_pub = self.create_publisher(String, "rov/pose_human_readable", 10)
        self.tag_marker_pub = self.create_publisher(MarkerArray, "rov/tags_markers", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.publish_static_transforms()
        self.publish_tag_markers()
        self.marker_timer = self.create_timer(3.0, self.publish_tag_markers)

        # AprilTag Detector
        self.detector = apriltag.Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=self.quad_decimate,
            quad_sigma=0.0,
            refine_edges=1
        )

        # State tracking
        self.is_running = True
        self.frame_count = 0
        self.lost_frames_count = 0
        self.prev_world_pos = None
        self.prev_R_wc = None
        self.filtered_pos = None
        self.filtered_quat = None
        self.filtered_cam_pos = None
        self.filtered_cam_quat = None
        self.last_R_wb_flu = None

        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        # Processing thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Localization worker loop started.")

    def publish_static_transforms(self):
        """Broadcasts static transforms for camera frames and all tags in TAG_MAP."""
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

    def _process_loop(self):
        last_seq = -1
        while rclpy.ok() and self.is_running:
            bgr_frame, recv_time, seq = self.cam1_buffer.get_latest(last_seen_seq=last_seq, timeout=0.1)
            if bgr_frame is None or seq == last_seq:
                continue
            last_seq = seq

            self.frame_count += 1
            now_stamp = self.get_clock().now().to_msg()

            # 1. AprilTag Detection
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            detected_tags = {}
            for r in self.detector.detect(gray):
                if r.tag_id in GLOBAL_TAG_CORNERS:
                    detected_tags[r.tag_id] = r.corners

            # Share detection results with AnnotatedPublisherNode in zero-copy RAM
            self.cam1_detections.put(detected_tags, seq)

            # 2. Dual-Engine PnP Solver
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
                            pos_cost = float(np.linalg.norm(t_wb - self.prev_world_pos)) if self.prev_world_pos is not None else 0.0
                            rot_cost = float(np.linalg.norm(R_wc - self.prev_R_wc)) if self.prev_R_wc is not None else -float(R_wc[2, 2])
                            total_cost = pos_cost + 0.35 * rot_cost
                            valid_candidates.append((total_cost, R_wc, t_wb, t_wc))

                    if valid_candidates:
                        valid_candidates.sort(key=lambda item: item[0])
                        _, R_wc_candidate, t_wb_candidate, t_wc_candidate = valid_candidates[0]
                        solved_new_pose = True
                        mode_str = f"SINGLE (tag {tag_id})"

            # 3. Filtering & Pose Update
            if solved_new_pose and t_wb_candidate is not None:
                self.lost_frames_count = 0
                self.prev_world_pos = t_wb_candidate.copy()
                self.prev_R_wc = R_wc_candidate.copy()

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

            # 4. Broadcast TF & Publish Pose
            if self.filtered_pos is not None:
                x_b, y_b, z_b = self.filtered_pos
                qx_r, qy_r, qz_r, qw_r = self.filtered_quat

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

            # Periodic terminal print & Live FPS measurement (1 Hz)
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
        super().destroy_node()
