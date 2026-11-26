#!/usr/bin/env python3
"""
Common configurations, constants, math utilities, and thread-safe shared ring buffers
for zero-copy / intra-process communication between BlueROV2 nodes.
"""

import os
import threading
import time
import numpy as np
import cv2

# ==============================================================================
# 1. GLOBAL CONFIGURATION & TAG MAP
# ==============================================================================
REFERENCE_TAG_ID = 2
TAG_FAMILY = "tagStandard41h12"
SMALL_TAG_SIZE = 0.05
BIG_TAG_SIZE = 0.126

TAG_MAP = {
    2: {"pos": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "size": BIG_TAG_SIZE},
    1: {"pos": [0.24, 0.265, 0.02], "rpy": [0.0, 90.0, 0.0], "size": BIG_TAG_SIZE},
    3: {"pos": [-0.30, 0.13, 0.0], "rpy": [0.0, 90.0, 0.0], "size": BIG_TAG_SIZE},
    9: {"pos": [0.0, 0.575, 0.0], "rpy": [270.0, 0.0, 0.0], "size": BIG_TAG_SIZE},

    1001: {"pos": [-0.260, 0.285, 0.01], "rpy": [0.0, 270.0, 0.0], "size": SMALL_TAG_SIZE},
    1003: {"pos": [0.285, 0.098, 0.0], "rpy": [0.0, 270.0, 0.0], "size": SMALL_TAG_SIZE}
}

# ROV Camera calibration parameters in water
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FX, FY, CX, CY = 1055.3052, 1054.4018, 977.0207, 600.9562
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)
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

# ==============================================================================
# 3. THREAD-SAFE ZERO-COPY SHARED BUFFERS
# ==============================================================================
class SharedFrameBuffer:
    """
    Lock-guarded, condition-signaled single-element ring buffer.
    Always maintains the freshest frame with zero accumulation lag.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._timestamp = None
        self._seq = 0

    def put(self, frame, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self._frame = frame
            self._timestamp = timestamp
            self._seq += 1
            self._cond.notify_all()

    def get_latest(self, last_seen_seq=-1, timeout=0.1):
        """
        Returns (frame, timestamp, seq) if a newer frame exists.
        Blocks up to timeout seconds waiting for a new frame.
        """
        with self._cond:
            start_t = time.time()
            while self._seq <= last_seen_seq:
                remaining = timeout - (time.time() - start_t)
                if remaining <= 0:
                    return None, None, last_seen_seq
                self._cond.wait(timeout=remaining)
            return self._frame, self._timestamp, self._seq


class SharedDetectionsBuffer:
    """
    Thread-safe storage for detected tags and metadata from Localization node
    to be consumed by AnnotatedPublisherNode.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._detections = {}
        self._seq = 0

    def put(self, detections, seq):
        with self._lock:
            self._detections = detections
            self._seq = seq

    def get_for_seq(self, seq):
        with self._lock:
            if self._seq == seq:
                return self._detections.copy()
            return self._detections.copy()


class SharedContext:
    """Central container holding inter-node shared buffers within the same process."""
    def __init__(self):
        self.cam1_buffer = SharedFrameBuffer()
        self.cam2_buffer = SharedFrameBuffer()
        self.cam1_detections = SharedDetectionsBuffer()
        self.cam2_detections = SharedDetectionsBuffer()
