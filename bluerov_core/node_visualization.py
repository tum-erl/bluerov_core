#!/usr/bin/env python3
"""
BlueROV2 3D Pose Visualizer Node (ROS 2 Jazzy)
Node: rov_visualizer_node
Features:
  - Subscribes to /rov/pose (geometry_msgs/msg/PoseStamped)
  - Real-time interactive 3D display using Matplotlib 3D
  - World Origin (0,0,0) centered with XYZ reference frame (RGB = XYZ)
  - AprilTag markers depicted on the ground plane (Z=0)
  - ROV 3D position marker with its own local body orientation frame (RGB = XYZ)
  - Historical trajectory trail showing the ROV's movement path
  - Live coordinates & Euler angles (Roll, Pitch, Yaw) overlay
  - Mouse rotation (click & drag to orbit, scroll to zoom)
"""

import math
import os
import threading
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

# ==============================================================================
# CONFIGURATION & TAG MAP (matching node_localization_timer.py)
# ==============================================================================
SMALL_TAG_SIZE = 0.05
BIG_TAG_SIZE = 0.125

TAG_MAP = {
    3: {
        "pos": [-0.301, -0.126, 0.055],
        "rpy": [0.0, 270.0, 0.0],
        "size": BIG_TAG_SIZE,
    },
    1: {
        "pos": [0.263, -0.283, 0.0],
        "rpy": [0.0, 0.0, 0.0],
        "size": BIG_TAG_SIZE,
    },
    2: {
        "pos": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
        "size": BIG_TAG_SIZE,
    },

    9: {
        "pos": [0.0, -0.582, -0.06],
        "rpy": [0.0, 90.0, 0.0],
        "size": BIG_TAG_SIZE,
    },

    # 1000: {
    #     "pos": [0.0, -0.589, -0.020],
    #     "rpy": [270.0, 0.0, 0.0],
    #     "size": SMALL_TAG_SIZE,
    # },

    1001: {
        "pos": [0.128, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
        "size": SMALL_TAG_SIZE,
    },

    # 1002: {
    #     "pos": [-0.236, 0.02, -0.02],
    #     "rpy": [270.0, 0.0, 0.0],
    #     "size": SMALL_TAG_SIZE,
    # },

    1003: {
        "pos": [0.285, -0.098, 0.013],
        "rpy": [0.0, 90.0, 0.0],
        "size": SMALL_TAG_SIZE,
    }
}


AXIS_LENGTH_WORLD = 0.25   # Length of world origin axes [m]
AXIS_LENGTH_ROV = 0.15     # Length of ROV body axes [m]
MAX_TRAIL_POINTS = 300     # Maximum history points to display for trajectory
RENDER_FPS = 30            # Update frequency of the 3D visualizer (Hz)


# ==============================================================================
# HELPER FUNCTIONS: QUATERNIONS & ROTATION
# ==============================================================================
def rpy_to_rotation_matrix(rpy_deg):
    """Converts Roll, Pitch, Yaw in degrees to a 3x3 rotation matrix (Z-Y-X convention)."""
    r_rad, p_rad, y_rad = np.radians(rpy_deg)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(r_rad), -np.sin(r_rad)],
        [0.0, np.sin(r_rad), np.cos(r_rad)],
    ], dtype=np.float64)
    Ry = np.array([
        [np.cos(p_rad), 0.0, np.sin(p_rad)],
        [0.0, 1.0, 0.0],
        [-np.sin(p_rad), 0.0, np.cos(p_rad)],
    ], dtype=np.float64)
    Rz = np.array([
        [np.cos(y_rad), -np.sin(y_rad), 0.0],
        [np.sin(y_rad), np.cos(y_rad), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return Rz @ Ry @ Rx


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Converts a quaternion to a 3x3 orthonormal rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def rotation_matrix_to_rpy(R):
    """Converts a 3x3 rotation matrix to Roll, Pitch, Yaw in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ==============================================================================
# ROS 2 SUBSCRIBER NODE
# ==============================================================================
class ROVVisualizerNode(Node):

    def __init__(self):
        super().__init__("rov_visualizer_node")

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "rov/pose",
            self.pose_callback,
            10,
        )

        self.lock = threading.Lock()
        self.latest_pose = None
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.total_msgs = 0
        self.last_msg_time = None

        self.get_logger().info("ROV 3D Visualizer Node initialized.")
        self.get_logger().info("Listening to topic: /rov/pose ...")

    def pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.latest_pose = msg
            self.last_msg_time = time.time()
            self.total_msgs += 1

            p = msg.pose.position
            self.trajectory_x.append(p.x)
            self.trajectory_y.append(p.y)
            self.trajectory_z.append(p.z)

            if len(self.trajectory_x) > MAX_TRAIL_POINTS:
                self.trajectory_x.pop(0)
                self.trajectory_y.pop(0)
                self.trajectory_z.pop(0)


# ==============================================================================
# 3D RENDERER (MATPLOTLIB)
# ==============================================================================
class Visualizer3D:

    def __init__(self, node: ROVVisualizerNode):
        self.node = node
        self.is_running = True

        plt.style.use("default")
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("BlueROV2 3D Pose Visualization")
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.fig.canvas.mpl_connect("close_event", self.on_close)

        # Plot elements that persist / get updated (Mapped: X=Right, Y=Depth/Z_world, Z=Up/Y_world)
        self.trail_line, = self.ax.plot([], [], [], color="#3498db", lw=1.8, alpha=0.7, label="ROV Path")
        self.rov_point = self.ax.scatter([], [], [], color="#e74c3c", s=90, depthshade=False, label="ROV Position")

        # Quiver arrows for ROV body orientation frame (X=Red, Y=Green, Z=Blue)
        self.rov_quiver_x = None
        self.rov_quiver_y = None
        self.rov_quiver_z = None

        # Text overlay
        self.info_text = self.fig.text(
            0.02, 0.94,
            "Waiting for /rov/pose...",
            fontsize=10,
            family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
        )

        self.setup_static_scene()

    def on_close(self, event):
        self.is_running = False

    def setup_static_scene(self):
        """Draws the fixed World Origin and reference AprilTags with Y Up and X Left (Right-Handed System)."""
        L = AXIS_LENGTH_WORLD
        # World Origin Axes:
        # X (Left) -> Plot Axis 0 (Red) pointing towards -X_plot
        self.ax.quiver(0, 0, 0, -L, 0, 0, color="red", lw=2.5, arrow_length_ratio=0.15)
        self.ax.text(-L * 1.15, 0, 0, "X", color="red", fontweight="bold", fontsize=9)

        # Z (Distance/Forward) -> Plot Axis 1 (Blue)
        self.ax.quiver(0, 0, 0, 0, L, 0, color="blue", lw=2.5, arrow_length_ratio=0.15)
        self.ax.text(0, L * 1.15, 0, "Z", color="blue", fontweight="bold", fontsize=9)

        # Y (Up / Vertical) -> Plot Axis 2 (Green)
        self.ax.quiver(0, 0, 0, 0, 0, L, color="green", lw=2.5, arrow_length_ratio=0.15)
        self.ax.text(0, 0, L * 1.15, "Y", color="green", fontweight="bold", fontsize=9)

        # Origin point marker
        self.ax.scatter([0], [0], [0], color="black", s=50, marker="o", label="Origin (Tag 2)")

        # Legend proxy entries for clean labeling
        self.ax.plot([], [], [], color="#2c3e50", lw=2, label="AprilTag")

        # Draw AprilTags properly oriented in 3D using their RPY angles
        for tag_id, tag_data in TAG_MAP.items():
            pos = tag_data["pos"]
            rpy = tag_data["rpy"]
            size = tag_data["size"]
            hs = size / 2.0
            cx, cy, cz = pos

            # Local corners in tag frame:
            # Tag lies on XY plane (Z=0), with +Z pointing forward out of the tag's face
            local_corners = np.array([
                [-hs, -hs, 0.0],
                [ hs, -hs, 0.0],
                [ hs,  hs, 0.0],
                [-hs,  hs, 0.0],
                [-hs, -hs, 0.0],  # Close the square loop
            ], dtype=np.float64)

            R_tag = rpy_to_rotation_matrix(rpy)
            t_tag = np.array(pos, dtype=np.float64).reshape(3, 1)

            # Global 3D corners: (5, 3)
            world_corners = (R_tag @ local_corners.T + t_tag).T

            # In plot coordinates: X_plot = -X_world, Y_plot = Z_world (depth), Z_plot = Y_world (height)
            tag_plot_xs = -world_corners[:, 0]
            tag_plot_ys = world_corners[:, 2]
            tag_plot_zs = world_corners[:, 1]

            # Draw tag square boundary
            self.ax.plot(tag_plot_xs, tag_plot_ys, tag_plot_zs, color="#2c3e50", lw=2, linestyle="-")

            # Draw inner marker square for clear visual appearance
            inner_hs = hs * 0.55
            local_inner = np.array([
                [-inner_hs, -inner_hs, 0.0],
                [ inner_hs, -inner_hs, 0.0],
                [ inner_hs,  inner_hs, 0.0],
                [-inner_hs,  inner_hs, 0.0],
                [-inner_hs, -inner_hs, 0.0],
            ], dtype=np.float64)
            world_inner = (R_tag @ local_inner.T + t_tag).T
            self.ax.plot(-world_inner[:, 0], world_inner[:, 2], world_inner[:, 1], color="#7f8c8d", lw=1.0, linestyle=":")

            # Label text in the center of the tag in dark yellow
            self.ax.text(
                -cx, cz, cy,
                f"{tag_id}",
                color="#b8860b",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

        self.ax.legend(loc="upper right", fontsize=8)

        # Labels
        self.ax.set_xlabel("X [m]", fontsize=10, labelpad=8)
        self.ax.set_ylabel("Z [m] (Distance)", fontsize=10, labelpad=8)
        self.ax.set_zlabel("Y [m] (Up)", fontsize=10, labelpad=8)

        # Perspective view
        self.ax.view_init(elev=20, azim=-65)
        self.update_limits(0.0, 0.0, 1.0)

    def update_limits(self, x, y, z):
        """Maintains cubic aspect ratio around the scene so distances are to scale."""
        span = max(abs(x), abs(y), abs(z), 0.8) * 1.25
        self.ax.set_xlim([-span, span])
        self.ax.set_ylim([-0.2, span * 1.8])   # Z_world (Distance)
        self.ax.set_zlim([-span, span])        # Y_world (Up / Height)
        try:
            self.ax.set_box_aspect([1.0, 1.0, 1.0])
        except AttributeError:
            pass

    def update_frame(self):
        """Called periodically to refresh the 3D plot."""
        with self.node.lock:
            pose = self.node.latest_pose
            xs = list(self.node.trajectory_x)
            ys = list(self.node.trajectory_y)
            zs = list(self.node.trajectory_z)
            last_time = self.node.last_msg_time

        if pose is None:
            plt.pause(0.05)
            return

        p = pose.pose.position
        q = pose.pose.orientation

        # Update trajectory trail (Mapped: X_plot=-x, Y_plot=z, Z_plot=y)
        if len(xs) > 1:
            plot_xs = [-x for x in xs]
            self.trail_line.set_data(plot_xs, zs)
            self.trail_line.set_3d_properties(ys)

        # Update ROV point marker: (-X, Z_world, Y_world)
        self.rov_point._offsets3d = ([-p.x], [p.z], [p.y])

        # Compute ROV Orientation and Body Frame Axes
        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        roll, pitch, yaw = rotation_matrix_to_rpy(R)

        L = AXIS_LENGTH_ROV
        dir_x = R @ np.array([L, 0.0, 0.0])
        dir_y = R @ np.array([0.0, L, 0.0])
        dir_z = R @ np.array([0.0, 0.0, L])

        # Remove previous orientation quivers
        if self.rov_quiver_x is not None:
            self.rov_quiver_x.remove()
        if self.rov_quiver_y is not None:
            self.rov_quiver_y.remove()
        if self.rov_quiver_z is not None:
            self.rov_quiver_z.remove()

        # Draw ROV Body Frame (Mapped: u=-dir[0], v=dir[2], w=dir[1]):
        # Body X = Red, Body Y = Green, Body Z = Blue
        self.rov_quiver_x = self.ax.quiver(-p.x, p.z, p.y, -dir_x[0], dir_x[2], dir_x[1], color="#e74c3c", lw=2.2, arrow_length_ratio=0.2)
        self.rov_quiver_y = self.ax.quiver(-p.x, p.z, p.y, -dir_y[0], dir_y[2], dir_y[1], color="#2ecc71", lw=2.2, arrow_length_ratio=0.2)
        self.rov_quiver_z = self.ax.quiver(-p.x, p.z, p.y, -dir_z[0], dir_z[2], dir_z[1], color="#3498db", lw=2.2, arrow_length_ratio=0.2)

        # Update Live Information Box
        age = time.time() - last_time if last_time else 0.0
        status_color = "ONLINE" if age < 1.0 else f"NO DATA ({age:.1f}s ago)"

        info = (
            f"Status      : {status_color}\n"
            f"Position    : X={p.x:+.3f}m, Y={p.y:+.3f}m, Z={p.z:+.3f}m\n"
            f"Orientation : Roll={roll:+.1f}°, Pitch={pitch:+.1f}°, Yaw={yaw:+.1f}°\n"
            f"Distance    : {math.sqrt(p.x**2 + p.y**2 + p.z**2):.3f} m\n"
            f"Frames Recv : {self.node.total_msgs}"
        )
        self.info_text.set_text(info)

        # Dynamic limit adjustment if ROV moves far
        dist = max(abs(p.x), abs(p.y), abs(p.z))
        if dist > 1.2:
            self.update_limits(p.x, p.y, p.z)

        self.fig.canvas.draw_idle()
        plt.pause(1.0 / RENDER_FPS)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ROVVisualizerNode()

    # Spin ROS 2 callbacks in a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    visualizer = Visualizer3D(node)

    try:
        while visualizer.is_running and rclpy.ok():
            visualizer.update_frame()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
