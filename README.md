# BlueROV2 Core

ROS 2 Jazzy localization, vision, and telemetry pipeline for BlueROV2.

Features:
- **All-in-One Node (`bluerov_system_node`)**: High-rate visual localization (30+ FPS) using AprilTag detection (`tagStandard41h12`) with dual-engine PnP (Multi-tag SQPNP + Single-tag IPPE_SQUARE + EMA filtering).
- **Dual Camera Support**: 
  - Primary Camera (UDP port 5600): Navigation, 6-DoF pose estimation, and visual tracking overlay.
  - Secondary Camera (UDP port 5604): Auxiliary inspection feed with auto-detection and green bounding boxes for AprilTags.
- **Foxglove Studio Optimized**: Fast JPEG-compressed video streaming (960x540 downscaled) to minimize bandwidth and eliminate WebSocket latency.
- **3D Visualization & URDF**: `robot_state_publisher` broadcasting 3D model transforms and physical tag markers.

---

## 1. Quick Start with Docker

### A. Build the Docker Image
From the repository root:
```bash
cd docker
docker build -t bluerov_core .
```

### B. Launch the Container
```bash
./run.sh
```
*(This mounts your workspace inside the container at `/root/ros2_ws` and forwards network/X11 graphics).*

### C. Build the Workspace
Inside the container terminal:
```bash
cd /root/ros2_ws
colcon build --packages-select bluerov_core --symlink-install
source install/setup.bash
```

To open additional terminals inside the same running container:
```bash
docker exec -it ros2_container bash
source /root/ros2_ws/install/setup.bash
```

---

## 2. Running Directly on Host (Without Docker)

If you have ROS 2 Jazzy installed locally on Ubuntu 24.04:

```bash
cd ~/Projects/Robotics/BlueROV2_ws
colcon build --packages-select bluerov_core --symlink-install
source install/setup.bash
```

---

## 3. Running the System (3 Nodes)

Run each of the following commands in a separate terminal:

### Terminal 1: 3D Model / URDF Publisher
Publishes the robot description and static links for Foxglove:
```bash
ros2 launch bluerov_core bringup.launch.py
```

### Terminal 2: Foxglove WebSocket Bridge
Connects ROS 2 topics to Foxglove Studio via WebSockets (`ws://localhost:8765`):
```bash
ros2 run foxglove_bridge foxglove_bridge
```

### Terminal 3: BlueROV All-in-One System Node
Starts video ingestion (ports 5600 & 5604), tag detection, localization, and compressed image streams:
```bash
ros2 run bluerov_core bluerov_system
```

---

## 4. Foxglove Studio Configuration

1. Open **Foxglove Studio** and connect to `ws://localhost:8765` (or `ws://<HOST_IP>:8765`).
2. Add panels:
   - **Image Panel 1**: Subscribe to `/camera/image_annotated` (Primary navigation camera with tags).
   - **Image Panel 2**: Subscribe to `/camera2/image_annotated` (Secondary camera with green tags).
   - **3D Panel**: 
     - Set Fixed Frame: `world`
     - Enable **URDF** (`/robot_description`)
     - Enable **Markers** (`/rov/tags_markers`)
     - Enable **Pose** (`/rov/pose`)
   - **Teleplot / State**: Subscribe to `/rov/pose_human_readable` for live text telemetry.

---

## 5. Topics Summary

| Topic | Type | Description |
|---|---|---|
| `/camera/image_annotated` | `sensor_msgs/CompressedImage` | Primary camera feed with multi-colored AprilTag overlays |
| `/camera2/image_annotated` | `sensor_msgs/CompressedImage` | Secondary inspection feed with green tag outlines |
| `/rov/pose` | `geometry_msgs/PoseStamped` | 6-DoF ROV pose in FLU world coordinates |
| `/rov/camera_pose` | `geometry_msgs/PoseStamped` | Camera 1 pose in world frame |
| `/rov/tags_markers` | `visualization_msgs/MarkerArray` | 3D visual cubes of active AprilTags in the world |
| `/rov/pose_human_readable`| `std_msgs/String` | Human-readable XYZ (m) and RPY (degrees) string |
| `/tf` / `/tf_static` | `tf2_msgs/TFMessage` | Dynamic ROV transforms and static tag locations |
