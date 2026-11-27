# BlueROV2 Core

ROS 2 Jazzy localization pipeline using AprilTags for BlueROV2.

## Key Features
- **Specialized Modular Nodes**: The pipeline is split into 4 clear, single-purpose nodes:
  - `Camera1Node`: Captures the primary camera video stream from UDP port `5600` via GStreamer.
  - `Camera2Node`: Captures the secondary camera video stream from UDP port `5604` via GStreamer (even thouht it is not used for the localization).
  - `LocalizationNode`: Detects AprilTags (`tagStandard41h12`) and computes the 3D ROV pose using OpenCV PnP (Multi-tag SQPNP + Single-tag IPPE_SQUARE).
  - `AnnotatedPublisherNode`: Draws tag overlays on the video and publishes lightweight JPEG frames only when a subscriber is actively listening (0% idle CPU overhead).
- **Zero-Copy RAM Sharing**: All 4 nodes run inside a single process (`MultiThreadedExecutor`) and exchange raw frames directly through thread-safe in-RAM ring buffers, keeping the code clean and modular with zero network or ROS serialization latency.
- **Decoupled Launch System**: Run core localization independently in headless mode, launch Foxglove visualization on demand, or start everything together with an all-in-one master launch file.
- **Foxglove Studio Integration**: Zero-latency JPEG-compressed video streaming (960x540 downscaled) with 1MB websocket buffer limits and full 3D URDF model tracking.

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
colcon build --packages-select bluerov_core --symlink-install
source install/setup.bash
```

To open additional terminals inside the same running container, run:
```bash
docker exec -it ros2_container bash
```
And inside the new terminal:
```bash
source install/setup.bash
```

---

## 2. Live Operation (Connected to BlueROV2)

When connected to the physical BlueROV2, you can launch the system according to your operational needs:

### Option A: All-in-One Master Launch (Recommended for full operation)
Starts the entire system (Core Localization + 3D URDF + Foxglove Bridge) in a single command:
```bash
ros2 launch bluerov_core system.launch.py
```

### Option B: Decoupled Operation (Headless & On-Demand UI)

#### Terminal 1: Core Localization (Headless)
Runs only the cameras, AprilTag PnP localization, and pose publishing:
```bash
ros2 launch bluerov_core localization.launch.py
```
*Optional parameters:*
```bash
ros2 launch bluerov_core localization.launch.py cam1_port:=5600 cam2_port:=5604 enable_cam2:=true
```

#### Terminal 2: Visualization & Foxglove Bridge
Publishes the robot URDF model and starts the Foxglove WebSocket bridge (`ws://localhost:8765`):
```bash
ros2 launch bluerov_core visualization.launch.py
```

---

## 3. Offline Testing (Using Recorded Videos)

You can test and run the entire AprilTag localization & Foxglove visualization pipeline **without a physical BlueROV2**, using the pre-recorded air datasets included in the `recordings/` directory *(captured using `bluerov_core/0take_video.py`)*.

The `replay_camera` node replaces the physical robot by streaming any `.mp4` file as a live H.264 RTP/UDP feed to `127.0.0.1:5600`, so the rest of the ROS 2 pipeline runs identically to live operation:

1. **Terminal 1** (Start the simulated Camera 1 stream — uses the latest video in `recordings/` by default):
   ```bash
   ros2 run bluerov_core replay_camera --loop
   ```
   *(To choose a specific video from the repository:)*
   ```bash
   ros2 run bluerov_core replay_camera --video recordings/video_tag2_only.mp4 --loop
   ```

2. **Terminal 2** (Launch the full localization & visualization pipeline):
   ```bash
   ros2 launch bluerov_core system.launch.py enable_cam2:=false
   ```

> **Note:** Before running either Live Operation or Offline Testing, make sure you have selected the appropriate camera calibration parameters (**air** vs **water**) inside `bluerov_core/common.py`.

---

## 4. Foxglove Studio Configuration

1. Open **Foxglove Studio** and connect to `ws://localhost:8765` (or `ws://<HOST_IP>:8765`).
2. Add panels:
   - **Image Panel 1**: Subscribe to `/camera/image_annotated` (Primary navigation camera with detected tags).
   - **Image Panel 2**: Subscribe to `/camera2/image_annotated` (Secondary camera feed with tag overlays).
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
| `/camera/image_annotated` | `sensor_msgs/CompressedImage` | Primary camera feed with multi-colored AprilTag overlays (on-demand) |
| `/camera2/image_annotated` | `sensor_msgs/CompressedImage` | Secondary inspection feed with green tag outlines (on-demand) |
| `/rov/pose` | `geometry_msgs/PoseStamped` | 6-DoF ROV pose in FLU world coordinates (published only when tags are visible) |
| `/rov/tags_markers` | `visualization_msgs/MarkerArray` | 3D visual cubes of active AprilTags in the world |
| `/rov/pose_human_readable`| `std_msgs/String` | Human-readable XYZ (m) and RPY (degrees) telemetry string |
| `/robot_description` | `std_msgs/String` | BlueROV2 URDF description for 3D visualizers |
| `/tf` / `/tf_static` | `tf2_msgs/TFMessage` | Dynamic ROV transforms and static tag locations |

---

## 6. Experimental Setup & Results

Live water tank experiments demonstrating real-time AprilTag visual localization, tracking, and dual-camera monitoring:

<p align="center">
  <img src="image1.jpg" alt="Experiment Setup" width="48%">
  <img src="image2.jpg" alt="Foxglove Studio View" width="48%">
</p>
