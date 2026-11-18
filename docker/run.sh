#!/usr/bin/env bash

# Βρίσκει τον φάκελο όπου είναι αποθηκευμένο ΑΥΤΟ το script (δηλαδή το docker/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Η ρίζα του repository (bluerov_core)
TARGET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Δικαιώματα προβολής γραφικών (X11)
xhost +local:root > /dev/null 2>&1

echo "Starting BlueROV ROS 2 container with workspace: $TARGET_DIR"

docker run -it --rm \
    --name ros2_container \
    --net=host \
    --ipc=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$TARGET_DIR:/root/ros2_ws" \
    --device=/dev/video0:/dev/video0 \
    bluerov_core
