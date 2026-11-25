#!/usr/bin/env python3
"""
System Container Runner (system_container)
Instantiates the modular BlueROV2 nodes in a single process using a MultiThreadedExecutor.
Enables instant zero-copy in-RAM frame sharing while keeping code strictly modularized.
"""

import os
import rclpy
from rclpy.executors import MultiThreadedExecutor

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from bluerov_core.common import SharedContext
from bluerov_core.camera1_node import Camera1Node
from bluerov_core.camera2_node import Camera2Node
from bluerov_core.localization_node import LocalizationNode
from bluerov_core.annotated_publisher_node import AnnotatedPublisherNode


def main(args=None):
    rclpy.init(args=args)

    # 1. Initialize shared zero-copy in-RAM context
    shared_ctx = SharedContext()

    # 2. Instantiate individual modular nodes
    cam1_node = Camera1Node(shared_buffer=shared_ctx.cam1_buffer)
    cam2_node = Camera2Node(shared_buffer=shared_ctx.cam2_buffer)
    loc_node = LocalizationNode(
        cam1_buffer=shared_ctx.cam1_buffer,
        cam1_detections=shared_ctx.cam1_detections
    )
    annotated_node = AnnotatedPublisherNode(
        cam1_buffer=shared_ctx.cam1_buffer,
        cam1_detections=shared_ctx.cam1_detections,
        cam2_buffer=shared_ctx.cam2_buffer
    )

    # 3. Add all nodes to a MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(cam1_node)
    executor.add_node(cam2_node)
    executor.add_node(loc_node)
    executor.add_node(annotated_node)

    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        cam1_node.destroy_node()
        cam2_node.destroy_node()
        loc_node.destroy_node()
        annotated_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
