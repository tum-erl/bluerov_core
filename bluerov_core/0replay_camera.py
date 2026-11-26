#!/usr/bin/env python3
"""
BlueROV2 Camera 1 Replay Script (replay_camera.py)
Διαβάζει ένα αποθηκευμένο .mp4 βίντεο και το στέλνει ως H.264 RTP/UDP stream
στη θύρα 5600 (127.0.0.1:5600), προσομοιώνοντας την Κάμερα 1 του BlueROV2.

Χρήση:
  python3 bluerov_core/replay_camera.py
  python3 bluerov_core/replay_camera.py --video recordings/video_20261126_121500.mp4 --loop
  ή:
  ros2 run bluerov_core replay_camera
"""

import argparse
import glob
import os
import sys
import time
import cv2
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def find_latest_video(base_dir="recordings"):
    files = sorted(glob.glob(os.path.join(base_dir, "*.mp4")))
    return files[-1] if files else None


def init_gst_sender(width, height, port=5600):
    Gst.init(None)
    encoders = [
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30",
        "avenc_h264",
    ]
    for enc in encoders:
        cmd = (
            f"appsrc name=src is-live=true format=time "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate=30/1 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"{enc} ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host=127.0.0.1 port={port} sync=false"
        )
        try:
            pipe = Gst.parse_launch(cmd)
            src = pipe.get_by_name("src")
            if pipe.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE and src is not None:
                print(f"[+] Streaming H.264 UDP -> 127.0.0.1:{port} ({enc.split()[0]})")
                return pipe, src
            pipe.set_state(Gst.State.NULL)
        except Exception:
            pass

    raise RuntimeError("Αδυναμία εκκίνησης GStreamer H.264 encoder (x264enc / avenc_h264).")


def main():
    parser = argparse.ArgumentParser(description="Replay recorded MP4 as BlueROV2 Camera 1 UDP stream")
    parser.add_argument("--video", type=str, default="", help="Path στο αρχείο .mp4 (default: το πιο πρόσφατο)")
    parser.add_argument("--port", type=int, default=5600, help="UDP port (default: 5600)")
    parser.add_argument("--loop", action="store_true", help="Επανάληψη βίντεο σε λούπα")
    args, _ = parser.parse_known_args()

    video_path = args.video or find_latest_video("recordings")
    if not video_path or not os.path.exists(video_path):
        print("[!] Δεν βρέθηκε αρχείο .mp4 στον φάκελο 'recordings/'. Δώστε --video <path>.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Αδυναμία ανοίγματος βίντεο: {video_path}", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0
    frame_dt = 1.0 / fps

    print(f"[*] Αναπαραγωγή: {video_path} ({width}x{height} @ {fps:.1f} FPS, loop={args.loop})")
    pipe, appsrc = init_gst_sender(width, height, port=args.port)

    pts_ns = 0
    dur_ns = int(frame_dt * Gst.SECOND)

    try:
        while True:
            t_next = time.monotonic()
            ret, frame = cap.read()
            if not ret or frame is None:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            buf.pts = pts_ns
            buf.duration = dur_ns
            appsrc.emit("push-buffer", buf)
            pts_ns += dur_ns

            t_next += frame_dt
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[*] Διακοπή αναπαραγωγής (Ctrl+C).")
    finally:
        cap.release()
        pipe.set_state(Gst.State.NULL)
        print("[*] Τερματισμός replay.")


if __name__ == "__main__":
    main()
