#!/usr/bin/env python3
"""
BlueROV2 Video Recorder (0take_video.py)
- Λήψη ροής H.264 σε BGR μέσω GStreamer (UDP port 5600).
- Αποθήκευση σε αρχείο MP4 (recordings/video_<timestamp>.mp4).
- Προαιρετική προβολή σε παράθυρο OpenCV (αν υπάρχει DISPLAY).
"""

import os
import sys
import time
from datetime import datetime
import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

UDP_PORT = 5600
SAVE_DIR = "recordings"


class Video:
    """BlueROV2 zero-latency GStreamer BGR receiver."""

    def __init__(self, port=5600):
        Gst.init(None)
        self.port = port
        self.running = True
        self.sink_name = f"rec_sink_{port}"

        command = (
            f"udpsrc port={self.port} buffer-size=65536 "
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            f"rtpjitterbuffer latency=50 drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 max-threads=2 ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name={self.sink_name} emit-signals=false sync=false max-buffers=1 drop=true"
        )
        self.video_pipe = Gst.parse_launch(command)
        self.video_sink = self.video_pipe.get_by_name(self.sink_name)
        self.video_pipe.set_state(Gst.State.PLAYING)

    def get_latest_frame(self, timeout=0.2):
        if not self.running or not self.video_sink:
            return None

        sample = self.video_sink.emit("try-pull-sample", int(timeout * 1e9))
        if sample is None:
            return None

        while True:
            newer = self.video_sink.emit("try-pull-sample", 0)
            if newer is None:
                break
            sample = newer

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        h, w = caps.get_value("height"), caps.get_value("width")
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            return np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8).copy()
        finally:
            buf.unmap(map_info)

    def stop(self):
        self.running = False
        if self.video_pipe:
            self.video_pipe.set_state(Gst.State.NULL)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(SAVE_DIR, f"video_{ts}.mp4")
    has_display = bool(os.environ.get("DISPLAY"))

    print(f"[*] Αποθήκευση βίντεο στο: {video_path}")
    print(f"[*] Αναμονή για ροή βίντεο στη θύρα UDP {UDP_PORT}...")
    print("[*] Πατήστε Ctrl+C (ή 'q' στο παράθυρο) για τερματισμό.")

    video = Video(port=UDP_PORT)
    writer = None
    frame_count = 0
    t0 = time.monotonic()
    last_print = t0

    if has_display:
        cv2.namedWindow("BlueROV2 Recording", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = video.get_latest_frame(timeout=0.2)
            if frame is None:
                continue

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                print(f"[+] Ξεκίνησε η καταγραφή ({w}x{h} @ 30 FPS)!")

            writer.write(frame)
            frame_count += 1

            now = time.monotonic()
            if now - last_print >= 1.0:
                print(f"[REC {now - t0:5.1f}s] Frames: {frame_count}", flush=True)
                last_print = now

            if has_display:
                cv2.imshow("BlueROV2 Recording", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\n[*] Διακοπή από τον χρήστη (Ctrl+C).")
    finally:
        if writer is not None:
            writer.release()
        video.stop()
        if has_display:
            cv2.destroyAllWindows()
        print(f"[*] Αποθηκεύτηκαν {frame_count} frames στο {video_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
