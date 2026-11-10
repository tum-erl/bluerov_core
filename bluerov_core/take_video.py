#!/usr/bin/env python3
"""
BlueROV2 Direct Video Stream Viewer & 30 FPS Frame Saver
- Λήψη ροής H.264 σε GRAY8 μέσω GStreamer.
- Προβολή σε παράθυρο OpenCV.
- Ρυθμισμένη καταγραφή στα 30 FPS.
- Ασύγχρονη αποθήκευση εικόνων σε ξεχωριστό background thread.
- Προορισμός: ~/ros2_ws/BLUEROV_PHOTOS/
"""

import os
import queue
import sys
import threading
import time
import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ==============================================================================
# ΡΥΘΜΙΣΕΙΣ ΑΠΟΘΗΚΕΥΣΗΣ & FPS
# ==============================================================================
TARGET_FPS = 1.0
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.0333 δευτερόλεπτα

# Ανάλυση του path (το os.path.expanduser μεταφράζει το ~ σε /root μέσα στο container)
SAVE_DIR = os.path.expanduser("~/ros2_ws/BLUEROV_PHOTOS")
os.makedirs(SAVE_DIR, exist_ok=True)

UDP_PORT = 5600


# ==============================================================================
# ASYNC DISK WRITER WORKER
# ==============================================================================
class AsyncImageWriter(threading.Thread):
    """Αποθηκεύει εικόνες στο δίσκο ασύγχρονα για να μην μπλοκάρει το live stream."""

    def __init__(self, max_queue_size=120):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True

    def save(self, filepath, frame):
        try:
            # Αν η ουρά γεμίσει (π.χ. πολύ αργός δίσκος), απορρίπτει το παλαιότερο καρέ
            self.queue.put_nowait((filepath, frame))
        except queue.Full:
            pass

    def run(self):
        # Παράμετροι ταχείας κωδικοποίησης JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        while self.running or not self.queue.empty():
            try:
                filepath, frame = self.queue.get(timeout=0.1)
                cv2.imwrite(filepath, frame, encode_params)
                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


# ==============================================================================
# BLUEROV2 VIDEO CAPTURE CLASS
# ==============================================================================
class Video:
    """BlueROV video capture class με βελτιστοποιημένο GRAY8 GStreamer pipeline."""

    def __init__(self, port=5600):
        Gst.init(None)

        self.port = port
        self.latest_frame = None
        self.running = True
        self.condition = threading.Condition()

        self.video_source = (
            f"udpsrc port={self.port} "
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            f"rtpjitterbuffer latency=0 drop-on-latency=true"
        )
        self.video_codec = "! rtph264depay ! h264parse ! avdec_h264 max-threads=2"
        self.video_decode = "! videoconvert ! video/x-raw, format=GRAY8"
        self.video_sink_conf = (
            "! appsink name=appsink0 emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        if not config:
            config = [
                "videotestsrc",
                "! videoconvert ! video/x-raw, format=GRAY8",
                "! appsink name=appsink0",
            ]

        command = " ".join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_sink = self.video_pipe.get_by_name("appsink0")
        self.video_pipe.set_state(Gst.State.PLAYING)

    @staticmethod
    def gst_to_opencv(sample):
        """Εξάγει απευθείας 2D uint8 NumPy array (H, W) για Grayscale επεξεργασία."""
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        h = caps_structure.get_value("height")
        w = caps_structure.get_value("width")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            frame = np.ndarray((h, w), buffer=map_info.data, dtype=np.uint8).copy()
            return frame
        finally:
            buf.unmap(map_info)

    def run(self):
        self.start_gst([
            self.video_source,
            self.video_codec,
            self.video_decode,
            self.video_sink_conf,
        ])
        self.video_sink.connect("new-sample", self.callback)

    def callback(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            frame = self.gst_to_opencv(sample)
            if frame is not None:
                with self.condition:
                    self.latest_frame = frame
                    self.condition.notify()

        return Gst.FlowReturn.OK

    def get_latest_frame_blocking(self):
        """Μπλοκάρει αποδοτικά μέχρι να φτάσει νέο frame."""
        with self.condition:
            while self.latest_frame is None and self.running:
                self.condition.wait(timeout=0.1)

            if not self.running:
                return None

            frame = self.latest_frame
            self.latest_frame = None  # Consume frame
            return frame

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.video_pipe:
            self.video_pipe.set_state(Gst.State.NULL)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print(f"[*] Αποθήκευση εικόνων στο: {SAVE_DIR}")
    print(f"[*] Ρυθμός καταγραφής: {TARGET_FPS} FPS")
    print(f"[*] Αναμονή για ροή βίντεο στη θύρα UDP {UDP_PORT}...")
    print("[*] Πατήστε 'q' ή 'ESC' στο παράθυρο προβολής για έξοδο.")

    writer = AsyncImageWriter(max_queue_size=120)
    writer.start()

    video = Video(port=UDP_PORT)
    window_name = "BlueROV2 Grayscale (Recording 1 FPS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    saved_count = 0
    last_saved_time = 0.0
    first_frame = True

    try:
        while True:
            frame = video.get_latest_frame_blocking()
            if frame is None:
                if not video.running:
                    break
                continue

            current_time = time.time()

            if first_frame:
                print(f"[+] Ελήφθησαν δεδομένα! Ανάλυση: {frame.shape[1]}x{frame.shape[0]}")
                last_saved_time = current_time
                first_frame = False

            # Έλεγχος χρονισμού για διατήρηση σταθερού ρυθμού 30 FPS
            if current_time - last_saved_time >= FRAME_INTERVAL:
                timestamp_str = f"{current_time:.3f}"
                filename = f"frame_{saved_count:06d}_{timestamp_str}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)

                writer.save(filepath, frame)
                saved_count += 1
                last_saved_time = current_time

            # Προβολή
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[*] Διακοπή από τον χρήστη (Ctrl+C).")
    finally:
        print(f"[*] Συνολικά καρέ που στάλθηκαν για αποθήκευση: {saved_count}")
        print("[*] Ολοκλήρωση εγγραφής εκκρεμών εικόνων στο δίσκο...")
        writer.stop()
        writer.join(timeout=3.0)
        video.stop()
        cv2.destroyAllWindows()
        print("[*] Τερματισμός επιτυχής.")
        sys.exit(0)


if __name__ == "__main__":
    main()