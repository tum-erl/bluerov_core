#!/usr/bin/env python3
"""
BlueROV2 Direct Video Stream Viewer with AprilTag Detection
- Λαμβάνει H.264 stream μέσω UDP απευθείας σε GRAY8.
- Εντοπίζει AprilTags χρησιμοποιώντας τη βιβλιοθήκη pupil_apriltags.
- Σχεδιάζει πράσινο τετράγωνο γύρω από κάθε tag και αναγράφει το Tag ID.
"""

import sys
import threading
import time
import cv2
import gi
import numpy as np
import pupil_apriltags as apriltag

gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ==============================================================================
# ΡΥΘΜΙΣΕΙΣ APRILTAG
# ==============================================================================
# Επίλεξε την οικογένεια των tags σου:
# Επιλογές: 'tagStandard41h12', 'tag36h11', 'tag25h9', 'tag16h5', 'tagCircle21h7', κ.λπ.
TAG_FAMILY = "tagStandard41h12"

UDP_PORT = 5600


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
        """Εξάγει απευθείας 2D uint8 NumPy array (H, W) για Grayscale εικόνα."""
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
# MAIN SCRIPT
# ==============================================================================
def main():
    print(f"[*] Εκκίνηση ανιχνευτή AprilTag για οικογένεια: '{TAG_FAMILY}'")
    detector = apriltag.Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=2.0,  # Υποδειγματοληψία για ταχύτερο εντοπισμό (μειώστε σε 1.0 για μακρινά tags)
        quad_sigma=0.0,
        refine_edges=1,
    )

    print(f"[*] Αναμονή για ροή βίντεο στη θύρα UDP {UDP_PORT}...")
    print("[*] Πατήστε 'q' ή 'ESC' στο παράθυρο προβολής για έξοδο.")

    video = Video(port=UDP_PORT)
    window_name = f"BlueROV2 AprilTag Detection ({TAG_FAMILY})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    first_frame = True
    try:
        while True:
            gray_frame = video.get_latest_frame_blocking()
            if gray_frame is None:
                if not video.running:
                    break
                continue

            if first_frame:
                print(f"[+] Ελήφθησαν δεδομένα! Ανάλυση: {gray_frame.shape[1]}x{gray_frame.shape[0]}")
                first_frame = False

            # 1. Εντοπισμός των tags απευθείας στο Grayscale frame
            tags = detector.detect(gray_frame)

            # 2. Μετατροπή σε BGR αποκλειστικά για την απεικόνιση έγχρωμων σχεδίων
            display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # 3. Σχεδίαση πράσινου περιγράμματος γύρω από κάθε tag
            for tag in tags:
                # Οι κορυφές έρχονται ως float32 (4, 2)
                corners = tag.corners.astype(int)

                # Σχεδίαση του περιγράμματος (κλειστό πολύγωνο, πράσινο χρώμα BGR: (0, 255, 0), πάχος 3px)
                cv2.polylines(
                    display_frame,
                    [corners.reshape((-1, 1, 2))],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=3,
                )

                # Σχεδίαση του κέντρου του tag (κόκκινη κουκκίδα)
                cx, cy = int(tag.center[0]), int(tag.center[1])
                cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)

                # Αναγραφή του Tag ID πάνω από το tag
                label = f"ID: {tag.tag_id}"
                text_origin = (corners[0][0], max(0, corners[0][1] - 10))
                cv2.putText(
                    display_frame,
                    label,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Προβολή του αποτελέσματος
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[*] Διακοπή από τον χρήστη (Ctrl+C).")
    finally:
        print("[*] Τερματισμός GStreamer και κλείσιμο παραθύρων...")
        video.stop()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()