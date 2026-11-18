#!/usr/bin/env python3
"""
Laptop Webcam 1 Hz Frame & Video Recorder
- Ανοίγει την κάμερα του laptop (V4L2 / OpenCV).
- Προβάλλει ζωντανά την εικόνα σε πραγματικό χρόνο (smooth display).
- Αποθηκεύει καρέ με σταθερό ρυθμό 1 Hz (1 καρέ κάθε 1 δευτερόλεπτο).
- Αποθηκεύει:
    1. Μεμονωμένες εικόνες JPEG υψηλής ποιότητας (frame_XXXXXX.jpg).
    2. Αρχείο βίντεο MP4 με ρυθμό 1 FPS (video_1hz.mp4).
- Ασύγχρονη εγγραφή (Background Queue Worker) ώστε να μην κολλάει το live preview.
- Προορισμός: LAPTOP_PHOTOS/session_YYYYMMDD_HHMMSS/ (ή μέσω ορίσματος --output-dir).
"""

import argparse
import datetime
import os
import queue
import sys
import threading
import time
import cv2
import numpy as np

# ==============================================================================
# ΡΥΘΜΙΣΕΙΣ ΚΑΤΑΓΡΑΦΗΣ
# ==============================================================================
TARGET_FPS = 1.0
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 1.0 δευτερόλεπτο

REQUEST_WIDTH = 1920
REQUEST_HEIGHT = 1080


# ==============================================================================
# ASYNC DISK WRITER WORKER
# ==============================================================================
class AsyncDiskWriter(threading.Thread):
    """Αποθηκεύει εικόνες στο δίσκο ασύγχρονα για να παραμένει ομαλό το live preview."""

    def __init__(self, video_writer=None, max_queue_size=120):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.video_writer = video_writer
        self.running = True

    def save_frame(self, filepath, frame):
        try:
            self.queue.put_nowait((filepath, frame.copy()))
        except queue.Full:
            print("[!] ΠΡΟΣΟΧΗ: Η ουρά εγγραφής γέμισε! Παράλειψη καρέ.")

    def run(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        while self.running or not self.queue.empty():
            try:
                filepath, frame = self.queue.get(timeout=0.1)

                # 1. Αποθήκευση εικόνας JPEG
                if filepath:
                    cv2.imwrite(filepath, frame, encode_params)

                # 2. Προσθήκη στο αρχείο βίντεο MP4 (1 FPS)
                if self.video_writer is not None:
                    self.video_writer.write(frame)

                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


# ==============================================================================
# ΚΥΡΙΑ ΕΚΤΕΛΕΣΗ
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Laptop Webcam 1 Hz Video & Frame Recorder")
    parser.add_argument("--camera-id", type=int, default=0, help="ID κάμερας (προεπιλογή: 0)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Φάκελος αποθήκευσης (αν κενό: LAPTOP_PHOTOS/session_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--no-video-file",
        action="store_true",
        help="Αν οριστεί, αποθηκεύει μόνο εικόνες και όχι αρχείο .mp4",
    )
    args = parser.parse_args()

    # 1. Δημιουργία φακέλου αποθήκευσης
    timestamp_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        session_dir = os.path.abspath(args.output_dir)
    else:
        # Προτίμηση στο ~/ros2_ws/LAPTOP_PHOTOS αν υπάρχει, αλλιώς τοπικά
        home_ws = os.path.expanduser("~/ros2_ws/LAPTOP_PHOTOS")
        base_dir = home_ws if os.path.isdir(os.path.expanduser("~/ros2_ws")) else os.path.abspath("LAPTOP_PHOTOS")
        session_dir = os.path.join(base_dir, f"session_{timestamp_session}")

    os.makedirs(session_dir, exist_ok=True)

    print("=" * 70)
    print(" Laptop Webcam 1 Hz Recorder")
    print(f"[*] Φάκελος αποθήκευσης: {session_dir}")
    print(f"[*] Ρυθμός καταγραφής:   {TARGET_FPS} Hz (1 καρέ κάθε {FRAME_INTERVAL:.1f} sec)")
    print(f"[*] Άνοιγμα κάμερας ID:   {args.camera_id}")
    print("=" * 70)

    # 2. Άνοιγμα κάμερας laptop μέσω OpenCV (V4L2)
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"[!] ΣΦΑΛΜΑ: Αδυναμία ανοίγματος της κάμερας {args.camera_id}.")
        sys.exit(1)

    # Ρύθμιση MJPG & ανάλυσης
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Κάμερα ενεργή! Πραγματική ανάλυση: {actual_w}x{actual_h}")

    # 3. Αρχικοποίηση VideoWriter (1.0 FPS)
    video_writer = None
    video_filepath = None
    if not args.no_video_file:
        video_filepath = os.path.join(session_dir, f"video_1hz_{timestamp_session}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filepath, fourcc, TARGET_FPS, (actual_w, actual_h))
        if not video_writer.isOpened():
            # Fallback σε AVI / MJPG αν το mp4v δεν υποστηρίζεται
            video_filepath = os.path.join(session_dir, f"video_1hz_{timestamp_session}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(video_filepath, fourcc, TARGET_FPS, (actual_w, actual_h))
        print(f"[+] Δημιουργία αρχείου βίντεο: {video_filepath}")

    # 4. Εκκίνηση ασύγχρονου Disk Writer
    writer_thread = AsyncDiskWriter(video_writer=video_writer, max_queue_size=120)
    writer_thread.start()

    window_name = "Laptop Camera (Recording @ 1 Hz) - 'q' to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 720))

    saved_count = 0
    start_time = time.time()
    last_saved_time = 0.0
    flash_saved_until = 0.0

    print("[*] Η καταγραφή ξεκίνησε! Πατήστε 'q' ή 'ESC' στο παράθυρο για έξοδο.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()

            # Έλεγχος χρονισμού 1 Hz (1 καρέ κάθε 1.0 sec)
            if current_time - last_saved_time >= FRAME_INTERVAL:
                saved_count += 1
                last_saved_time = current_time
                flash_saved_until = current_time + 0.35  # Visual flash για 350ms

                # Όνομα αρχείου εικόνας με timestamp
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"frame_{saved_count:06d}_{timestamp_str}.jpg"
                filepath = os.path.join(session_dir, filename)

                writer_thread.save_frame(filepath, frame)

                elapsed = current_time - start_time
                print(
                    f"\r[REC 1Hz] Αποθηκεύτηκαν: {saved_count:4d} καρέ | Χρόνος: {elapsed:6.1f}s | Τελευταίο: {filename}",
                    end="",
                    flush=True,
                )

            # --- HUD Overlay στο Live Window ---
            display = frame.copy()
            elapsed_sec = int(current_time - start_time)
            mins = elapsed_sec // 60
            secs = elapsed_sec % 60

            # Ένδειξη Εγγραφής (Κόκκινος κύκλος & REC)
            blink = int(current_time * 2) % 2 == 0
            rec_color = (0, 0, 255) if blink else (0, 100, 200)
            cv2.circle(display, (35, 35), 10, rec_color, -1)
            cv2.putText(
                display,
                f"REC 1.0 Hz | {mins:02d}:{secs:02d}",
                (55, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Στατιστικά
            info_text = f"Frames: {saved_count} | Res: {actual_w}x{actual_h}"
            cv2.putText(
                display,
                info_text,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Visual Flash όταν αποθηκεύεται νέο καρέ
            if current_time < flash_saved_until:
                cv2.putText(
                    display,
                    "[SAVED 1 Hz]",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Οδηγία εξόδου
            cv2.putText(
                display,
                "Press 'q' or 'ESC' to Stop",
                (30, actual_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[*] Ελήφθη εντολή τερματισμού.")
                break

    except KeyboardInterrupt:
        print("\n[*] Διακοπή από τον χρήστη (Ctrl+C).")
    finally:
        print("\n[*] Ολοκλήρωση εγγραφών και κλείσιμο...")
        writer_thread.stop()
        writer_thread.join(timeout=3.0)

        if video_writer is not None:
            video_writer.release()
            print(f"[+] Το βίντεο αποθηκεύτηκε: {video_filepath}")

        cap.release()
        cv2.destroyAllWindows()

        print("=" * 70)
        print(f"[+] Συνολικά αποθηκεύτηκαν {saved_count} καρέ στο:")
        print(f"    {session_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
