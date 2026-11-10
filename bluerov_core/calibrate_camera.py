#!/usr/bin/env python3
"""
Camera Calibration Script for BlueROV2 Underwater Imagery
- Διαβάζει εικόνες από το απόλυτο path: /root/ros2_ws/BLUEROV_PHOTOS/selected_photos
- Υπολογίζει K (FX, FY, CX, CY) και D (distortion coefficients)
- Αποθηκεύει το αποτέλεσμα σε 'calibration_params.txt' στον ίδιο φάκελο με το script
"""

import glob
import os
import cv2
import numpy as np

# ==============================================================================
# ΡΥΘΜΙΣΕΙΣ CHECKERBOARD & PATHS
# ==============================================================================
CHECKERBOARD_INTERNAL_CORNERS = (7, 5)  # Εσωτερικές γωνίες (στήλες, γραμμές)
SQUARE_SIZE_METERS = 0.040              # Μέγεθος πλευράς τετραγώνου σε μέτρα

# Απόλυτο μονοπάτι για τις φωτογραφίες
IMAGES_DIR = "/root/ros2_ws/BLUEROV_PHOTOS/selected_photos_in_air"

# Αποθήκευση του txt αρχείου στον ίδιο φάκελο όπου βρίσκεται το τρέχον script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_TXT_FILE = os.path.join(SCRIPT_DIR, "calibration_params.txt")


def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cols, rows = CHECKERBOARD_INTERNAL_CORNERS
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE_METERS

    objpoints = []
    imgpoints = []

    search_path = os.path.join(IMAGES_DIR, "*.[jJ][pP]*[gG]")
    images = sorted(glob.glob(search_path))

    if not images:
        print(f"[!] Δεν βρέθηκαν εικόνες στο: {IMAGES_DIR}")
        return

    print(f"[*] Βρέθηκαν {len(images)} εικόνες στο {IMAGES_DIR}. Έναρξη επεξεργασίας...")

    gray_shape = None
    successful_frames = 0

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray_shape is None:
            gray_shape = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_INTERNAL_CORNERS,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            successful_frames += 1
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            print(f"  [{idx+1}/{len(images)}] OK: {os.path.basename(fname)}")
        else:
            print(f"  [{idx+1}/{len(images)}] FAILED: {os.path.basename(fname)}")

    print(f"\n[*] Επιτυχής εντοπισμός σε {successful_frames}/{len(images)} εικόνες.")

    if successful_frames < 10:
        print("[!] ΠΡΟΕΙΔΟΠΟΙΗΣΗ: Απαιτούνται τουλάχιστον 10-15 εικόνες για αξιόπιστο calibration.")
        if successful_frames == 0:
            return

    # Υπολογισμός Calibration
    print("[*] Εκτέλεση cv2.calibrateCamera...")
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # Υπολογισμός Mean Reprojection Error
    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"[*] Mean Reprojection Error: {mean_error:.4f} pixels")

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    d_vals = ", ".join([f"{val:.8f}" for val in D.ravel()])

    # Δομή κειμένου για το txt αρχείο
    output_content = (
        f"# Mean Reprojection Error: {mean_error:.4f} px\n\n"
        f"FX = {fx:.4f}\n"
        f"FY = {fy:.4f}\n"
        f"CX = {cx:.4f}\n"
        f"CY = {cy:.4f}\n\n"
        f"K = np.array([\n"
        f"    [FX, 0.0, CX],\n"
        f"    [0.0, FY, CY],\n"
        f"    [0.0, 0.0, 1.0]\n"
        f"], dtype=np.float64)\n\n"
        f"D = np.array([{d_vals}], dtype=np.float64)\n"
    )

    # Εγγραφή στο txt αρχείο
    with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n[+] Τα αποτελέσματα αποθηκεύτηκαν στο:\n    {OUTPUT_TXT_FILE}\n")
    print(output_content)


if __name__ == "__main__":
    main()