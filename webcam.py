# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from model import LipReadingModel
import os

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH  = "best_model.pt"
LABELS_DIR  = r"E:\lrw_frames"
NUM_CLASSES = 500
CROP_SIZE   = 88

NUM_FRAMES  = 29
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ──────────────────────────────────────────────────────────


def load_classes(labels_dir):
    return sorted(os.listdir(labels_dir))


def detect_mouth_roi(frame_bgr, face_cascade):
    """
    Detect the largest face and return an estimated mouth bounding box
    (x, y, w, h) in original frame coords, or None if no face found.
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    # Lips occupy ~62-82% down the face, middle 50% horizontally
    mouth_x = fx + int(fw * 0.15)
    mouth_y = fy + int(fh * 0.66)
    mouth_w = int(fw * 0.70)
    mouth_h = int(fh * 0.28)
    return (mouth_x, mouth_y, mouth_w, mouth_h)


def preprocess_frame(frame, mouth_roi=None):
    """
    Crop tightly around the mouth ROI (with padding), resize to 88x88,
    convert to grayscale and normalize. Falls back to centre-crop if no ROI.
    """
    fh, fw = frame.shape[:2]

    if mouth_roi is not None:
        mx, my, mw, mh = mouth_roi
        # Add 40% horizontal and 60% vertical padding for context
        pad_x = int(mw * 0.40)
        pad_y = int(mh * 0.60)
        x1 = max(0,  mx - pad_x)
        y1 = max(0,  my - pad_y)
        x2 = min(fw, mx + mw + pad_x)
        y2 = min(fh, my + mh + pad_y)
        crop = frame[y1:y2, x1:x2]
    else:
        # Fallback: centre square crop
        side = min(fh, fw)
        cy, cx = fh // 2, fw // 2
        half = side // 2
        crop = frame[cy - half:cy + half, cx - half:cx + half]

    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - 0.4161) / 0.1688
    return crop  # (88, 88)


def predict(model, frames, classes):
    """Run inference on a 29-frame buffer -> top-3 (word, confidence%) list."""
    video = np.stack(frames, axis=0)           # (29, 88, 88)
    video = torch.tensor(video).unsqueeze(1)   # (29,  1, 88, 88)
    video = video.unsqueeze(0).to(DEVICE)      # ( 1, 29,  1, 88, 88)

    with torch.no_grad():
        outputs = model(video)
        probs   = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, 20)  # fetch extra to allow filtering
    results = [(classes[i.item()], p.item() * 100) for p, i in zip(top_probs, top_idx)]
    results = [(w, c) for w, c in results if w.upper() != "CASES"]
    return results[:5]


def draw_mouth_box(frame, mouth_roi, recording):
    """Draw corner-marker bounding box around the mouth region."""
    if mouth_roi is None:
        return frame

    x, y, w, h = mouth_roi
    color = (0, 255, 180) if recording else (80, 80, 200)
    thickness  = 2
    corner_len = max(8, w // 6)

    cv2.line(frame, (x, y),         (x + corner_len, y),         color, thickness)
    cv2.line(frame, (x, y),         (x, y + corner_len),          color, thickness)
    cv2.line(frame, (x + w, y),     (x + w - corner_len, y),     color, thickness)
    cv2.line(frame, (x + w, y),     (x + w, y + corner_len),      color, thickness)
    cv2.line(frame, (x, y + h),     (x + corner_len, y + h),     color, thickness)
    cv2.line(frame, (x, y + h),     (x, y + h - corner_len),      color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len),  color, thickness)

    label = "MOUTH"
    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, label, (x + (w - lw) // 2, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, results, buffer_size, recording, mouth_roi):
    """Composite the full HUD onto the display frame."""
    h, w = frame.shape[:2]

    frame = draw_mouth_box(frame, mouth_roi, recording)

    # Bottom prediction panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 180), (330, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Lip Reading",
                (10, h - 158), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    colors = [(0, 255, 180), (0, 200, 255), (180, 180, 180), (150, 150, 150), (120, 120, 120)]
    for i, (word, conf) in enumerate(results):
        cv2.putText(frame, f"{i+1}. {word}",
                    (10, h - 130 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60 if i == 0 else 0.50,
                    colors[i],
                    2 if i == 0 else 1,
                    cv2.LINE_AA)

    # Progress bar
    bar_fill  = int((buffer_size / NUM_FRAMES) * 300)
    bar_color = (0, 255, 180) if recording else (80, 80, 120)
    cv2.rectangle(frame, (10, h - 15), (310, h - 5), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 15), (10 + bar_fill, h - 5), bar_color, -1)
    cv2.putText(frame, f"{buffer_size}/{NUM_FRAMES}",
                (315, h - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Status badge
    if recording:
        cv2.circle(frame, (w - 55, 22), 8, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 42, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "PAUSED  [S] to start", (w - 200, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 200), 1, cv2.LINE_AA)

    return frame


# ─── MAIN ─────────────────────────────────────────────────
def main():
    print(f"Loading model from {MODEL_PATH} ...")
    classes = load_classes(LABELS_DIR)

    model = LipReadingModel(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded on {DEVICE}.")
    print("Controls:  S = Start recording (predict once)    Q = Quit")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print("WARNING: Could not load Haar cascade — falling back to centre crop.")
        face_cascade = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    recording    = False
    frame_buffer = []
    last_results = [("---", 0.0)] * 5
    mouth_roi    = None   # persisted between frames so we always have the last known ROI

    # Splash screen
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        splash = frame.copy()
        cv2.addWeighted(np.zeros_like(splash), 0.55, splash, 0.45, 0, splash)
        msg = "Press  S  to Start"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(splash, msg, ((w - mw) // 2, (h + mh) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.imshow("Lip Reading - Live", splash)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s') and not recording:
            # Only start if not already recording
            recording    = True
            frame_buffer = []
            last_results = [("---", 0.0)] * 5
            print("[INFO] Recording started.")

        ret, frame = cap.read()
        if not ret:
            break

        # Always detect mouth (use last known ROI if detection fails this frame)
        if face_cascade:
            detected = detect_mouth_roi(frame, face_cascade)
            if detected is not None:
                mouth_roi = detected   # update only when face is found

        # Frame buffering — crop around mouth ROI
        if recording:
            processed = preprocess_frame(frame, mouth_roi)
            frame_buffer.append(processed)

            if len(frame_buffer) == NUM_FRAMES:
                last_results = predict(model, frame_buffer, classes)
                frame_buffer = []
                recording    = False   # ONE prediction then stop
                print("[INFO] Done. Press S to predict again.")

        display = draw_overlay(
            frame.copy(), last_results,
            len(frame_buffer), recording, mouth_roi
        )
        cv2.imshow("Lip Reading - Live", display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()