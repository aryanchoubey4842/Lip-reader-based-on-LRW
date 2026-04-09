# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
INPUT_DIR  = r"D:\lipread_mp4"        # folder with word subfolders
OUTPUT_DIR = r"E:\lrw_frames"     # where .npy files will be saved
FRAME_SIZE = 256   # resize each frame to this before cropping
CROP_SIZE  = 88    # final mouth ROI size (center crop)
NUM_FRAMES = 29    # LRW clips are always 29 frames
# ──────────────────────────────────────────────────────────


def crop_center(frame, crop_size):
    """Center-crop a frame to crop_size x crop_size."""
    h, w = frame.shape[:2]
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return frame[cy - half:cy + half, cx - half:cx + half]


def process_video(video_path, save_path):
    """
    Read a video, extract frames, center-crop to mouth ROI,
    convert to grayscale, and save as a .npy file.
    Shape saved: (NUM_FRAMES, CROP_SIZE, CROP_SIZE) — uint8
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return False

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to standard size
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Center crop to mouth ROI
        frame = crop_center(frame, CROP_SIZE)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"  [SKIP] No frames extracted: {video_path}")
        return False

    # Pad or trim to exactly NUM_FRAMES
    if len(frames) < NUM_FRAMES:
        # Pad by repeating last frame
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])
    else:
        frames = frames[:NUM_FRAMES]

    # Stack to (T, H, W) uint8 array
    video_array = np.stack(frames, axis=0).astype(np.uint8)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, video_array)
    return True


def main():
    print("Starting preprocessing...")

    if not os.path.exists(INPUT_DIR):
        print("ERROR: INPUT_DIR not found:", INPUT_DIR)
        return

    total, skipped = 0, 0

    for word in sorted(os.listdir(INPUT_DIR)):
        word_path = os.path.join(INPUT_DIR, word)
        if not os.path.isdir(word_path):
            continue

        for split in ["train", "val", "test"]:
            split_path = os.path.join(word_path, split)
            if not os.path.isdir(split_path):
                continue

            mp4_files = [f for f in os.listdir(split_path) if f.endswith(".mp4")]
            print(f"[{word}/{split}] {len(mp4_files)} files")

            for file in mp4_files:
                video_path = os.path.join(split_path, file)
                name       = file.replace(".mp4", "")
                save_path  = os.path.join(OUTPUT_DIR, word, split, name + ".npy")

                # Skip already processed files
                if os.path.exists(save_path):
                    total += 1
                    continue

                success = process_video(video_path, save_path)
                if success:
                    total += 1
                else:
                    skipped += 1

    print(f"\nDone. Processed: {total} | Skipped: {skipped}")


if __name__ == "__main__":
    main()