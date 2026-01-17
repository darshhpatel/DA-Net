import cv2
import os

def extract_frames(video_path, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"{prefix}_{idx:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames from {video_path} to {output_dir}")

if __name__ == "__main__":
    # Paths
    hazy_video = "hazy.mp4"
    clear_video = "clear.mp4"
    hazy_out = os.path.join("dataset", "RSID", "train", "hazy")
    clear_out = os.path.join("dataset", "RSID", "train", "GT")
    extract_frames(hazy_video, hazy_out, "hazy")
    extract_frames(clear_video, clear_out, "clear")
