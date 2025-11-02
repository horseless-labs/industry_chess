import cv2
import os
import sys

def extract_frames(video_path, output_dir="output_frames", frames_to_extract=5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has zero frames.")

    step = max(1, total_frames // frames_to_extract)
    frame_indices = [i * step for i in range(frames_to_extract)]
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipping frame {idx} (unreadable)")
            continue
        frame_filename = f"{base_name}_frame{idx:04d}.jpg"
        output_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")

    cap.release()
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py path/to/video.mp4 [output_dir] [num_frames]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_frames"
    frames_to_extract = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    extract_frames(video_path, output_dir, frames_to_extract)