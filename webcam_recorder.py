import cv2
import argparse
from datetime import datetime
from pathlib import Path

def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def best_fps(cap, fallback=30.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fallback if fps is None or fps <= 1 else fps

def try_writers(path_base, frame_size, fps):
    """Try a few codecs; return (writer, filepath) or (None, None)."""
    trials = [
        ("mp4v", ".mp4"),   # widely supported
        ("avc1", ".mp4"),   # sometimes works if system has H.264
        ("XVID", ".avi"),   # very compatible
        ("MJPG", ".avi"),   # large files but robust
    ]
    for fourcc_str, ext in trials:
        out_path = Path(f"{path_base}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer, out_path
        # ensure we release failed handles
        writer.release()
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Toggle webcam recording with 'r'. Quit with 'q'.")
    ap.add_argument("--device", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width",  type=int, default=1280, help="Requested width")
    ap.add_argument("--height", type=int, default=720,  help="Requested height")
    ap.add_argument("--fps",    type=float, default=0.0, help="Force FPS (0 = auto)")
    ap.add_argument("--outdir", type=Path, default=Path("./recordings"), help="Output directory")
    ap.add_argument("--basename", type=str, default="capture", help="Base filename")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device, cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera device {args.device}")

    # Request resolution (some drivers ignore; we’ll read back whatever we get)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Determine actual frame size
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    frame_size = (w, h)

    # Determine FPS
    fps = args.fps if args.fps and args.fps > 0 else best_fps(cap, fallback=30.0)

    recording = False
    writer = None
    outfile = None

    overlay_font = cv2.FONT_HERSHEY_SIMPLEX

    print("Controls: 'r' = start/stop recording, 'n' = new file (while recording), 'q' = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; stopping.")
            break

        # Overlay HUD
        status = f"{'REC' if recording else 'LIVE'}  {w}x{h}@{fps:.1f}  Device:{args.device}"
        color = (0, 0, 255) if recording else (255, 255, 255)
        cv2.putText(frame, status, (10, 30), overlay_font, 0.8, color, 2, cv2.LINE_AA)
        if outfile:
            cv2.putText(frame, str(outfile.name), (10, 60), overlay_font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Webcam", frame)

        if recording and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Toggle recording
        if key == ord('r'):
            if not recording:
                base = args.outdir / f"{args.basename}-{timestamp()}"
                writer, outfile = try_writers(base, frame_size, fps)
                if writer is None:
                    print("Could not open any video writer. Try a different codec/container.")
                    recording = False
                    outfile = None
                else:
                    recording = True
                    print(f"Recording → {outfile}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped.")

        # Start a fresh file without stopping preview
        if key == ord('n') and recording:
            # Close current file and immediately open a new one
            if writer is not None:
                writer.release()
            base = args.outdir / f"{args.basename}-{timestamp()}"
            writer, outfile = try_writers(base, frame_size, fps)
            if writer is None:
                print("Failed to roll to a new file; continuing with preview only.")
                recording = False
                outfile = None
            else:
                print(f"New file → {outfile}")

    # Cleanup
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
