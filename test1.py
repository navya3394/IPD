import sys
import argparse

# ---------------------------------------------------------------------------
# Helper to ensure we are running inside the project's virtual environment.
# If the script is executed without the venv activated, the imports below will
# raise ImportError which we catch and present a clear message.
# ---------------------------------------------------------------------------
try:
    # c6 – clip loader
    from c6.clip_loader import load_clip
    # c7 – posture classifier
    from c6.posture_classifier import PostureFallbackEngine
    # c8 – crowd density estimator
    from c8.crowd_density import CrowdDensityEstimator
except ImportError as e:
    print("[ERROR] Required modules could not be imported. Make sure the virtual environment is activated (e.g. `uv pip install -r requirements.txt` then `uv run <script>`).")
    print("Details:", e)
    sys.exit(1)

# Optional: OpenCV is used for reading a single frame for the crowd‑density demo.
# Import lazily so the script still works if opencv-python is missing.
try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False
    print("[WARNING] opencv-python not installed – crowd‑density demo will be skipped.")

def run_clip_loader(video_path: str):
    """Load the first 16 frames of the video and print the tensor shape."""
    try:
        clip = load_clip(video_path, start_frame=0, num_frames=16, size=224)
        print(f"[c6] Clip tensor shape: {clip.shape}")
    except Exception as exc:
        print(f"[c6] Error loading clip: {exc}")

def run_posture_classifier():
    """Run the fallback posture engine on a dummy key‑point array.
    The real pipeline would feed actual pose keypoints; for a quick sanity check we
    use a zero‑filled array of the expected shape (17 points × 2 coordinates).
    """
    try:
        import numpy as np
        dummy_kp = np.zeros((17, 3), dtype=np.float32)  # (x, y, conf) shape
        engine = PostureFallbackEngine()
        # Provide a dummy track id (e.g., 0) as required by the API
        result = engine.get_action(track_id=0, keypoints=dummy_kp)
        label = result["action_label"]
        latency = engine.get_last_latency_ms()
        print(f"[c7] Dummy posture label: {label} (latency {latency:.2f} ms)")
    except Exception as exc:
        print(f"[c7] Error in posture classifier: {exc}")

def run_crowd_density(video_path: str):
    """Process the first frame of *video_path* with the crowd‑density estimator.
    If OpenCV is unavailable the function simply reports that the demo is skipped.
    """
    if not _has_cv2:
        print("[c8] Skipping crowd‑density demo because cv2 is not installed.")
        return
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to read a frame from the video.")
        estimator = CrowdDensityEstimator()
        result = estimator.estimate(frame)
        latency = estimator.get_last_latency_ms()
        print(f"[c8] Crowd density result: {result} (latency {latency:.2f} ms)")
    except Exception as exc:
        print(f"[c8] Error in crowd‑density estimator: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Run c6 (clip loader), c7 (posture), and c8 (crowd density) on a video.")
    parser.add_argument("video", help="Path to the input video file")
    args = parser.parse_args()

    video_path = args.video
    print(f"Running unified test on video: {video_path}\n")
    run_clip_loader(video_path)
    run_posture_classifier()
    run_crowd_density(video_path)

if __name__ == "__main__":
    main()