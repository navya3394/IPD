import cv2
import numpy as np
import time
import sys
from typing import Generator, Dict, Any


class VideoIngestor:
    """Video ingestion component.

    Reads frames from a video file or RTSP stream and yields them at a target frame rate.
    Frames are returned as dictionaries matching the system-wide contract.
    """

    def __init__(self, source: str, target_fps: int = 15):
        import os
        self.source = source
        self.target_fps = target_fps
        # Verify the path exists (for file sources)
        if not os.path.isfile(self.source) and not self.source.lower().startswith(('rtsp://', 'http://', 'https://')):
            raise FileNotFoundError(f"Video source not found: {self.source}")
        # Use FFmpeg backend explicitly for better codec support on Windows
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV cannot open '{self.source}'. "
                "The file may be corrupted, missing codecs, or the container is unsupported. "
                "Try re‑encoding the video or using a different file."
            )
        # Native FPS may be 0 for some streams; fall back to 30.
        self.native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / self.target_fps
        # Ratio for skipping frames when native fps > target fps.
        self._skip_ratio = self.native_fps / self.target_fps if self.native_fps > self.target_fps else 1.0
        self._next_yield = 0.0
        self._frame_id = 0

    def get_frame_stream(self) -> Generator[Dict[str, Any], None, None]:
        """Generator that yields frames paced to ``target_fps``.

        Yields:
            dict: ``{"frame": np.ndarray, "frame_id": int, "timestamp": float}``
        """
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    # End of video/stream
                    break
                # Determine if this frame should be yielded.
                if self.native_fps > self.target_fps:
                    # Skip frames evenly.
                    if self._next_yield <= self._frame_id * self._skip_ratio:
                        # Yield this frame.
                        self._frame_id += 1
                        yield {
                            "frame": frame,
                            "frame_id": self._frame_id,
                            "timestamp": time.time(),
                        }
                        self._next_yield += self._skip_ratio
                    else:
                        # Skip this frame.
                        self._frame_id += 1
                        continue
                else:
                    # Native fps <= target_fps – emit every frame, pacing with sleep.
                    self._frame_id += 1
                    yield {
                        "frame": frame,
                        "frame_id": self._frame_id,
                        "timestamp": time.time(),
                    }
                    # Pace to target_fps.
                    time.sleep(self._frame_interval)
        finally:
            self._cap.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <video_path_or_rtsp_url> [target_fps]")
        sys.exit(1)
    video_path = sys.argv[1]
    target_fps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    ingestor = VideoIngestor(video_path, target_fps)
    start = time.time()
    count = 0
    for frame_dict in ingestor.get_frame_stream():
        count += 1
        print(f"frame_id={frame_dict['frame_id']}, timestamp={frame_dict['timestamp']:.3f}")
    elapsed = time.time() - start
    print(f"Total frames yielded: {count}, elapsed time: {elapsed:.2f}s")
