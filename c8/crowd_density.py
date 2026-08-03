import sys
import time
import math
import numpy as np
import cv2

# Lazy import for YOLO model (ultralytics)
try:
    from ultralytics import YOLO
    _has_ultralytics = True
except ImportError:
    print("ultralytics not installed. Please install it with `pip install ultralytics`.")
    _has_ultralytics = False

# Lazy import for decord (used in __main__ video reading fallback)
try:
    import decord
    _has_decord = True
except ImportError:
    print("decord not installed. Falling back to cv2 for video reading.")
    _has_decord = False

# ---------------------------------------------------------------------------
# Core algorithmic helpers extracted from reference_day4_crowd_tiling.py
# (display / UI code stripped out)
# ---------------------------------------------------------------------------

def get_zone(cx, cy, w, h, grid_rows=3, grid_cols=3):
    """Assign a point (cx, cy) to a grid zone.
    Returns (row, col) indices within the grid.
    """
    col = min(int(cx / w * grid_cols), grid_cols - 1)
    row = min(int(cy / h * grid_rows), grid_rows - 1)
    return row, col

def apply_nms(boxes, scores, iou_thresh):
    """Simple NMS implementation for axis‑aligned boxes.
    `boxes` is an (N, 4) array of [x1, y1, x2, y2].
    `scores` is (N,) confidence values.
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def detect_on_tiles(model, frame, tile_size=320, overlap=80, conf_thresh=0.35):
    """Run YOLO detection on overlapping tiles of the frame.
    Returns a list of detections where each detection is a dict with keys:
    - 'box'  : [x1, y1, x2, y2]
    - 'conf' : confidence score
    """
    h, w = frame.shape[:2]
    stride = tile_size - overlap
    boxes = []
    scores = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = frame[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            results = model(tile, conf=conf_thresh, classes=[0], verbose=False)
            if not results or len(results[0].boxes) == 0:
                continue
            for det in results[0].boxes:
                bx = det.xyxy[0].cpu().numpy()
                conf = float(det.conf[0].cpu().numpy())
                # translate tile coordinates back to frame coordinates
                bx[0] += x
                bx[1] += y
                bx[2] += x
                bx[3] += y
                boxes.append(bx)
                scores.append(conf)
    if not boxes:
        return []
    boxes_arr = np.stack(boxes)
    scores_arr = np.array(scores)
    keep_idxs = apply_nms(boxes_arr, scores_arr, iou_thresh=0.4)
    detections = []
    for idx in keep_idxs:
        detections.append({"box": boxes_arr[idx].tolist(), "conf": float(scores_arr[idx])})
    return detections

# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------
class CrowdDensityEstimator:
    def __init__(self, grid_rows=3, grid_cols=3, model_variant="yolov8n", tile_size=320,
                 tile_overlap=80, conf_threshold=0.35, nms_thresh=0.4):
        if not _has_ultralytics:
            raise ImportError("ultralytics is required for CrowdDensityEstimator. Install with `pip install ultralytics`")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.conf_threshold = conf_threshold
        self.nms_thresh = nms_thresh
        # Load YOLO model variant once
        self.model = YOLO(f"{model_variant}.pt")
        self._last_latency_ms = 0.0

    def estimate(self, frame: np.ndarray) -> dict:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame supplied to estimate()")
        start = time.perf_counter()
        h, w = frame.shape[:2]
        # Detect persons on tiles
        detections = detect_on_tiles(self.model, frame, tile_size=self.tile_size,
                                     overlap=self.tile_overlap, conf_thresh=self.conf_threshold)
        # Initialize zone counters
        zone_counts = [[0 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        total_people = 0
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            row, col = get_zone(cx, cy, w, h, self.grid_rows, self.grid_cols)
            zone_counts[row][col] += 1
            total_people += 1
        max_zone_density = max(max(row) for row in zone_counts) if zone_counts else 0
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = latency_ms
        return {
            "zone_counts": zone_counts,
            "total_people": total_people,
            "max_zone_density": max_zone_density,
            "latency_ms": latency_ms,
        }

    def get_last_latency_ms(self) -> float:
        return self._last_latency_ms

# ---------------------------------------------------------------------------
# Simple CLI sanity‑check: iterate over a video once per second
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crowd_density.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    # Open video with decord if available, else cv2
    if _has_decord:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu())
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            step = int(round(fps))  # approx once per second
            estimator = CrowdDensityEstimator()
            for idx in range(0, total_frames, step):
                frame = vr[idx].asnumpy()
                result = estimator.estimate(frame)
                print(f"Frame {idx}: total={result['total_people']}, zones={result['zone_counts']}")
            sys.exit(0)
        except Exception as e:
            print(f"decord failed ({e}), falling back to cv2")
    # cv2 fallback
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video with cv2: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = int(round(fps))
    estimator = CrowdDensityEstimator()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            result = estimator.estimate(frame)
            print(f"Frame {frame_idx}: total={result['total_people']}, zones={result['zone_counts']}")
        frame_idx += 1
    cap.release()
