"""
CCTV Crowd Detection — Tiling + YOLOv8
Splits frame into overlapping tiles, detects people in each tile,
merges results using Non-Maximum Suppression (NMS)
Best for: overhead CCTV, dense crowds, small persons
Requirements: pip install ultralytics opencv-python
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────

VIDEO_PATH   = "/Users/digneshbhansali/Desktop/RWF-2000/train/NonFight/14YZVngQ_0.avi"  # ← change this
SKIP_FRAMES  = 2      # process every 2nd frame
CONFIDENCE   = 0.35   # lower than normal — small people have lower confidence
NMS_THRESH   = 0.4    # overlap threshold for merging duplicate detections
TILE_SIZE    = 320    # size of each tile in pixels
TILE_OVERLAP = 80     # overlap between tiles to avoid missing people at edges

GRID_ROWS = 3
GRID_COLS = 3
ZONE_ALERT_THRESHOLD = 3

# ─── Load Model ───────────────────────────────────────────────────────────────

model = YOLO("yolov8m.pt")  # m = better small person detection than n/s
                              # no -pose since crowd is too dense for reliable pose
print("[INFO] Model loaded. Press Q to quit, S to save snapshot.")

# ─── Zone Colors ──────────────────────────────────────────────────────────────

ZONE_NORMAL_COLOR = (255, 255, 255)
ZONE_ACTIVE_COLOR = (0, 255, 255)
ZONE_ALERT_COLOR  = (0, 0, 255)
BBOX_COLOR        = (0, 255, 0)

# ─── Helper: Detect on Tiles ──────────────────────────────────────────────────

def detect_on_tiles(frame, model):
    """
    Splits frame into overlapping tiles, runs YOLO on each,
    returns all detections in original frame coordinates.
    """
    h, w = frame.shape[:2]
    all_boxes  = []  # [x1, y1, x2, y2]
    all_scores = []  # confidence scores

    y = 0
    while y < h:
        x = 0
        while x < w:
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)
            tile = frame[y:y2, x:x2]

            if tile.size == 0:
                x += TILE_SIZE - TILE_OVERLAP
                continue

            # Run YOLO on tile
            results = model(tile, conf=CONFIDENCE, classes=[0], verbose=False)

            for box in results[0].boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Remap coordinates back to full frame
                fx1 = bx1 + x
                fy1 = by1 + y
                fx2 = bx2 + x
                fy2 = by2 + y

                # Filter out detections that are too small (noise)
                box_h = fy2 - fy1
                box_w = fx2 - fx1
                if box_h < 20 or box_w < 10:
                    continue

                all_boxes.append([fx1, fy1, fx2, fy2])
                all_scores.append(conf)

            x += TILE_SIZE - TILE_OVERLAP

        y += TILE_SIZE - TILE_OVERLAP

    return all_boxes, all_scores

# ─── Helper: Non-Maximum Suppression ─────────────────────────────────────────

def apply_nms(boxes, scores):
    """
    Removes duplicate detections from overlapping tiles.
    Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    boxes_array  = np.array(boxes, dtype=np.float32)
    scores_array = np.array(scores, dtype=np.float32)

    # Convert to (x, y, w, h) for cv2.dnn.NMSBoxes
    nms_boxes = []
    for b in boxes_array:
        nms_boxes.append([int(b[0]), int(b[1]),
                          int(b[2] - b[0]), int(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(nms_boxes, scores_array.tolist(),
                                CONFIDENCE, NMS_THRESH)

    if len(indices) == 0:
        return []

    return indices.flatten().tolist()

# ─── Helper: Get Zone ────────────────────────────────────────────────────────

def get_zone(cx, cy, frame_w, frame_h):
    col = min(int(cx / frame_w * GRID_COLS), GRID_COLS - 1)
    row = min(int(cy / frame_h * GRID_ROWS), GRID_ROWS - 1)
    return row, col

# ─── Helper: Draw Zone Grid ───────────────────────────────────────────────────

def draw_zone_grid(frame, zone_counts):
    h, w = frame.shape[:2]
    zone_w = w // GRID_COLS
    zone_h = h // GRID_ROWS

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            count = zone_counts[row][col]

            x1 = col * zone_w
            y1 = row * zone_h
            x2 = x1 + zone_w
            y2 = y1 + zone_h

            if count == 0:
                color = ZONE_NORMAL_COLOR
                alpha = 0.03
            elif count < ZONE_ALERT_THRESHOLD:
                color = ZONE_ACTIVE_COLOR
                alpha = 0.12
            else:
                color = ZONE_ALERT_COLOR
                alpha = 0.22

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            zone_name = f"Z{row * GRID_COLS + col + 1}"
            cv2.putText(frame, zone_name, (x1 + 8, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            if count > 0:
                cv2.putText(frame, f"{count}p", (x1 + 8, y1 + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if count >= ZONE_ALERT_THRESHOLD:
                cv2.putText(frame, "ALERT", (x1 + 8, y1 + 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_ALERT_COLOR, 2)

# ─── Main Loop ────────────────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] Cannot open video. Check your path.")
        return

    frame_count = 0
    last_frame  = None
    start_time  = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]
        frame_count += 1

        # ── Skip frames ───────────────────────────────────────────────────────
        if frame_count % SKIP_FRAMES != 0:
            if last_frame is not None:
                cv2.imshow("CCTV Crowd Detection — Tiling + Zones", last_frame)
                cv2.waitKey(1)
            continue

        # ── Detect people using tiling ────────────────────────────────────────
        all_boxes, all_scores = detect_on_tiles(frame, model)

        # ── Remove duplicate detections with NMS ──────────────────────────────
        keep_indices = apply_nms(all_boxes, all_scores)

        # ── Count people per zone + draw boxes ────────────────────────────────
        zone_counts  = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
        person_count = 0

        for idx in keep_indices:
            x1, y1, x2, y2 = all_boxes[idx]
            conf = all_scores[idx]

            # Person center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Assign to zone
            row, col = get_zone(cx, cy, w, h)
            zone_counts[row][col] += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 1)
            cv2.putText(frame, f"{conf:.0%}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, BBOX_COLOR, 1)

            person_count += 1

        # ── Draw zones on top ─────────────────────────────────────────────────
        draw_zone_grid(frame, zone_counts)

        # ── Alert zone count ──────────────────────────────────────────────────
        alert_count = sum(
            1 for row in zone_counts
            for count in row
            if count >= ZONE_ALERT_THRESHOLD
        )

        # ── HUD ───────────────────────────────────────────────────────────────
        elapsed     = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(frame, f"People: {person_count}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Alert Zones: {alert_count}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if alert_count > 0 else (0, 255, 255), 2)
        cv2.putText(frame, "S=Save  Q=Quit", (w - 180, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        last_frame = frame.copy()
        cv2.imshow("CCTV Crowd Detection — Tiling + Zones", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting.")
            break
        elif key == ord('s'):
            filename = f"crowd_snapshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()