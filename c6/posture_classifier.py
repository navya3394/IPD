import sys
import time
import math

# Geometry constants extracted from reference_walk_posture.ipynb
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

MIN_CONF = 0.4

def get_kp(kp_array, idx, min_conf=MIN_CONF):
    """Return (x, y) if confidence >= min_conf, else None."""
    x, y, c = kp_array[idx]
    return (float(x), float(y)) if c >= min_conf else None

def angle_between(a, b, c):
    """Angle at point b formed by points a-b-c (in degrees)."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = math.sqrt(ax ** 2 + ay ** 2) * math.sqrt(cx ** 2 + cy ** 2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

def classify_posture(kp):
    """Classify posture given a list of 17 (x, y, conf) keypoints.
    Returns 'walking', 'standing', or 'unknown' exactly as in the notebook.
    """
    l_hip = get_kp(kp, LEFT_HIP)
    r_hip = get_kp(kp, RIGHT_HIP)
    l_knee = get_kp(kp, LEFT_KNEE)
    r_knee = get_kp(kp, RIGHT_KNEE)
    l_ankle = get_kp(kp, LEFT_ANKLE)
    r_ankle = get_kp(kp, RIGHT_ANKLE)

    if not all([l_hip, r_hip, l_knee, r_knee]):
        return "unknown"

    hip_width = abs(r_hip[0] - l_hip[0]) if r_hip and l_hip else 50
    score = 0

    # Knee‑ankle angle check
    l_ka = angle_between(l_hip, l_knee, l_ankle) if l_ankle else None
    r_ka = angle_between(r_hip, r_knee, r_ankle) if r_ankle else None
    if (l_ka and 100 < l_ka < 165) or (r_ka and 100 < r_ka < 165):
        score += 1

    # Hip‑knee horizontal offset
    hip_cx = (l_hip[0] + r_hip[0]) / 2
    knee_cx = (l_knee[0] + r_knee[0]) / 2
    if abs(knee_cx - hip_cx) > hip_width * 0.3:
        score += 1

    # Ankle separation
    if l_ankle and r_ankle:
        if abs(l_ankle[0] - r_ankle[0]) > hip_width * 0.5:
            score += 1

    # Knee vertical disparity
    if abs(l_knee[1] - r_knee[1]) > 15:
        score += 1

    if score >= 2:
        return "walking"
    elif score == 1:
        return "unknown"
    else:
        return "standing"

class PostureFallbackEngine:
    """Thin wrapper used by downstream pipelines.
    Tracks latency of the last classification.
    """
    def __init__(self):
        self._last_latency_ms = 0.0

    def get_action(self, track_id: int, keypoints: list) -> dict:
        if len(keypoints) != 17 or any(len(k) != 3 for k in keypoints):
            raise ValueError("keypoints must be shape (17,3)")
        start = time.perf_counter()
        posture = classify_posture(keypoints)
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        # Map to project taxonomy
        if posture == "walking":
            action_label = "walk"
            confidence = 1.0
        else:
            # both "standing" and "unknown" map to "normal"
            action_label = "normal"
            confidence = 0.3 if posture == "unknown" else 1.0
        return {
            "track_id": track_id,
            "action_label": action_label,
            "confidence": confidence,
            "source": "posture_fallback",
            "frame_id": -1,
            "timestamp": time.time(),
        }

    def get_last_latency_ms(self) -> float:
        return self._last_latency_ms

# ---------------------------------------------------------------------------
# Simple sanity‑check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Walking pose – hips 0/50, knees directly below hips, ankles far apart
    walking_kp = [
        [0, 0, 1],  # placeholder for NOSE etc.
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, 1],  # LEFT_SHOULDER placeholder
        [0, 0, 1],  # RIGHT_SHOULDER placeholder
        [0, 0, 1], [0, 0, 1],
        [0, 0, 1], [0, 0, 1],
        [0, 0, 1],  # LEFT_HIP (x=0)
        [50, 0, 1],  # RIGHT_HIP (x=50)
        [0, -120, 1],  # LEFT_KNEE directly under left hip
        [50, -120, 1],  # RIGHT_KNEE directly under right hip
        [-10, -200, 1],  # LEFT_ANKLE far left
        [60, -200, 1],   # RIGHT_ANKLE far right
        [0, 0, 1],  # extra placeholder to reach 17 points
    ]
    # Standing pose – minimal score
    standing_kp = [
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, 1], [0, 0, 1],
        [0, 0, 1],
        [50, 0, 1],
        [0, -120, 1],
        [50, -120, 1],
        [0, -200, 0.2],  # low confidence ankle -> ignored
        [50, -200, 0.2],
        [0, 0, 1],
    ]
    # Incomplete / low confidence pose – triggers unknown
    unknown_kp = [
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, 1], [0, 0, 1],
        [0, 0, 0.1],  # hips below confidence threshold
        [50, 0, 0.1],
        [0, -120, 0.1],
        [50, -120, 0.1],
        [0, -200, 0.1],
        [50, -200, 0.1],
        [0, 0, 1],
    ]
    for name, kp in [("walking", walking_kp), ("standing", standing_kp), ("unknown", unknown_kp)]:
        try:
            result = classify_posture(kp)
            print(f"{name.capitalize()} test -> {result}")
        except Exception as e:
            print(f"Error in {name} test: {e}")
