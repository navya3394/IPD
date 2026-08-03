import sys
import os
import time
from typing import List

import torch

# Lazy import torchvision for transforms
try:
    import torchvision.transforms as T
    _has_torchvision = True
except ImportError:
    print("torchvision not installed. Please install torchvision (`pip install torchvision`).")
    _has_torchvision = False

# Lazy import decord; fallback to cv2 if unavailable
try:
    import decord
    _has_decord = True
except ImportError:
    print("decord not installed. Falling back to cv2 for video reading.")
    _has_decord = False

# cv2 is imported for the fallback path
try:
    import cv2
    _has_cv2 = True
except ImportError:
    print("[WARNING] opencv-python (cv2) not installed – clip_loader will fall back to decord only.")
    _has_cv2 = False

if _has_torchvision:
    CLIP_TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    CLIP_TRANSFORM = None

def load_clip(video_path: str, start_frame: int, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    """Load a clip of ``num_frames`` from ``video_path`` starting at ``start_frame``.

    The function tries to use decord first; if decord is unavailable or fails,
    it falls back to ``cv2.VideoCapture``. Frames are converted to RGB, transformed
    with a shared torchvision pipeline, and stacked into a tensor of shape
    ``(num_frames, 3, size, size)``.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")

    backend = None
    frames: List[torch.Tensor] = []

    # Attempt decord backend
    if _has_decord:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu())
            total = len(vr)
            if start_frame + num_frames > total:
                raise RuntimeError(
                    f"Not enough frames from start_frame; required {num_frames}, got {total - start_frame}"
                )
            # decord returns frames as RGB numpy arrays (H, W, C)
            for idx in range(start_frame, start_frame + num_frames):
                frame = vr[idx].asnumpy()
                img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                if CLIP_TRANSFORM:
                    img = CLIP_TRANSFORM(img)
                frames.append(img)
            backend = "decord"
        except Exception as e:
            print(f"decord backend failed ({e}), falling back to cv2")
            backend = None

    # cv2 fallback
    if backend is None:
        if not _has_cv2:
            raise RuntimeError("opencv-python (cv2) not installed – cannot read video via cv2 fallback.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with cv2: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame + num_frames > total:
        cap.release()
        raise RuntimeError(
            f"Not enough frames from start_frame; required {num_frames}, got {total - start_frame}"
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to read frame from cv2")
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        if CLIP_TRANSFORM:
            img = CLIP_TRANSFORM(img)
        frames.append(img)
    cap.release()
    backend = "cv2"

    print(f"Backend used: {backend}")
    clip_tensor = torch.stack(frames)  # (num_frames, 3, size, size)
    return clip_tensor

# ---------------------------------------------------------------------------
# Dataset that reads a CSV manifest and yields (clip_tensor, label_idx)
# ---------------------------------------------------------------------------
import pandas as pd
from torch.utils.data import Dataset

class ClipManifestDataset(Dataset):
    def __init__(self, manifest_csv_path: str, split: str):
        self.df = pd.read_csv(manifest_csv_path)
        if "split" not in self.df.columns:
            raise RuntimeError("Manifest CSV missing required 'split' column")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        start_frame = int(row["start_frame"])
        num_frames = int(row["num_frames"])
        label_idx = int(row["label_idx"])
        clip = load_clip(video_path, start_frame, num_frames)
        return clip, label_idx

# ---------------------------------------------------------------------------
# Simple CLI sanity‑check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clip_loader.py <video_path> [start_frame]")
        sys.exit(1)
    video_path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    try:
        clip = load_clip(video_path, start)
        print(f"Loaded clip tensor shape: {clip.shape}")
    except Exception as exc:
        print(f"Error loading clip: {exc}")
        sys.exit(1)
