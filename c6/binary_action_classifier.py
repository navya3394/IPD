import os
import sys
import time
import torch
from typing import Dict

# Local imports
from .clip_loader import load_clip
from .slowfast_binary_model import load_binary_model, PackPathway

class BinaryActionClassifier:
    """Binary action classifier using a pre‑trained SlowFast checkpoint.

    Attributes
    ----------
    model : torch.nn.Module
        Loaded SlowFast model ready for inference.
    config : dict
        Configuration dictionary extracted from the checkpoint (num_frames, alpha, size, ...).
    device : str
        Device string used for inference ("cuda" or "cpu").
    last_latency_ms : float
        Latency of the most recent inference call (excluding clip loading).
    """

    def __init__(self, checkpoint_path: str = "/c6/checkpoints/slowfast_fighting_binary.pth", device: str = "auto"):
        # Make path resolution robust (handles absolute Unix path vs relative project path vs Windows paths)
        if not os.path.isfile(checkpoint_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            clean_path = checkpoint_path.replace("\\", "/").lstrip("/")
            if clean_path.startswith("c6/"):
                clean_path = clean_path[3:]
            # Try relative to the module file directory
            alt_path = os.path.join(base_dir, clean_path)
            if os.path.isfile(alt_path):
                checkpoint_path = alt_path
            else:
                # Try relative to the project root (parent of c6/)
                proj_root = os.path.dirname(base_dir)
                alt_path_proj = os.path.join(proj_root, checkpoint_path.replace("\\", "/").lstrip("/"))
                if os.path.isfile(alt_path_proj):
                    checkpoint_path = alt_path_proj
                else:
                    # Also try relative to project root without the c6 prefix if it had one
                    alt_path_proj_noc6 = os.path.join(proj_root, clean_path)
                    if os.path.isfile(alt_path_proj_noc6):
                        checkpoint_path = alt_path_proj_noc6

        self.model, self.config = load_binary_model(checkpoint_path, device)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_latency_ms = 0.0
        # PackPathway uses the same alpha as used during training (stored in config)
        self.pack_pathway = PackPathway(alpha=self.config.get("alpha", 4))

    def classify_clip(self, video_path: str, start_frame: int = 0) -> Dict:
        """Classify a video clip as Fighting or Normal.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        start_frame : int, default 0
            Frame index to start reading from.

        Returns
        -------
        dict
            ``{"label": str, "confidence": float, "raw_score": float, "latency_ms": float}``
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load clip using the exact number of frames the checkpoint expects
        num_frames = self.config.get("num_frames", 32)
        clip = load_clip(video_path, start_frame, num_frames=num_frames, transform_variant="kinetics")
        # Expected shape from load_clip: (T, C, H, W). Rearrange to (C, T, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Build SlowFast pathways
        pathways = self.pack_pathway(clip)  # list of two tensors [slow, fast]
        # Add batch dimension to each pathway
        inputs = [p.unsqueeze(0).to(self.device) for p in pathways]

        # Inference (measure only forward pass)
        torch.cuda.synchronize() if self.device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(inputs)
        torch.cuda.synchronize() if self.device == "cuda" else None
        end = time.perf_counter()
        self.last_latency_ms = (end - start) * 1000.0

        # Model returns a tensor of shape (batch, 1)
        raw_score = torch.sigmoid(logits).squeeze().item()
        label = "Fighting" if raw_score >= 0.5 else "Normal"
        confidence = raw_score if label == "Fighting" else 1.0 - raw_score

        return {
            "label": label,
            "confidence": confidence,
            "raw_score": raw_score,
            "latency_ms": self.last_latency_ms,
        }

    def get_last_latency_ms(self) -> float:
        """Return the latency (in milliseconds) of the most recent inference call."""
        return self.last_latency_ms

# ---------------------------------------------------------------------------
# Simple command‑line demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python binary_action_classifier.py <video_path> [start_frame]")
        sys.exit(1)
    video_path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    classifier = BinaryActionClassifier()
    try:
        result = classifier.classify_clip(video_path, start)
        import json
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Error during classification: {exc}")
        sys.exit(1)
