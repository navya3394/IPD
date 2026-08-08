import torch
import os
from typing import Tuple, Dict

class PackPathway:
    """Create SlowFast pathways.

    Parameters
    ----------
    alpha: int, default 4
        Temporal stride between frames for the slow pathway.
    """
    def __init__(self, alpha: int = 4):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor) -> list:
        """Split a tensor into slow and fast pathways.

        Args:
            x: Tensor of shape (C, T, H, W).
        Returns:
            List containing [slow_pathway, fast_pathway] where:
                - fast_pathway is the original tensor (all frames).
                - slow_pathway samples every ``alpha``‑th frame along the temporal dimension.
        """
        # Fast pathway keeps all frames
        fast_pathway = x
        # Compute indices for slow pathway
        T = x.shape[1]
        indices = torch.linspace(0, T - 1, steps=T // self.alpha, dtype=torch.long)
        slow_pathway = torch.index_select(x, dim=1, index=indices)
        return [slow_pathway, fast_pathway]

def load_binary_model(checkpoint_path: str, device: str = "auto") -> Tuple[torch.nn.Module, Dict]:
    """Load a SlowFast R50 model fine‑tuned for binary classification.

    The checkpoint is expected to contain the keys:
        - ``model_state``: state_dict for the model.
        - ``num_frames``, ``alpha``, ``size``: configuration used during training.
        - ``val_acc``, ``epoch`` (optional, returned in the config dict).

    Parameters
    ----------
    checkpoint_path: str
        Path to ``slowfast_fighting_binary.pth``.
    device: str, default "auto"
        "auto" selects CUDA if available, otherwise CPU.

    Returns
    -------
    model: torch.nn.Module
        The SlowFast model ready for inference (eval mode).
    config: dict
        Dictionary with at least ``num_frames``, ``alpha``, ``size``.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required_keys = ["model_state", "num_frames", "alpha", "size"]
    for k in required_keys:
        if k not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: {k}")

    # Load base SlowFast R50 without pretrained weights (we will load the checkpoint state)
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load SlowFast model via torch.hub: {e}")

    # Replace final projection layer with a single‑output linear layer
    # The original head layout in slowfast_r50 is model.blocks[-1].proj
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = torch.nn.Linear(in_features, 1)

    # Load checkpoint state dict
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint state_dict: {e}")

    # Device handling
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    config = {k: checkpoint[k] for k in ["num_frames", "alpha", "size"]}
    # Preserve optional information for debugging / logging
    for opt_key in ["val_acc", "epoch"]:
        if opt_key in checkpoint:
            config[opt_key] = checkpoint[opt_key]
    return model, config
