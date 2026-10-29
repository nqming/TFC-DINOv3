"""Strict weight loading, plus optimizer/epoch state for pretraining resume."""
import torch
from config_files.extension import DEFAULTS


def save_checkpoint(path, model, optimizer, epoch, config):
    metadata = {key: getattr(config, key, value) for key, value in DEFAULTS.items()}
    metadata["TSlength_aligned"] = config.TSlength_aligned
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "config": metadata}, path)


def load_checkpoint(path, model, device, config, optimizer=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = checkpoint.get("config", {})
    for key in ("encoder", "TSlength_aligned", "dinov3_model", "dinov3_image_size"):
        if key in metadata and metadata[key] != getattr(config, key, DEFAULTS.get(key)):
            raise ValueError(f"Checkpoint configuration mismatch: {key}")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Legacy weight-only checkpoint cannot resume optimizer; use it for fine-tuning")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))
