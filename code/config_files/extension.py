"""Validated JSON overrides shared by all inherited dataset configurations."""
import json
import math
from pathlib import Path


DEFAULTS = dict(encoder="original", custom_loss_enabled=False, custom_loss_weight=0.0,
                dinov3_repo="../dinov3-official", dinov3_model="dinov3_vits16",
                dinov3_pretrained=False, dinov3_weights=None, dinov3_freeze=True,
                dinov3_image_size=224)


def configure(config, path=None):
    values = dict(DEFAULTS)
    if path:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(overrides) - set(DEFAULTS) - {"batch_size", "target_batch_size", "num_epoch", "lr"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        values.update(overrides)
    if values["encoder"] not in {"original", "dinov3"}:
        raise ValueError("encoder must be original or dinov3")
    for key in ("custom_loss_enabled", "dinov3_pretrained", "dinov3_freeze"):
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    weight = values["custom_loss_weight"]
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("custom_loss_weight must be finite and nonnegative")
    for key, value in values.items():
        setattr(config, key, value)
    return config
