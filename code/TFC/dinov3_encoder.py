"""Time-series adapters around the official Meta DINOv3 ViT implementation."""
from pathlib import Path
import importlib
import sys

import torch
from torch import nn
from torch.nn import functional as F
from model import TFC


class GramianAdapter(nn.Module):
    """Map [B,1,L] to a bounded GASF image; no FFT is performed here."""

    def __init__(self, image_size=224):
        super().__init__()
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of ViT patch size 16")
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] < 2:
            raise ValueError("Expected [batch, 1, length>=2], as produced by TF-C dataloader")
        if not torch.isfinite(x).all():
            raise ValueError("Adapter input contains NaN/Inf")
        x = F.interpolate(x.float(), size=self.image_size, mode="linear", align_corners=False)
        centered = x - x.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        u = torch.tanh(centered / scale) * .999
        v = (1 - u.square()).sqrt()
        gaf = u.transpose(1, 2) @ u - v.transpose(1, 2) @ v
        rgb = ((gaf[:, None] + 1) / 2).expand(-1, 3, -1, -1)
        return (rgb - self.mean) / self.std


class DINOv3Encoder(nn.Module):
    """Official local Hub entry point, followed by a trainable feature adapter."""

    def __init__(self, config):
        super().__init__()
        repo = Path(config.dinov3_repo).expanduser().resolve()
        if not (repo / "hubconf.py").is_file():
            raise ValueError("dinov3_repo must point to a clone of facebookresearch/dinov3")
        if config.dinov3_model not in {"dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"}:
            raise ValueError("Supported official backbones: dinov3_vits16/vitb16/vitl16")
        kwargs = {"pretrained": config.dinov3_pretrained}
        if config.dinov3_pretrained:
            if not config.dinov3_weights:
                raise ValueError("Pretrained mode requires an authorized local checkpoint path")
            weights = Path(config.dinov3_weights).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(weights)
            kwargs["weights"] = str(weights)
        # Import the exact official Hub factory without hubconf's unrelated task heads.
        # Current hubconf eagerly imports segmentation/detection dependencies.
        sys.path.insert(0, str(repo))
        try:
            factories = importlib.import_module("dinov3.hub.backbones")
            if Path(factories.__file__).resolve() != repo / "dinov3" / "hub" / "backbones.py":
                raise RuntimeError("A different dinov3 checkout is already imported; restart Python")
            self.backbone = getattr(factories, config.dinov3_model)(**kwargs)
        finally:
            sys.path.remove(str(repo))
        self.adapter = GramianAdapter(config.dinov3_image_size)
        self.feature_projection = nn.Linear(self.backbone.embed_dim, config.TSlength_aligned)
        self.set_frozen(config.dinov3_freeze)

    def set_frozen(self, frozen):
        self.frozen = bool(frozen)
        self.backbone.requires_grad_(not self.frozen)
        self.backbone.train(self.training and not self.frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        # Keep autograd through input even with frozen backbone parameters.
        features = self.backbone.forward_features(self.adapter(x))["x_norm_clstoken"]
        return self.feature_projection(features)


class DINOv3TFC(TFC):
    """Reuse TF-C forward contract without allocating discarded Transformers."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.transformer_encoder_t = DINOv3Encoder(config)
        self.transformer_encoder_f = DINOv3Encoder(config)
        self.projector_t = self._projector(config.TSlength_aligned)
        self.projector_f = self._projector(config.TSlength_aligned)

    @staticmethod
    def _projector(length):
        return nn.Sequential(nn.Linear(length, 256), nn.BatchNorm1d(256),
                             nn.ReLU(), nn.Linear(256, 128))


def build_model(config):
    """Preserve original TFC class, projector names and return signature."""

    encoder = getattr(config, "encoder", "original")
    if encoder == "dinov3":
        return DINOv3TFC(config)
    if encoder == "original":
        return TFC(config)
    raise ValueError(f"Unknown encoder: {encoder}")
