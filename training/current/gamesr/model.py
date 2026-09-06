from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.residual_scale


class GameSR(nn.Module):
    """Compact 1.5x RGB GameSR network.

    The v1.4 default architecture does all learned feature extraction at LR
    resolution, then upsamples only the 3-channel residual. This reduces
    convolutional work by about scale^2 versus doing the residual stack at HR,
    while keeping the cheap bilinear base path.

    ``architecture='hr_residual'`` is retained for compatibility with older
    configurations/checkpoints when explicitly requested.
    """

    def __init__(
        self,
        channels: int = 8,
        blocks: int = 6,
        scale: float = 1.5,
        architecture: str = "lr_residual",
        conv_layers: int | None = None,
    ):
        super().__init__()
        self.scale = float(scale)
        self.architecture = str(architecture)
        if self.architecture == "lr_residual":
            self.head = nn.Conv2d(3, channels, 3, 1, 1)
            self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        elif self.architecture == "hr_residual":
            self.head = nn.Conv2d(3, channels, 3, 1, 1)
            self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        elif self.architecture == "npu_sequential":
            total_layers = int(conv_layers if conv_layers is not None else 8)
            if total_layers < 2:
                raise ValueError("npu_sequential requires conv_layers >= 2")
            self.conv_layers = total_layers
            self.head = nn.Conv2d(3, channels, 3, 1, 1)
            hidden = []
            for _ in range(total_layers - 2):
                hidden.extend((nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(inplace=True)))
            self.body = nn.Sequential(*hidden)
            self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        else:
            raise ValueError(f"Unknown GameSR architecture: {self.architecture}")

    def forward_core(self, lr: torch.Tensor) -> torch.Tensor:
        """Return only the learned low-resolution RGB residual.

        Ryzen AI deployment exports this method. Resize, base-image bypass,
        addition and clamp are deliberately left to the Magpie shader so the
        compiler sees one continuous Conv/ReLU graph.
        """
        if self.architecture in ("lr_residual", "npu_sequential"):
            feat = F.relu(self.head(lr), inplace=True)
            return self.tail(self.body(feat))
        raise RuntimeError("hr_residual has no low-resolution deployment core")

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)
        if self.architecture in ("lr_residual", "npu_sequential"):
            residual_lr = self.forward_core(lr)
            residual = F.interpolate(residual_lr, size=base.shape[-2:], mode="bilinear", align_corners=False)
        else:
            feat = F.relu(self.head(base), inplace=True)
            residual = self.tail(self.body(feat))
        return torch.clamp(base + residual, 0.0, 1.0)


def build_model(config: dict) -> GameSR:
    model_config = config.get("model", {})
    return GameSR(
        channels=int(model_config["channels"]),
        # Sequential NPU profiles do not use residual blocks.  Keep a default
        # so their intentionally minimal YAML configs can be loaded normally.
        blocks=int(model_config.get("blocks", 6)),
        scale=float(config.get("scale", 1.5)),
        architecture=str(model_config.get("architecture", "lr_residual")),
        conv_layers=model_config.get("conv_layers"),
    )
