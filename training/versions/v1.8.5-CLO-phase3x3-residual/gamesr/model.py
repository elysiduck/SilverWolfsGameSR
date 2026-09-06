from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PhaseAwareDecoder3x3(nn.Module):
    """Learned rational 3/2 decoder with nine spatial sub-pixel phases.

    For every 2x2 latent block the decoder emits a 3x3 RGB output block. Each
    output phase owns a learned 3x3 kernel, so spatial reconstruction is no
    longer determined by bilinear interpolation.
    """

    def __init__(self, input_channels: int, base_skip: float = 1.0):
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_skip = float(base_skip)
        self.weight = nn.Parameter(torch.empty(3, 3, 3, self.input_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(3, 3, 3))
        # Start as exact bilinear, then learn only the missing phase-dependent
        # detail.  A random direct-RGB branch makes a tiny model spend most of
        # its capacity recovering colour/low frequencies it already receives
        # for free from the base path.
        nn.init.zeros_(self.weight)

    def forward(self, latent: torch.Tensor, lr: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = latent.shape
        if h % 2 or w % 2:
            raise ValueError("phase-aware 3/2 decoder requires even latent height and width")
        padded = F.pad(latent, (1, 1, 1, 1), mode="replicate")
        phase_rgb = F.conv2d(
            padded,
            self.weight.view(27, self.input_channels, 3, 3),
            self.bias.view(27),
        ).view(b, 3, 3, 3, h, w)
        output = latent.new_empty((b, 3, h * 3 // 2, w * 3 // 2))
        for phase_y in range(3):
            source_y = 1 if phase_y == 2 else 0
            for phase_x in range(3):
                source_x = 1 if phase_x == 2 else 0
                output[:, :, phase_y::3, phase_x::3] = phase_rgb[
                    :, phase_y, phase_x, :, source_y::2, source_x::2
                ]
        if lr is not None and self.base_skip != 0.0:
            base = F.interpolate(lr, size=output.shape[-2:], mode="bilinear", align_corners=False)
            output = output + base * self.base_skip
        return output


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
        output_channels: int = 3,
        decoder_type: str = "bilinear_then_1x1",
        decoder_base_skip: float = 1.0,
    ):
        super().__init__()
        self.scale = float(scale)
        self.architecture = str(architecture)
        self.output_channels = int(output_channels)
        self.decoder_type = str(decoder_type)
        self.decoder_base_skip = float(decoder_base_skip)
        if self.output_channels < 3:
            raise ValueError("output_channels (O) must be >= 3")
        if self.architecture == "lr_residual":
            self.head = nn.Conv2d(3, channels, 3, 1, 1)
            self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.tail = nn.Conv2d(channels, self.output_channels, 3, 1, 1)
        elif self.architecture == "hr_residual":
            self.head = nn.Conv2d(3, channels, 3, 1, 1)
            self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.tail = nn.Conv2d(channels, self.output_channels, 3, 1, 1)
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
            self.tail = nn.Conv2d(channels, self.output_channels, 3, 1, 1)
        else:
            raise ValueError(f"Unknown GameSR architecture: {self.architecture}")

        # O3 keeps the original RGB-residual path exactly. O4/O6 use a tiny
        # learned decoder that is deliberately just a 1x1 channel mix, making
        # the later HLSL implementation a few dot products per output pixel.
        if self.output_channels == 3:
            self.decoder = None
        elif self.decoder_type == "phase_aware_3x3_v1":
            self.decoder = PhaseAwareDecoder3x3(self.output_channels, self.decoder_base_skip)
        elif self.decoder_type == "bilinear_then_1x1":
            self.decoder = nn.Conv2d(self.output_channels, 3, 1)
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

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
        if self.architecture in ("lr_residual", "npu_sequential"):
            latent_lr = self.forward_core(lr)
            if self.decoder_type == "phase_aware_3x3_v1" and self.decoder is not None:
                return torch.clamp(self.decoder(latent_lr, lr), 0.0, 1.0)
            base = F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)
            latent = F.interpolate(latent_lr, size=base.shape[-2:], mode="bilinear", align_corners=False)
            residual = self.decoder(latent) if self.decoder is not None else latent
        else:
            base = F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)
            feat = F.relu(self.head(base), inplace=True)
            residual = self.tail(self.body(feat))
        return torch.clamp(base + residual, 0.0, 1.0)


def build_model(config: dict) -> GameSR:
    model_config = config.get("model", {})
    deployment = config.get("deployment", {})
    decoder_config = deployment.get("decoder", {})
    return GameSR(
        channels=int(model_config["channels"]),
        # Sequential NPU profiles do not use residual blocks.  Keep a default
        # so their intentionally minimal YAML configs can be loaded normally.
        blocks=int(model_config.get("blocks", 6)),
        scale=float(config.get("scale", 1.5)),
        architecture=str(model_config.get("architecture", "lr_residual")),
        conv_layers=model_config.get("conv_layers"),
        output_channels=int(model_config.get("output_channels", 3)),
        decoder_type=str(deployment.get("postprocess", "bilinear_then_1x1")),
        decoder_base_skip=float(decoder_config.get("base_skip", 1.0)),
    )
