from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class ValidationBatchMetrics:
    l1: torch.Tensor
    mse: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor
    edge_error: torch.Tensor
    pred_edge_strength: torch.Tensor
    target_edge_strength: torch.Tensor
    baseline_mse: torch.Tensor
    baseline_psnr: torch.Tensor
    detail_psnr: torch.Tensor
    smooth_psnr: torch.Tensor
    baseline_detail_psnr: torch.Tensor
    detail_edge_error: torch.Tensor
    oversharp_excess_ratio: torch.Tensor
    undersharp_deficit_ratio: torch.Tensor


def _luma(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def _target_detail_map(target: torch.Tensor, low: float = 0.006, high: float = 0.050) -> torch.Tensor:
    """Continuous [0,1] target-detail map used only for diagnostics."""
    y = _luma(target.float())
    dx = F.pad((y[..., :, 1:] - y[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((y[..., 1:, :] - y[..., :-1, :]).abs(), (0, 0, 0, 1))
    local = F.avg_pool2d(0.5 * (dx + dy), kernel_size=3, stride=1, padding=1)
    hi = max(float(low) + 1e-6, float(high))
    return ((local - float(low)) / (hi - float(low))).clamp(0.0, 1.0)


def _weighted_mse_per_image(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    error = (prediction.float() - target.float()).square()
    channels = prediction.shape[1]
    numerator = (error * weight).sum(dim=(1, 2, 3))
    denominator = (weight.sum(dim=(1, 2, 3)) * channels).clamp_min(1e-8)
    return (numerator / denominator).clamp_min(1e-12)


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Return SSIM for every image in an NCHW batch."""
    pad = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=pad) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return ssim_map.clamp(-1.0, 1.0).mean(dim=(1, 2, 3))


def edge_strength_per_image(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))


def edge_error_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = prediction[..., :, 1:] - prediction[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return (px - tx).abs().mean(dim=(1, 2, 3)) + (py - ty).abs().mean(dim=(1, 2, 3))


def detail_edge_error_per_image(prediction: torch.Tensor, target: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    px = prediction[..., :, 1:] - prediction[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    wx = 0.5 * (detail[..., :, 1:] + detail[..., :, :-1])
    wy = 0.5 * (detail[..., 1:, :] + detail[..., :-1, :])
    channels = prediction.shape[1]
    ex = ((px - tx).abs() * wx).sum(dim=(1, 2, 3)) / (wx.sum(dim=(1, 2, 3)) * channels).clamp_min(1e-8)
    ey = ((py - ty).abs() * wy).sum(dim=(1, 2, 3)) / (wy.sum(dim=(1, 2, 3)) * channels).clamp_min(1e-8)
    return ex + ey


def sharpness_balance_per_image(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive edge excess/deficit relative to target edge energy.

    0 is ideal for each direction. These are diagnostics, not scores to maximize.
    """
    py = _luma(prediction.float())
    ty = _luma(target.float())
    pdx = (py[..., :, 1:] - py[..., :, :-1]).abs()
    tdx = (ty[..., :, 1:] - ty[..., :, :-1]).abs()
    pdy = (py[..., 1:, :] - py[..., :-1, :]).abs()
    tdy = (ty[..., 1:, :] - ty[..., :-1, :]).abs()
    target_energy = tdx.sum(dim=(1, 2, 3)) + tdy.sum(dim=(1, 2, 3))
    excess_energy = F.relu(pdx - tdx).sum(dim=(1, 2, 3)) + F.relu(pdy - tdy).sum(dim=(1, 2, 3))
    deficit_energy = F.relu(tdx - pdx).sum(dim=(1, 2, 3)) + F.relu(tdy - pdy).sum(dim=(1, 2, 3))
    # Bounded 0..1 diagnostics. On a perfectly flat GT, any predicted edge is
    # correctly reported as 100% excess instead of exploding numerically.
    excess = excess_energy / (target_energy + excess_energy).clamp_min(1e-8)
    deficit = deficit_energy / (target_energy + deficit_energy).clamp_min(1e-8)
    return excess, deficit


def validation_batch_metrics(prediction: torch.Tensor, target: torch.Tensor, lr: torch.Tensor) -> ValidationBatchMetrics:
    """Compute exact per-image validation diagnostics without extra networks."""
    prediction = prediction.float()
    target = target.float()
    lr = lr.float()

    diff = prediction - target
    l1 = diff.abs().mean(dim=(1, 2, 3))
    mse = (diff * diff).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    ssim = ssim_per_image(prediction, target)
    edge_error = edge_error_per_image(prediction, target)
    pred_edge_strength = edge_strength_per_image(prediction)
    target_edge_strength = edge_strength_per_image(target)

    detail = _target_detail_map(target)
    # A small floor keeps per-image PSNR defined even on fully flat/cinematic
    # validation frames. These remain strongly detail/smooth weighted, not masks.
    detail_weight = 0.05 + 0.95 * detail
    smooth_weight = 0.05 + 0.95 * (1.0 - detail)
    detail_mse = _weighted_mse_per_image(prediction, target, detail_weight)
    smooth_mse = _weighted_mse_per_image(prediction, target, smooth_weight)
    detail_psnr = -10.0 * torch.log10(detail_mse)
    smooth_psnr = -10.0 * torch.log10(smooth_mse)
    detail_edge_error = detail_edge_error_per_image(prediction, target, detail)
    oversharp, undersharp = sharpness_balance_per_image(prediction, target)

    baseline = F.interpolate(lr, size=target.shape[-2:], mode="bilinear", align_corners=False)
    baseline_diff = baseline - target
    baseline_mse = (baseline_diff * baseline_diff).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    baseline_psnr = -10.0 * torch.log10(baseline_mse)
    baseline_detail_mse = _weighted_mse_per_image(baseline, target, detail_weight)
    baseline_detail_psnr = -10.0 * torch.log10(baseline_detail_mse)

    return ValidationBatchMetrics(
        l1=l1, mse=mse, psnr=psnr, ssim=ssim,
        edge_error=edge_error,
        pred_edge_strength=pred_edge_strength,
        target_edge_strength=target_edge_strength,
        baseline_mse=baseline_mse,
        baseline_psnr=baseline_psnr,
        detail_psnr=detail_psnr,
        smooth_psnr=smooth_psnr,
        baseline_detail_psnr=baseline_detail_psnr,
        detail_edge_error=detail_edge_error,
        oversharp_excess_ratio=oversharp,
        undersharp_deficit_ratio=undersharp,
    )


def distribution_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "min": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "min": float(tensor.min().item()),
    }
