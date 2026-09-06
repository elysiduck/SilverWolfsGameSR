from __future__ import annotations

import csv
import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


KERNELS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}


def read_manifest(path: Path) -> list[Path]:
    """Read a prepared dataset manifest without re-statting every image.

    The dataset preparation/scanning stage already validates source images.  Older
    GameSR versions called ``Path.is_file()`` for every row every time a Dataset
    object was constructed.  On Windows/SMB this can cost minutes and made stages
    [2/8] and [3/8] look frozen while CPU/GPU stayed almost idle.

    Set GAMESR_VALIDATE_MANIFEST_FILES=1 when a full existence re-check is
    explicitly wanted.  Normal training uses lexical absolute paths only.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "path" not in rows[0]:
        raise ValueError(f"Manifest is missing the 'path' column: {path}")

    files: list[Path] = []
    for row in rows:
        raw = (row.get("path") or "").strip()
        if not raw:
            continue
        item = Path(raw).expanduser()
        if not item.is_absolute():
            item = path.parent / item
        # os.path.abspath is lexical and avoids filesystem resolution/stat calls.
        files.append(Path(os.path.abspath(os.fspath(item))))

    if os.environ.get("GAMESR_VALIDATE_MANIFEST_FILES", "0").strip().lower() in {"1", "true", "yes", "on"}:
        missing = [item for item in files if not item.is_file()]
        if missing:
            sample = "\n".join(f"  - {item}" for item in missing[:5])
            raise FileNotFoundError(
                f"Manifest {path} references {len(missing)} missing image(s). First entries:\n{sample}"
            )
    return files


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


class GameSRDataset(Dataset):
    """HR screenshot dataset.

    v1.3 moves synthetic degradation to the GPU during training. The CPU only
    decodes and crops the HR screenshot, substantially reducing per-sample CPU
    work for tiny models.
    """

    def __init__(
        self,
        manifest: Path,
        hr_patch: int,
        scale: float,
        augment: bool,
        deterministic_kernel: str = "lanczos",
        return_lr: bool = False,
    ):
        self.files = read_manifest(manifest)
        if not self.files:
            raise ValueError(f"No images in {manifest}")
        self.hr_patch = int(hr_patch)
        self.scale = float(scale)
        self.lr_patch = round(self.hr_patch / self.scale)
        if abs(self.lr_patch * self.scale - self.hr_patch) > 1e-6:
            raise ValueError("hr_patch must produce an integer LR patch")
        self.augment = bool(augment)
        self.deterministic_kernel = KERNELS.get(deterministic_kernel, Image.Resampling.LANCZOS)
        self.return_lr = bool(return_lr)

    def __len__(self) -> int:
        return len(self.files)

    def _crop(self, image: Image.Image, rng: random.Random | None = None) -> Image.Image:
        w, h = image.size
        if w < self.hr_patch or h < self.hr_patch:
            ratio = max(self.hr_patch / w, self.hr_patch / h)
            image = image.resize((round(w * ratio), round(h * ratio)), Image.Resampling.BICUBIC)
            w, h = image.size
        if self.augment:
            source_rng = rng if rng is not None else random
            x = source_rng.randint(0, w - self.hr_patch)
            y = source_rng.randint(0, h - self.hr_patch)
        else:
            x = (w - self.hr_patch) // 2
            y = (h - self.hr_patch) // 2
        return image.crop((x, y, x + self.hr_patch, y + self.hr_patch))

    def read_source_bytes(self, index: int) -> bytes:
        """Read one compressed source image as bytes.

        The RAM streaming loader calls this from a dedicated network I/O thread
        pool so SMB file opens/reads can overlap with GPU work and with each
        other.  No local-disk cache is created.
        """
        with self.files[index].open("rb") as handle:
            return handle.read()

    def decode_hr_bytes(self, payload: bytes, rng: random.Random | None = None) -> torch.Tensor:
        """Decode/crop an already-prefetched image held in RAM."""
        with Image.open(io.BytesIO(payload)) as source:
            hr_image = self._crop(source.convert("RGB"), rng=rng)
        return to_tensor(hr_image)

    def decode_validation_pair_bytes(self, payload: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one deterministic validation sample from compressed bytes.

        Validation uses a center crop and a fixed resize kernel, so the LR/HR pair
        is identical on every epoch.  v1.7.6 can therefore keep this processed
        pair in RAM and skip both SMB I/O and PIL resize/decode on later epochs.
        """
        with Image.open(io.BytesIO(payload)) as source:
            hr_image = self._crop(source.convert("RGB"), rng=None)
        hr = to_tensor(hr_image)
        lr_image = hr_image.resize((self.lr_patch, self.lr_patch), self.deterministic_kernel)
        return to_tensor(lr_image), hr

    def __getitem__(self, index: int):
        # Standard DataLoader path.  The streaming path uses read_source_bytes +
        # decode_hr_bytes so file I/O can run well ahead of the current batch.
        with Image.open(self.files[index]) as source:
            hr_image = self._crop(source.convert("RGB"))
        hr = to_tensor(hr_image)
        if not self.return_lr:
            return hr
        lr_image = hr_image.resize((self.lr_patch, self.lr_patch), self.deterministic_kernel)
        return to_tensor(lr_image), hr


def _blur3(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def _luma(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def detail_score_per_image(
    target: torch.Tensor,
    *,
    low: float = 0.006,
    high: float = 0.030,
) -> torch.Tensor:
    """Estimate useful game-detail density for each HR patch.

    The score is computed from *target* luminance gradients, so naturally blurred
    DOF/Bloom/cinematic patches stay low while UI/text/geometry patches rise.
    Values are normalized to [0, 1] using absolute thresholds rather than a
    per-batch ranking, preventing an all-blurry batch from manufacturing a
    "high detail" winner.
    """
    y = _luma(target.float())
    dx = (y[..., :, 1:] - y[..., :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (y[..., 1:, :] - y[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    energy = 0.5 * (dx + dy)
    lo = float(low)
    hi = max(lo + 1e-6, float(high))
    return ((energy - lo) / (hi - lo)).clamp(0.0, 1.0)



def information_weight_per_image(
    target: torch.Tensor,
    *,
    strength: float = 1.0,
    detail_low: float = 0.006,
    detail_high: float = 0.030,
    low_information_weight: float = 0.80,
    high_information_weight: float = 1.15,
    flat_black_weight: float = 0.80,
    dark_mean_threshold: float = 0.12,
    flat_std_threshold: float = 0.035,
    min_weight: float = 0.75,
    max_weight: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized per-patch weights for late GameSR polishing.

    v1.8.2 keeps every training patch, but spends slightly less gradient on
    low-information patches and slightly more on target patches containing
    useful game detail.  Nearly-black *and* flat patches (cinematic bars / empty
    panels) receive an additional conservative reduction.  Black UI panels with
    text are protected because their variance/detail score is high.

    ``strength`` is curriculum-controlled: 0 reproduces uniform v1.8.1 L1;
    1 applies the full information-aware profile. Weights are mean-normalized so
    the average L1 gradient scale stays close to the old trainer.
    """
    strength = min(1.0, max(0.0, float(strength)))
    detail = detail_score_per_image(target, low=detail_low, high=detail_high).detach()
    if strength <= 0.0:
        ones = torch.ones_like(detail)
        return ones, detail, torch.zeros_like(detail)

    y = _luma(target.float())
    mean_luma = y.mean(dim=(1, 2, 3))
    std_luma = y.flatten(1).std(dim=1, unbiased=False)

    # Soft confidence that a patch is an uninformative dark+flat region.
    # detail gating prevents subtitles/white text on black from being suppressed.
    dark = ((float(dark_mean_threshold) - mean_luma) / max(float(dark_mean_threshold), 1e-6)).clamp(0.0, 1.0)
    flat = ((float(flat_std_threshold) - std_luma) / max(float(flat_std_threshold), 1e-6)).clamp(0.0, 1.0)
    low_detail = (1.0 - detail).clamp(0.0, 1.0)
    flat_black = (dark * flat * low_detail).detach()

    lo_w = float(low_information_weight)
    hi_w = max(lo_w, float(high_information_weight))
    desired = lo_w + (hi_w - lo_w) * detail
    black_w = min(1.0, max(0.0, float(flat_black_weight)))
    desired = desired * (1.0 - (1.0 - black_w) * flat_black)

    # Ramp from uniform weighting, normalize, then apply conservative caps.
    weights = 1.0 + strength * (desired - 1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    min_w = max(0.05, float(min_weight))
    max_w = max(min_w, float(max_weight))
    # Clamp after normalization. The post-clamp mean remains very close to 1,
    # while the advertised absolute caps stay exact.
    weights = weights.clamp(min_w, max_w)
    return weights.detach(), detail, flat_black


def _detail_weight_map(
    target: torch.Tensor,
    *,
    floor: float = 0.15,
    low: float = 0.006,
    high: float = 0.050,
) -> torch.Tensor:
    """Spatial target-detail mask used by the high-frequency objective.

    Smooth/intentional-blur regions keep a small floor weight so hallucinated
    ringing is still penalized, while genuine target edges and texture receive
    most of the HF supervision.
    """
    y = _luma(target.float())
    dx = F.pad((y[..., :, 1:] - y[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((y[..., 1:, :] - y[..., :-1, :]).abs(), (0, 0, 0, 1))
    local = _blur3(0.5 * (dx + dy))
    lo = float(low)
    hi = max(lo + 1e-6, float(high))
    detail = ((local - lo) / (hi - lo)).clamp(0.0, 1.0)
    floor = min(1.0, max(0.0, float(floor)))
    return floor + (1.0 - floor) * detail


def _resize_game_tensor(x: torch.Tensor, size: tuple[int, int], kernel: str) -> torch.Tensor:
    kernel = str(kernel).lower()
    if kernel == "bicubic":
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    if kernel == "area":
        return F.interpolate(x, size=size, mode="area")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def gpu_game_degradation(
    hr: torch.Tensor,
    scale: float,
    cfg: dict | None = None,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Fast game-oriented degradation executed on GPU.

    v1.8.6 adds a stable curriculum mode for phase-residual SR. In that mode:
    - degradation probability really follows ``strength`` instead of keeping a
      high hidden floor;
    - amplitudes start mild and ramp gradually;
    - a clean resize anchor remains in every batch so robustness training never
      replaces the core 720p->1080p reconstruction task;
    - resize-kernel diversity is itself curriculum-controlled.

    Configs without ``curriculum_mode: stable_v2`` retain the legacy v1.8.5
    behaviour for compatibility with O3 / older experiments.
    """
    cfg = cfg or {}
    strength = min(1.0, max(0.0, float(strength)))
    lr_h = round(hr.shape[-2] / float(scale))
    lr_w = round(hr.shape[-1] / float(scale))
    target_size = (lr_h, lr_w)

    stable_mode = str(cfg.get("curriculum_mode", "legacy")).lower() == "stable_v2"
    if stable_mode:
        probability_scale = strength
        amplitude_scale = 0.30 + 0.70 * strength
        clean_kernel = str(cfg.get("clean_resize_kernel", "bicubic")).lower()
        clean_lr = _resize_game_tensor(hr, target_size, clean_kernel)
    else:
        # v1.8.5 resume-safe legacy floor.
        probability_scale = 0.60 + 0.40 * strength
        amplitude_scale = 0.70 + 0.30 * strength
        clean_kernel = "bicubic"
        clean_lr = None

    x = hr

    # Mild pre-downsample blur.
    blur_p = float(cfg.get("blur_probability", 0.10)) * probability_scale
    if blur_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < blur_p
        blurred = _blur3(x)
        x = torch.where(mask, blurred, x)

    # Mild unsharp-mask style pre-processing.
    sharp_p = float(cfg.get("sharpen_probability", 0.08)) * probability_scale
    if sharp_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < sharp_p
        amount = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            0.02 * amplitude_scale, 0.10 * amplitude_scale
        )
        soft = _blur3(x)
        sharp = (x + amount * (x - soft)).clamp(0.0, 1.0)
        x = torch.where(mask, sharp, x)

    # Resize-kernel diversity ramps with the curriculum in stable mode. Early
    # epochs therefore see a consistent clean mapping instead of full kernel
    # randomness from step one.
    kernels = [str(k).lower() for k in cfg.get("resize_kernels", ["bicubic", "bilinear", "area"])]
    weights = cfg.get("resize_kernel_weights", [0.50, 0.40, 0.10])
    if not isinstance(weights, (list, tuple)) or len(weights) != len(kernels):
        weights = [1.0] * len(kernels)
    weights = [max(0.0, float(w)) for w in weights]
    if stable_mode and random.random() > strength:
        kernel = clean_kernel
    else:
        kernel = random.choices(kernels, weights=weights, k=1)[0] if sum(weights) > 0 else random.choice(kernels)
    x = _resize_game_tensor(x, target_size, kernel)

    # Mild post-sharpen.
    post_p = float(cfg.get("post_sharpen_probability", 0.08)) * probability_scale
    if post_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < post_p
        amount = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            0.02 * amplitude_scale, 0.10 * amplitude_scale
        )
        soft = _blur3(x)
        sharp = (x + amount * (x - soft)).clamp(0.0, 1.0)
        x = torch.where(mask, sharp, x)

    # Very mild brightness/contrast perturbation.
    bright_p = float(cfg.get("brightness_probability", 0.03)) * probability_scale
    if bright_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < bright_p
        delta = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            -0.012 * amplitude_scale, 0.012 * amplitude_scale
        )
        bright = (x + delta).clamp(0.0, 1.0)
        x = torch.where(mask, bright, x)

    contrast_p = float(cfg.get("contrast_probability", 0.05)) * probability_scale
    if contrast_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < contrast_p
        factor = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            1.0 - 0.03 * amplitude_scale, 1.0 + 0.03 * amplitude_scale
        )
        contrast = ((x - 0.5) * factor + 0.5).clamp(0.0, 1.0)
        x = torch.where(mask, contrast, x)

    # Quantization approximates mild compression without a CPU JPEG roundtrip.
    quant_p = float(cfg.get("quantization_probability", 0.025)) * probability_scale
    if quant_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < quant_p
        steps = random.choice([192, 224, 240])
        quantized = torch.round(x * steps) / steps
        x = torch.where(mask, quantized, x)

    noise_p = float(cfg.get("noise_probability", 0.04)) * probability_scale
    if noise_p > 0:
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < noise_p
        std = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            (0.20 * amplitude_scale) / 255.0, (1.00 * amplitude_scale) / 255.0
        )
        noisy = (x + torch.randn_like(x) * std).clamp(0.0, 1.0)
        x = torch.where(mask, noisy, x)

    if stable_mode and clean_lr is not None:
        # Keep a large clean anchor even at full robustness. This prevents the
        # tiny network from trading away clean SR quality just to fit synthetic
        # degradation. The fraction ramps from start -> end as strength rises.
        clean_start = min(1.0, max(0.0, float(cfg.get("clean_anchor_start", 1.00))))
        clean_end = min(1.0, max(0.0, float(cfg.get("clean_anchor_end", 0.75))))
        clean_fraction = clean_start + (clean_end - clean_start) * strength
        if clean_fraction > 0:
            clean_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < clean_fraction
            x = torch.where(clean_mask, clean_lr, x)

    return x.clamp(0.0, 1.0)


def high_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    detail_aware: bool = True,
    detail_floor: float = 0.15,
    detail_low: float = 0.006,
    detail_high: float = 0.050,
) -> torch.Tensor:
    """Lightweight, target-aware high-frequency loss for game SR.

    v1.8.1 changes the v1.8 global HF objective into a target-detail-aware one.
    UI/text/fine geometry receive strong supervision; intentional DOF, Bloom and
    cinematic soft regions are down-weighted instead of being treated as defects.
    This is training-only and does not alter the exported ONNX graph.
    """
    pred_y = _luma(prediction.float())
    target_y = _luma(target.float())
    pred_high = pred_y - _blur3(pred_y)
    target_high = target_y - _blur3(target_y)
    error = (pred_high - target_high).abs()
    if not detail_aware:
        return error.mean()
    weights = _detail_weight_map(
        target, floor=detail_floor, low=detail_low, high=detail_high
    )
    return (error * weights).mean()


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = prediction[..., :, 1:] - prediction[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def charbonnier_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    """Robust reconstruction loss with a zero-valued perfect-match baseline."""
    eps = float(epsilon)
    difference = prediction - target
    return (torch.sqrt(difference * difference + eps * eps) - eps).mean()


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
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
    return 1.0 - ssim_map.clamp(-1.0, 1.0).mean()


def multi_scale_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
    weights: tuple[float, ...] = (0.50, 0.30, 0.20),
) -> torch.Tensor:
    """Small dependency-free multi-scale SSIM objective for game imagery."""
    if len(scales) != len(weights) or not scales:
        raise ValueError("MS-SSIM scales and weights must be non-empty and equal length")
    total = prediction.new_zeros(())
    weight_sum = 0.0
    for scale, weight in zip(scales, weights):
        if scale < 1 or weight <= 0:
            continue
        pred_level = prediction if scale == 1 else F.avg_pool2d(prediction, scale, scale)
        target_level = target if scale == 1 else F.avg_pool2d(target, scale, scale)
        total = total + float(weight) * ssim_loss(pred_level, target_level)
        weight_sum += float(weight)
    return total / max(weight_sum, 1e-8)


def ringing_loss(prediction: torch.Tensor, target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Penalize values outside the target's local envelope to suppress halos."""
    pad = int(kernel_size) // 2
    local_max = F.max_pool2d(target, kernel_size, stride=1, padding=pad)
    local_min = -F.max_pool2d(-target, kernel_size, stride=1, padding=pad)
    above = F.relu(prediction - local_max)
    below = F.relu(local_min - prediction)
    return (above + below).mean()
