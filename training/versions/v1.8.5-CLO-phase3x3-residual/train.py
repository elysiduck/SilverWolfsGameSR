from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gamesr.data import (
    GameSRDataset, charbonnier_loss, detail_score_per_image, edge_loss,
    gpu_game_degradation, high_frequency_loss, information_weight_per_image,
    multi_scale_ssim_loss, ringing_loss, ssim_loss,
)
from gamesr.metrics import distribution_summary, validation_batch_metrics
from gamesr.model import build_model
from gamesr.streaming import (
    BackgroundBatchLoader,
    CudaPrefetchLoader,
    PersistentValidationLoader,
    RamStreamingBatchLoader,
    should_use_streaming,
    should_use_validation_cache,
    streaming_settings,
)
from gamesr.utils import append_csv, checkpoint_signature, load_config, save_preview, seed_everything, seed_worker, write_json


def resolve_config(config: dict, config_path: Path) -> dict:
    config = dict(config)
    base = config_path.parent.parent
    for key in ("root",):
        if key in config["data"] and not Path(config["data"][key]).is_absolute():
            config["data"][key] = str((base / config["data"][key]).resolve())
    if not Path(config["output"]).is_absolute():
        config["output"] = str((base / config["output"]).resolve())
    return config


def apply_v182_late_polish_profile(config: dict) -> dict:
    """Migrate v1.8/v1.8.1 settings to the v1.8.2 late-polish profile in memory.

    Exact old defaults are upgraded; values the user already customized are
    preserved. This lets an existing v1.8 generated YAML auto-resume without
    requiring a fresh config file. Checkpoint architecture compatibility is
    unaffected because model/scale/version are unchanged.
    """
    tcfg = config.setdefault("train", {})
    alg = tcfg.setdefault("algorithm_v18", {})
    if not bool(alg.get("enabled", True)):
        return config

    def replace_if_old(mapping: dict, key: str, old_values: tuple[float, ...], new_value: float) -> None:
        if key not in mapping:
            mapping[key] = new_value
            return
        try:
            current = float(mapping[key])
        except (TypeError, ValueError):
            return
        if any(abs(current - old) < 1e-9 for old in old_values):
            mapping[key] = new_value

    replace_if_old(alg, "edge_boost", (0.80,), 0.45)
    replace_if_old(alg, "ssim_boost", (0.75,), 0.50)
    replace_if_old(alg, "high_frequency_weight", (0.06, 0.04), 0.03)
    alg.setdefault("high_frequency_detail_aware", True)
    alg.setdefault("high_frequency_detail_floor", 0.15)
    alg.setdefault("high_frequency_detail_low", 0.006)
    alg.setdefault("high_frequency_detail_high", 0.050)
    alg.setdefault("game_aware_profile", "v1.8.1")

    dcfg = config.setdefault("data", {})
    hard = dcfg.setdefault("hard_patch_mining", {})
    replace_if_old(hard, "min_fraction", (0.15, 0.12), 0.10)
    replace_if_old(hard, "max_fraction", (0.40, 0.30), 0.25)
    replace_if_old(hard, "min_mix", (0.10, 0.08), 0.06)
    replace_if_old(hard, "max_mix", (0.35, 0.24), 0.20)
    hard.setdefault("weight_cap", 1.50)
    hard.setdefault("detail_floor", 0.45)
    hard.setdefault("detail_low", 0.006)
    hard.setdefault("detail_high", 0.030)

    curriculum = dcfg.setdefault("degradation_curriculum", {})
    replace_if_old(curriculum, "start_strength", (0.90,), 0.85)
    replace_if_old(curriculum, "full_strength_at", (0.25,), 0.35)

    degradation = dcfg.setdefault("degradation", {})
    replacements = {
        "blur_probability": ((0.12,), 0.10),
        "sharpen_probability": ((0.15, 0.18), 0.08),
        "post_sharpen_probability": ((0.16, 0.18), 0.08),
        "brightness_probability": ((0.08, 0.10), 0.03),
        "contrast_probability": ((0.12, 0.15), 0.05),
        "quantization_probability": ((0.08,), 0.025),
        "noise_probability": ((0.18,), 0.04),
    }
    for key, (olds, new_value) in replacements.items():
        replace_if_old(degradation, key, olds, new_value)
    degradation.setdefault("resize_kernels", ["bicubic", "bilinear", "area"])
    degradation.setdefault("resize_kernel_weights", [0.50, 0.40, 0.10])
    # v1.8.2 late polish / information-aware additions. Existing user-customized
    # v1.8.1 values are preserved; these keys did not exist previously.
    alg["game_aware_profile"] = "v1.8.2"
    alg.setdefault("polish_start", 0.50)
    alg.setdefault("polish_full_at", 0.75)
    alg.setdefault("polish_edge_weight", 0.0675)
    alg.setdefault("polish_ssim_weight", 0.030)
    alg.setdefault("polish_high_frequency_weight", 0.012)
    info = alg.setdefault("information_aware_l1", {})
    info.setdefault("enabled", True)
    info.setdefault("low_information_weight", 0.80)
    info.setdefault("high_information_weight", 1.15)
    info.setdefault("flat_black_weight", 0.80)
    info.setdefault("dark_mean_threshold", 0.12)
    info.setdefault("flat_std_threshold", 0.035)
    info.setdefault("min_weight", 0.75)
    info.setdefault("max_weight", 1.25)
    hard.setdefault("polish_fraction", 0.15)
    hard.setdefault("polish_mix", 0.09)
    config["trainer_version"] = "1.8.2"
    return config


def make_loader(
    config: dict,
    split: str,
    device: torch.device | None = None,
    *,
    reserved_training_cache_bytes: int = 0,
):
    dcfg = config["data"]
    train = split == "train"
    dataset = GameSRDataset(
        Path(dcfg["root"]) / f"{split}.csv",
        dcfg["hr_patch"],
        config["scale"],
        augment=train and bool(dcfg.get("augment", True)),
        deterministic_kernel=str(dcfg.get("val_resize_kernel", "lanczos")),
        return_lr=not train,
    )

    settings = streaming_settings(dcfg)

    # Remote training: persistent compressed-byte cache + rolling read-ahead.
    if train:
        use_streaming, remote_detected = should_use_streaming(dataset, settings)
        if use_streaming:
            loader = RamStreamingBatchLoader(
                dataset,
                int(dcfg["batch_size"]),
                shuffle=True,
                drop_last=bool(dcfg.get("drop_last", True)),
                seed=int(config.get("seed", 42)),
                read_ahead_samples=int(settings["read_ahead_samples"]),
                fetch_threads=int(settings["fetch_threads"]),
                decode_threads=int(settings["decode_threads"]),
                pin_memory=bool(settings["pin_memory"]),
                cache_budget_gib=settings.get("cache_budget_gib", "auto"),
                cache_reserve_gib=float(settings.get("cache_reserve_gib", 8.0)),
                cache_cap_gib=float(settings.get("cache_cap_gib", 12.0)),
                cache_auto_fraction=float(settings.get("cache_auto_fraction", 0.50)),
                cache_floor_gib=float(settings.get("cache_floor_gib", 1.0)),
                cache_protected_ratio=float(settings.get("cache_protected_ratio", 0.80)),
                emergency_free_gib=float(settings.get("emergency_free_gib", 4.0)),
                next_epoch_prefetch_samples=int(settings.get("next_epoch_prefetch_samples", 256)),
            )
            plan = loader.cache_plan
            print(
                f"[RAM] total={plan.get('total_gib', 0):.2f}GiB | currently available={plan.get('available_gib', 0):.2f}GiB | "
                f"cache={plan.get('budget_gib', 0):.2f}GiB ({plan.get('mode')}) | reserve={plan.get('reserve_gib', 0):.2f}GiB",
                flush=True,
            )
            loader = BackgroundBatchLoader(loader, queue_batches=int(settings.get("batch_queue", 4)))
            cuda_prefetch = bool(settings.get("cuda_prefetch", True)) and device is not None
            print(
                f"[STREAM] persistent RAM streaming ON | remote={remote_detected} | "
                f"window={settings['read_ahead_samples']} images | fetch_threads={settings['fetch_threads']} | "
                f"decode_threads={settings['decode_threads']} | batch_queue={settings.get('batch_queue', 4)} | "
                f"next_epoch={settings.get('next_epoch_prefetch_samples', 256)} | cuda_prefetch={cuda_prefetch}",
                flush=True,
            )
            print(
                "[STREAM] Cache stores compressed source bytes in RAM across epochs; "
                "nothing is copied to NVMe and RAM is released when training exits.",
                flush=True,
            )
            return CudaPrefetchLoader(loader, device) if cuda_prefetch else loader
        print(
            f"[STREAM] RAM read-ahead OFF | mode={settings.get('mode')} | remote={remote_detected}; "
            "using standard DataLoader.",
            flush=True,
        )

    # v1.7.6: deterministic validation samples are identical every epoch.  On a
    # remote dataset, cache the already decoded/center-cropped LR+HR tensors in
    # RAM. This removes both SMB and PIL/Lanczos work on later validation passes.
    if not train:
        use_val_cache, remote_detected = should_use_validation_cache(dataset, settings)
        if use_val_cache:
            loader = PersistentValidationLoader(
                dataset,
                batch_size=int(dcfg.get("val_batch_size", 16)),
                read_ahead_samples=int(settings.get("validation_read_ahead_samples", 96)),
                threads=int(settings.get("validation_threads", 12)),
                pin_memory=bool(settings.get("validation_pin_memory", True)),
                cache_budget_gib=settings.get("validation_cache_budget_gib", "auto"),
                cache_reserve_gib=float(settings.get("validation_cache_reserve_gib", 7.0)),
                cache_cap_gib=float(settings.get("validation_cache_cap_gib", 1.5)),
                cache_auto_fraction=float(settings.get("validation_cache_auto_fraction", 0.50)),
                cache_floor_mib=float(settings.get("validation_cache_floor_mib", 64.0)),
                emergency_free_gib=float(settings.get("emergency_free_gib", 4.0)),
                reserved_training_bytes=int(reserved_training_cache_bytes),
            )
            plan = loader.cache_plan
            print(
                f"[VALCACHE] deterministic tensor cache ON | remote={remote_detected} | "
                f"target={plan.get('budget_mib', 0):.0f}MiB / estimated_full={plan.get('estimated_mib', 0):.0f}MiB | "
                f"reserved_train={plan.get('reserved_training_gib', 0):.2f}GiB | "
                f"window={settings.get('validation_read_ahead_samples', 96)} | threads={settings.get('validation_threads', 12)}",
                flush=True,
            )
            return loader
        print(
            f"[VALCACHE] OFF | mode={settings.get('validation_mode')} | remote={remote_detected}; "
            "using standard validation DataLoader.",
            flush=True,
        )

    workers = int(dcfg.get("workers", 0))
    loader_kwargs = {
        "batch_size": int(dcfg["batch_size"] if train else dcfg.get("val_batch_size", 16)),
        "shuffle": train,
        "num_workers": workers,
        "pin_memory": bool(dcfg.get("pin_memory", True)),
        "drop_last": bool(train and dcfg.get("drop_last", True)),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(dcfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **loader_kwargs)


def _smoothstep01(value: float) -> float:
    x = min(1.0, max(0.0, float(value)))
    return x * x * (3.0 - 2.0 * x)


def build_v18_policy(epoch: int, epochs: int, tcfg: dict, dcfg: dict) -> dict:
    """Build the v1.8.x game-aware curriculum, including v1.8.2 polish."""
    cfg = dict(tcfg.get("algorithm_v18") or {})
    enabled = bool(cfg.get("enabled", True))
    progress = 1.0 if epochs <= 1 else (max(1, epoch) - 1) / max(1, epochs - 1)
    progress = min(1.0, max(0.0, progress))
    if not enabled:
        return {
            "enabled": False, "phase": "legacy", "progress": progress,
            "l1_weight": float(tcfg.get("l1_weight", 1.0)),
            "edge_weight": float(tcfg.get("edge_loss_weight", 0.05)),
            "ssim_weight": float(tcfg.get("ssim_loss_weight", 0.02)),
            "hf_weight": 0.0, "hf_every": 1,
            "hard_fraction": 0.0, "hard_mix": 0.0,
            "hard_weight_cap": 1.0, "hard_detail_floor": 1.0,
            "hard_detail_low": 0.006, "hard_detail_high": 0.030,
            "hf_detail_aware": False, "hf_detail_floor": 1.0,
            "hf_detail_low": 0.006, "hf_detail_high": 0.050,
            "degradation_strength": 1.0,
            "polish_ramp": 0.0, "information_strength": 0.0,
            "information_cfg": {},
        }

    foundation_end = min(0.80, max(0.01, float(cfg.get("foundation_end", 0.06))))
    detail_end = min(0.95, max(foundation_end + 0.01, float(cfg.get("detail_end", 0.30))))
    detail_ramp = _smoothstep01((progress - foundation_end) / max(1e-9, detail_end - foundation_end))
    refine_ramp = _smoothstep01((progress - detail_end) / max(1e-9, 1.0 - detail_end))

    polish_start = min(0.95, max(detail_end, float(cfg.get("polish_start", 0.50))))
    polish_full_at = min(1.0, max(polish_start + 0.01, float(cfg.get("polish_full_at", 0.75))))
    polish_ramp = _smoothstep01((progress - polish_start) / max(1e-9, polish_full_at - polish_start))
    if progress < foundation_end:
        phase = "foundation"
    elif progress < detail_end:
        phase = "detail"
    elif progress < polish_start:
        phase = "refine"
    else:
        phase = "polish"

    base_l1 = float(tcfg.get("l1_weight", 1.0))
    base_edge = float(tcfg.get("edge_loss_weight", 0.05))
    base_ssim = float(tcfg.get("ssim_loss_weight", 0.02))
    edge_boost = max(0.0, float(cfg.get("edge_boost", 0.45)))
    ssim_boost = max(0.0, float(cfg.get("ssim_boost", 0.50)))
    hf_max = max(0.0, float(cfg.get("high_frequency_weight", 0.03)))
    hf_ramp = min(1.0, 0.85 * detail_ramp + 0.15 * refine_ramp)

    edge_pre = base_edge * (1.0 + edge_boost * detail_ramp)
    ssim_pre = base_ssim * (1.0 + ssim_boost * detail_ramp)
    hf_pre = hf_max * hf_ramp
    edge_target = max(0.0, float(cfg.get("polish_edge_weight", 0.0675)))
    ssim_target = max(0.0, float(cfg.get("polish_ssim_weight", 0.030)))
    hf_target = max(0.0, float(cfg.get("polish_high_frequency_weight", 0.012)))
    edge_weight = edge_pre + (edge_target - edge_pre) * polish_ramp
    ssim_weight = ssim_pre + (ssim_target - ssim_pre) * polish_ramp
    hf_weight = hf_pre + (hf_target - hf_pre) * polish_ramp

    hard_cfg = dict(dcfg.get("hard_patch_mining") or {})
    hard_min = min(1.0, max(0.0, float(hard_cfg.get("min_fraction", 0.10))))
    hard_max = min(1.0, max(hard_min, float(hard_cfg.get("max_fraction", 0.25))))
    hard_pre = hard_min + (hard_max - hard_min) * (0.80 * detail_ramp + 0.20 * refine_ramp)
    mix_min = min(0.95, max(0.0, float(hard_cfg.get("min_mix", 0.06))))
    mix_max = min(0.95, max(mix_min, float(hard_cfg.get("max_mix", 0.20))))
    mix_pre = mix_min + (mix_max - mix_min) * (0.80 * detail_ramp + 0.20 * refine_ramp)
    polish_fraction = min(1.0, max(0.0, float(hard_cfg.get("polish_fraction", 0.15))))
    polish_mix = min(0.95, max(0.0, float(hard_cfg.get("polish_mix", 0.09))))
    hard_fraction = hard_pre + (polish_fraction - hard_pre) * polish_ramp
    hard_mix = mix_pre + (polish_mix - mix_pre) * polish_ramp

    degradation_cfg = dict(dcfg.get("degradation_curriculum") or {})
    degradation_start = min(1.0, max(0.0, float(degradation_cfg.get("start_strength", 0.85))))
    full_at = min(1.0, max(0.01, float(degradation_cfg.get("full_strength_at", 0.35))))
    degradation_strength = degradation_start + (1.0 - degradation_start) * _smoothstep01(progress / full_at)

    info_cfg = dict(cfg.get("information_aware_l1") or {})
    information_strength = polish_ramp if bool(info_cfg.get("enabled", True)) else 0.0

    return {
        "enabled": True, "phase": phase, "progress": progress, "l1_weight": base_l1,
        "edge_weight": edge_weight, "ssim_weight": ssim_weight, "hf_weight": hf_weight,
        "hf_every": max(1, int(cfg.get("high_frequency_every_n_steps", 2))),
        "hard_fraction": hard_fraction, "hard_mix": hard_mix,
        "hard_weight_cap": max(1.0, float(hard_cfg.get("weight_cap", 1.50))),
        "hard_detail_floor": min(1.0, max(0.0, float(hard_cfg.get("detail_floor", 0.45)))),
        "hard_detail_low": max(0.0, float(hard_cfg.get("detail_low", 0.006))),
        "hard_detail_high": max(0.000001, float(hard_cfg.get("detail_high", 0.030))),
        "hf_detail_aware": bool(cfg.get("high_frequency_detail_aware", True)),
        "hf_detail_floor": min(1.0, max(0.0, float(cfg.get("high_frequency_detail_floor", 0.15)))),
        "hf_detail_low": max(0.0, float(cfg.get("high_frequency_detail_low", 0.006))),
        "hf_detail_high": max(0.000001, float(cfg.get("high_frequency_detail_high", 0.050))),
        "degradation_strength": degradation_strength,
        "detail_ramp": detail_ramp, "refine_ramp": refine_ramp,
        "polish_ramp": polish_ramp, "information_strength": information_strength,
        "information_cfg": info_cfg,
    }


def hard_mined_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fraction: float,
    mix: float,
    *,
    detail_floor: float = 0.45,
    detail_low: float = 0.006,
    detail_high: float = 0.030,
    weight_cap: float = 1.50,
    information_strength: float = 0.0,
    information_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """v1.8.2 information-aware L1 + detail-aware online hard mining.

    The base L1 remains active on every patch. During polish, its per-patch
    contribution is gently reweighted by *target information value*; dark+flat
    cinematic bars are de-emphasized but never removed. Hard ranking remains
    reconstruction-error × target-detail. The final selected-sample multiplier
    is capped even after information weighting.
    """
    per_image = (prediction - target).abs().mean(dim=(1, 2, 3))
    raw_base = per_image.mean()
    icfg = dict(information_cfg or {})
    info_weights, detail, flat_black = information_weight_per_image(
        target,
        strength=information_strength,
        detail_low=detail_low,
        detail_high=detail_high,
        low_information_weight=float(icfg.get("low_information_weight", 0.80)),
        high_information_weight=float(icfg.get("high_information_weight", 1.15)),
        flat_black_weight=float(icfg.get("flat_black_weight", 0.80)),
        dark_mean_threshold=float(icfg.get("dark_mean_threshold", 0.12)),
        flat_std_threshold=float(icfg.get("flat_std_threshold", 0.035)),
        min_weight=float(icfg.get("min_weight", 0.75)),
        max_weight=float(icfg.get("max_weight", 1.25)),
    )
    info_base = (per_image * info_weights).sum() / info_weights.sum().clamp_min(1e-8)
    mean_detail = detail.mean()
    flat_black_fraction = (flat_black > 0.25).float().mean()
    info_weight_std = info_weights.float().std(unbiased=False)
    if per_image.numel() <= 1 or fraction <= 0.0 or mix <= 0.0:
        return info_base, raw_base.detach(), info_base.detach(), raw_base.detach(), mean_detail, 0.0, flat_black_fraction, info_weight_std

    n = int(per_image.numel())
    k = max(1, min(n, int(math.ceil(n * float(fraction)))))
    floor = min(1.0, max(0.0, float(detail_floor)))
    detail_factor = floor + (1.0 - floor) * detail
    rank_score = per_image.detach() * detail_factor
    indices = torch.topk(rank_score, k=k, largest=True, sorted=False).indices
    hard_raw = per_image.index_select(0, indices).mean()
    hard_weights = info_weights.index_select(0, indices)
    hard = (per_image.index_select(0, indices) * hard_weights).sum() / hard_weights.sum().clamp_min(1e-8)

    requested_mix = min(0.95, max(0.0, float(mix)))
    cap = max(1.0, float(weight_cap))
    effective_mix = requested_mix

    # Exact per-selected-sample multiplier relative to uniform raw L1 is affine
    # in mix. Solve the cap analytically, including information weights.
    sum_all = info_weights.sum().clamp_min(1e-8)
    sum_hard = hard_weights.sum().clamp_min(1e-8)
    a = n * hard_weights / sum_all
    b = n * hard_weights / sum_hard
    growing = b > a + 1e-9
    if cap <= 1.0:
        effective_mix = 0.0
    elif bool(growing.any()):
        allowed = (cap - a[growing]) / (b[growing] - a[growing]).clamp_min(1e-8)
        cap_mix = float(allowed.min().clamp(0.0, 1.0).item())
        effective_mix = min(effective_mix, cap_mix)
    if bool((a >= cap).any()):
        effective_mix = 0.0

    mixed = info_base * (1.0 - effective_mix) + hard * effective_mix
    return mixed, raw_base.detach(), info_base.detach(), hard_raw.detach(), mean_detail, float(effective_mix), flat_black_fraction, info_weight_std


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        d = self.decay
        for key, value in current.items():
            shadow = self.shadow[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(d).add_(value.detach(), alpha=1.0 - d)
            else:
                shadow.copy_(value.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Keep checkpoints portable. EMA itself remains on the model device while
        # training, but serialized tensors are stored on CPU. The model is tiny,
        # so this copy is negligible and avoids device-specific checkpoints.
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        # Checkpoints are intentionally loaded with map_location="cpu". Do NOT
        # replace self.shadow with those CPU tensors: the live model/EMA was
        # already created on CUDA. Copy every saved value into the existing
        # shadow tensor so its current device and dtype are preserved.
        expected = set(self.shadow)
        received = set(state)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"EMA checkpoint keys mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        for key, target in self.shadow.items():
            saved = state[key].detach().to(device=target.device, dtype=target.dtype)
            target.copy_(saved)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool,
    limit: int,
    channels_last: bool,
    tcfg: dict,
):
    """Validate with per-image reconstruction, edge and baseline diagnostics.

    v1.7.7 does not change the model or training objective.  It only makes the
    validation report more useful: the legacy L1/PSNR/SSIM values stay exactly
    comparable, while Delta-PSNR, edge error and lower-tail PSNR expose whether
    a high mean PSNR is merely hiding easy validation samples.
    """
    model.eval()
    sums = {
        "l1": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "edge_error": 0.0,
        "pred_edge_strength": 0.0,
        "target_edge_strength": 0.0,
        "baseline_mse": 0.0,
        "baseline_psnr": 0.0,
        "detail_psnr": 0.0,
        "smooth_psnr": 0.0,
        "baseline_detail_psnr": 0.0,
        "detail_edge_error": 0.0,
        "oversharp_excess_ratio": 0.0,
        "undersharp_deficit_ratio": 0.0,
    }
    image_count = 0
    sample = None
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    for batch_index, (lr, hr) in enumerate(loader):
        if limit > 0 and batch_index >= limit:
            break
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        if channels_last:
            lr = lr.contiguous(memory_format=torch.channels_last)
            hr = hr.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            pred = model(lr)

        batch_metrics = validation_batch_metrics(pred, hr, lr)
        for key in sums:
            sums[key] += float(getattr(batch_metrics, key).sum().item())

        psnr_values.extend(float(value) for value in batch_metrics.psnr.detach().cpu().tolist())
        ssim_values.extend(float(value) for value in batch_metrics.ssim.detach().cpu().tolist())
        image_count += int(hr.shape[0])
        if sample is None:
            sample = (lr[0].detach(), pred[0].detach(), hr[0].detach())

    if image_count <= 0:
        raise RuntimeError("Validation loader returned no samples.")

    mean = {key: value / image_count for key, value in sums.items()}
    psnr_dist = distribution_summary(psnr_values)
    ssim_dist = distribution_summary(ssim_values)
    edge_ratio = mean["pred_edge_strength"] / max(mean["target_edge_strength"], 1e-12)
    mse_reduction_pct = 100.0 * (1.0 - mean["mse"] / max(mean["baseline_mse"], 1e-12))
    delta_psnr = mean["psnr"] - mean["baseline_psnr"]
    detail_delta_psnr = mean["detail_psnr"] - mean["baseline_detail_psnr"]
    ssim_loss_value = 1.0 - mean["ssim"]
    ssim_every = max(1, int(tcfg.get("ssim_every_n_steps", 4)))
    effective_ssim_weight = float(tcfg.get("ssim_loss_weight", 0.0)) / ssim_every
    val_objective = (
        float(tcfg.get("l1_weight", 1.0)) * mean["l1"]
        + float(tcfg.get("edge_loss_weight", 0.05)) * mean["edge_error"]
        + effective_ssim_weight * ssim_loss_value
    )

    metrics = {
        "val_loss": mean["l1"],  # compatibility: historical val_loss was L1
        "val_l1": mean["l1"],
        "val_edge": mean["edge_error"],
        "val_ssim_loss": ssim_loss_value,
        "val_objective": val_objective,
        "psnr": mean["psnr"],
        "ssim": mean["ssim"],
        "psnr_std": psnr_dist["std"],
        "psnr_p10": psnr_dist["p10"],
        "psnr_min": psnr_dist["min"],
        "ssim_p10": ssim_dist["p10"],
        "baseline_psnr": mean["baseline_psnr"],
        "delta_psnr": delta_psnr,
        "mse_reduction_pct": mse_reduction_pct,
        "edge_strength_ratio": edge_ratio,
        "detail_psnr": mean["detail_psnr"],
        "smooth_psnr": mean["smooth_psnr"],
        "baseline_detail_psnr": mean["baseline_detail_psnr"],
        "detail_delta_psnr": detail_delta_psnr,
        "detail_edge_error": mean["detail_edge_error"],
        "oversharp_excess_ratio": mean["oversharp_excess_ratio"],
        "undersharp_deficit_ratio": mean["undersharp_deficit_ratio"],
    }
    return metrics, sample


def load_quality_history(path: Path) -> list[dict[str, float]]:
    """Load the minimal historical metrics needed for marginal-gain reporting.

    The reader is intentionally tolerant of old schemas so v1.7.7 can be
    dropped onto an existing v1.7.x output directory without starting over.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    result: list[dict[str, float]] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    epoch = int(float(row.get("epoch", "")))
                    psnr = float(row.get("psnr", ""))
                    ssim = float(row.get("ssim", ""))
                except (TypeError, ValueError):
                    continue
                result.append({"epoch": float(epoch), "psnr": psnr, "ssim": ssim})
    except OSError:
        return []
    # Keep only the latest parseable row for each epoch. This also makes the
    # trend robust if a run was interrupted after CSV append and the epoch was
    # later repeated from the previous checkpoint.
    by_epoch = {int(item["epoch"]): item for item in result}
    return [by_epoch[key] for key in sorted(by_epoch)]


def quality_trend(history: list[dict[str, float]], epoch: int, psnr: float, ssim: float, window: int) -> dict[str, float | int | None]:
    window = max(1, int(window))
    candidates = [item for item in history if int(item["epoch"]) < epoch]
    previous = candidates[-1] if candidates else None
    target_epoch = epoch - window
    older = None
    for item in candidates:
        if int(item["epoch"]) <= target_epoch:
            older = item
        else:
            break
    if older is None and candidates:
        older = candidates[0]

    result: dict[str, float | int | None] = {
        "psnr_gain_1": None,
        "ssim_gain_1": None,
        "trend_epochs": 0,
        "psnr_gain_window": None,
        "ssim_gain_window": None,
        "psnr_gain_per_epoch": None,
    }
    if previous is not None:
        result["psnr_gain_1"] = psnr - float(previous["psnr"])
        result["ssim_gain_1"] = ssim - float(previous["ssim"])
    if older is not None:
        span = max(1, epoch - int(older["epoch"]))
        psnr_gain = psnr - float(older["psnr"])
        result["trend_epochs"] = span
        result["psnr_gain_window"] = psnr_gain
        result["ssim_gain_window"] = ssim - float(older["ssim"])
        result["psnr_gain_per_epoch"] = psnr_gain / span
    return result


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, scheduler, scaler,
                    ema: ModelEMA | None, best_psnr: float, config: dict) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "best_psnr": best_psnr,
        "config": config,
    }
    # A power loss or forced close must not destroy the last usable checkpoint.
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def build_optimizer(model: nn.Module, tcfg: dict):
    kwargs = dict(
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    try:
        return AdamW(model.parameters(), fused=True, **kwargs), True
    except (TypeError, RuntimeError):
        return AdamW(model.parameters(), **kwargs), False


def main() -> None:
    parser = argparse.ArgumentParser(description="GameSR v1.8.2 late-polish + information-aware game SR trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-incompatible-resume", action="store_true", help="allow loading a checkpoint from a different training code version or architecture")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = apply_v182_late_polish_profile(resolve_config(load_config(config_path), config_path))
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be at least 1")
        config["data"]["batch_size"] = int(args.batch_size)

    seed_everything(int(config.get("seed", 42)))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable. Run check_env.py first.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tcfg = config["train"]
    amp = bool(tcfg.get("amp", True))
    channels_last = bool(tcfg.get("channels_last", True))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer, fused_optimizer = build_optimizer(model, tcfg)
    epochs = int(tcfg["epochs"])
    scheduler = build_scheduler(optimizer, epochs, int(tcfg.get("warmup_epochs", 5)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model, float(tcfg.get("ema_decay", 0.995))) if bool(tcfg.get("ema", True)) else None
    start_epoch, best_psnr = 1, -float("inf")

    resume_path = args.resume
    auto_path = output / "latest.pth"
    if resume_path is None and args.auto_resume and auto_path.exists():
        resume_path = str(auto_path)
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_config = state.get("config") or {}
        current_sig = checkpoint_signature(config)
        saved_sig = checkpoint_signature(saved_config)
        if current_sig != saved_sig and not args.allow_incompatible_resume:
            if args.auto_resume and args.resume is None:
                print("[RESUME] Existing checkpoint belongs to a different training configuration/version. Auto-resume skipped; starting a fresh run.")
                resume_path = None
            else:
                raise SystemExit("Checkpoint configuration/version is incompatible with the current run. Re-train from scratch or pass --allow-incompatible-resume explicitly.")
        if resume_path:
            print(f"Resuming from: {resume_path}")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            if ema is not None and state.get("ema") is not None:
                ema.load_state_dict(state["ema"])
            start_epoch = int(state["epoch"]) + 1
            best_psnr = float(state.get("best_psnr", best_psnr))
            if start_epoch > epochs:
                print("Training already reached configured epochs; nothing to do.")
                return

    print("[DATA] Loading train/val manifests (fast mode: no per-image existence re-scan)...", flush=True)
    manifest_start = time.perf_counter()
    train_loader = make_loader(config, "train", device=device)
    train_cache_plan = dict(getattr(train_loader, "cache_plan", {}) or {})
    val_loader = make_loader(
        config, "val", device=None,
        reserved_training_cache_bytes=int(train_cache_plan.get("budget_bytes", 0)),
    )
    print(
        f"[DATA] Ready: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"in {time.perf_counter() - manifest_start:.2f}s",
        flush=True,
    )
    writer = SummaryWriter(output / "runs")
    training_log_path = output / "training_log.csv"
    quality_history = load_quality_history(training_log_path)
    degradation_cfg = config["data"].get("degradation", {})
    v18_cfg = dict(tcfg.get("algorithm_v18") or {})
    hard_cfg = dict(config["data"].get("hard_patch_mining") or {})
    phase_loss_cfg = dict(tcfg.get("phase_aware_loss") or {})
    phase_loss_enabled = bool(phase_loss_cfg.get("enabled", False))

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Parameters: {parameter_count:,}")
    streaming_active = bool(getattr(train_loader, "stats_snapshot", None))
    worker_label = "RAM-streaming threads" if streaming_active else str(config['data'].get('workers', 0))
    print(f"Batch: {config['data']['batch_size']} | workers: {worker_label} | AMP: {amp} | channels_last: {channels_last} | fused_AdamW: {fused_optimizer}")
    print("GPU monitor tip: Windows Task Manager -> change 3D graph to CUDA/Compute_0, or run: nvidia-smi -l 1")
    print(f"Train images: {len(train_loader.dataset)} | Val images: {len(val_loader.dataset)} | Val batch: {config['data'].get('val_batch_size', 16)}")
    bench_meta = output / "batch_selection.json"
    if bench_meta.exists():
        print(f"Batch benchmark: {bench_meta}")
    if bool(v18_cfg.get("enabled", True)):
        print("[V1.8.2] Game-aware + late-polish ON: information-aware L1 + detail-hard + masked HF", flush=True)
        print(
            f"[V1.8.2] hard_fraction={float(hard_cfg.get('min_fraction', 0.10)):.2f}..{float(hard_cfg.get('max_fraction', 0.25)):.2f} | "
            f"hard_mix={float(hard_cfg.get('min_mix', 0.06)):.2f}..{float(hard_cfg.get('max_mix', 0.20)):.2f} cap={float(hard_cfg.get('weight_cap', 1.50)):.2f}x | "
            f"HF max={float(v18_cfg.get('high_frequency_weight', 0.03)):.3f} detail_mask={bool(v18_cfg.get('high_frequency_detail_aware', True))}", flush=True
        )
    if phase_loss_enabled:
        print(
            "[PHASE-AWARE LOSS] Charbonnier + MS-SSIM + gradient + frequency + ringing", flush=True
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        policy = build_v18_policy(epoch, epochs, tcfg, config["data"])
        phase_detail_ramp = 1.0
        if phase_loss_enabled:
            ramp_start = float(phase_loss_cfg.get("detail_ramp_start", 0.08))
            ramp_full = max(ramp_start + 1e-6, float(phase_loss_cfg.get("detail_ramp_full_at", 0.40)))
            ramp_x = min(1.0, max(0.0, (policy["progress"] - ramp_start) / (ramp_full - ramp_start)))
            phase_detail_ramp = ramp_x * ramp_x * (3.0 - 2.0 * ramp_x)
        total_loss = total_l1 = total_info_l1 = total_edge = total_ssim = total_hf = total_hard_l1 = total_ringing = 0.0
        total_detail_score = total_effective_hard_mix = total_flat_black_fraction = total_info_weight_std = 0.0
        ssim_evaluations = hf_evaluations = 0
        h2d_seconds = 0.0
        compute_seconds = 0.0
        wait_seconds = 0.0
        loop_start = time.perf_counter()
        last_batch_end = loop_start
        gpu_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        print(f"[EPOCH {epoch}] Starting DataLoader; waiting for first batch...", flush=True)
        print(
            f"[V1.8.2] phase={policy['phase']} progress={100.0 * policy['progress']:.0f}% "
            f"loss=L1:{policy['l1_weight']:.3f} Edge:{policy['edge_weight']:.3f} "
            f"SSIM:{policy['ssim_weight']:.3f} HF:{policy['hf_weight']:.3f} | "
            f"hard_top={100.0 * policy['hard_fraction']:.0f}% mix={100.0 * policy['hard_mix']:.0f}% "
            f"cap={policy['hard_weight_cap']:.2f}x detail_floor={policy['hard_detail_floor']:.2f} "
            f"degrade={100.0 * policy['degradation_strength']:.0f}% polish={100.0 * policy['polish_ramp']:.0f}% "
            f"info={100.0 * policy['information_strength']:.0f}%", flush=True
        )
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        first_batch_wall = time.perf_counter()
        for step, hr in enumerate(progress, start=1):
            if step == 1:
                print(f"[EPOCH {epoch}] First batch arrived after {time.perf_counter() - first_batch_wall:.2f}s", flush=True)
            now = time.perf_counter()
            wait_seconds += now - last_batch_end
            h2d_start = now
            # CudaPrefetchLoader already copied the next batch on a dedicated
            # CUDA stream.  The fallback loader still uses non-blocking H2D here.
            if not hr.is_cuda:
                hr = hr.to(device, non_blocking=True)
            if channels_last:
                hr = hr.contiguous(memory_format=torch.channels_last)
            h2d_seconds += time.perf_counter() - h2d_start

            # CUDA events give GPU timing without a blocking synchronize on every step.
            # v1.7.2 synchronized before and after each step, which prevented normal
            # CPU/DataLoader/H2D/GPU overlap and artificially lowered throughput.
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()

            lr = gpu_game_degradation(hr, config["scale"], degradation_cfg, strength=policy["degradation_strength"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred = model(lr)
                if phase_loss_enabled:
                    epsilon = float(phase_loss_cfg.get("charbonnier_epsilon", 0.001))
                    per_image_charb = torch.sqrt((pred - hr).square() + epsilon * epsilon).mean(dim=(1, 2, 3))
                    loss_l1 = per_image_charb.mean()
                    raw_l1 = (pred - hr).abs().mean().detach()
                    info_l1 = loss_l1.detach()
                    hard_l1 = raw_l1
                    detail_per_image = detail_score_per_image(hr)
                    detail_score = detail_per_image.mean().detach()
                    effective_hard_mix = 0.0
                    flat_black_fraction = pred.new_zeros(())
                    info_weight_std = pred.new_zeros(())
                    if bool(hard_cfg.get("enabled", True)) and per_image_charb.numel() > 1:
                        hard_fraction = min(1.0, max(0.0, float(policy["hard_fraction"])))
                        requested_mix = min(0.25, max(0.0, float(policy["hard_mix"])))
                        hard_count = max(1, min(per_image_charb.numel(), int(math.ceil(per_image_charb.numel() * hard_fraction))))
                        rank = per_image_charb.detach() * (0.45 + 0.55 * detail_per_image)
                        indices = torch.topk(rank, k=hard_count, largest=True, sorted=False).indices
                        hard_charb = per_image_charb.index_select(0, indices).mean()
                        loss_l1 = loss_l1 * (1.0 - requested_mix) + hard_charb * requested_mix
                        hard_l1 = (pred - hr).abs().mean(dim=(1, 2, 3)).index_select(0, indices).mean().detach()
                        effective_hard_mix = requested_mix
                else:
                    loss_l1, raw_l1, info_l1, hard_l1, detail_score, effective_hard_mix, flat_black_fraction, info_weight_std = hard_mined_l1(
                        pred, hr, policy["hard_fraction"], policy["hard_mix"],
                        detail_floor=policy["hard_detail_floor"],
                        detail_low=policy["hard_detail_low"],
                        detail_high=policy["hard_detail_high"],
                        weight_cap=policy["hard_weight_cap"],
                        information_strength=policy["information_strength"],
                        information_cfg=policy["information_cfg"],
                    )
                loss_edge = edge_loss(pred, hr)
                ssim_every = max(1, int(
                    phase_loss_cfg.get("ms_ssim_every_n_steps", 1)
                    if phase_loss_enabled else tcfg.get("ssim_every_n_steps", 4)
                ))
                ssim_computed = step % ssim_every == 0 and (
                    float(phase_loss_cfg.get("ms_ssim_weight", 0.10)) > 0
                    if phase_loss_enabled else policy["ssim_weight"] > 0
                )
                if ssim_computed:
                    loss_ssim = (
                        multi_scale_ssim_loss(pred.float(), hr.float())
                        if phase_loss_enabled else ssim_loss(pred.float(), hr.float())
                    )
                    ssim_evaluations += 1
                else:
                    loss_ssim = pred.new_zeros(())
                hf_every = max(1, int(
                    phase_loss_cfg.get("frequency_every_n_steps", 1)
                    if phase_loss_enabled else policy["hf_every"]
                ))
                hf_computed = step % hf_every == 0 and (
                    float(phase_loss_cfg.get("frequency_weight", 0.02)) > 0
                    if phase_loss_enabled else policy["hf_weight"] > 0
                )
                if hf_computed:
                    loss_hf = high_frequency_loss(
                        pred.float(), hr.float(),
                        detail_aware=policy["hf_detail_aware"],
                        detail_floor=policy["hf_detail_floor"],
                        detail_low=policy["hf_detail_low"],
                        detail_high=policy["hf_detail_high"],
                    )
                    hf_evaluations += 1
                else:
                    loss_hf = pred.new_zeros(())
                loss_ringing = ringing_loss(pred, hr) if phase_loss_enabled else pred.new_zeros(())
                if phase_loss_enabled:
                    # Reconstruction is dominant at the start, when the model
                    # is already an exact bilinear baseline. Bring perceptual
                    # and sharpening objectives in only after it is stable.
                    loss = (
                        float(phase_loss_cfg.get("charbonnier_weight", 1.0)) * loss_l1
                        + phase_detail_ramp * float(phase_loss_cfg.get("ms_ssim_weight", 0.10)) * loss_ssim
                        + (0.25 + 0.75 * phase_detail_ramp) * float(phase_loss_cfg.get("gradient_weight", 0.05)) * loss_edge
                        + phase_detail_ramp * float(phase_loss_cfg.get("frequency_weight", 0.02)) * loss_hf
                        + phase_detail_ramp * float(phase_loss_cfg.get("ringing_weight", 0.01)) * loss_ringing
                    )
                else:
                    loss = (
                        policy["l1_weight"] * loss_l1 + policy["edge_weight"] * loss_edge
                        + policy["ssim_weight"] * loss_ssim + policy["hf_weight"] * loss_hf
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = float(tcfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            gpu_end.record()
            gpu_events.append((gpu_start, gpu_end))

            total_loss += float(loss.item())
            total_l1 += float(raw_l1.item())
            total_info_l1 += float(info_l1.item())
            total_hard_l1 += float(hard_l1.item())
            total_detail_score += float(detail_score.item())
            total_effective_hard_mix += float(effective_hard_mix)
            total_flat_black_fraction += float(flat_black_fraction.item())
            total_info_weight_std += float(info_weight_std.item())
            total_edge += float(loss_edge.detach().item())
            total_ringing += float(loss_ringing.detach().item())
            if ssim_computed:
                total_ssim += float(loss_ssim.detach().item())
            if hf_computed:
                total_hf += float(loss_hf.detach().item())
            step_elapsed = max(time.perf_counter() - loop_start, 1e-6)
            throughput = (step * hr.shape[0]) / step_elapsed
            progress.set_postfix(loss=f"{loss.item():.5f}", img_s=f"{throughput:.0f}")
            last_batch_end = time.perf_counter()

        # One synchronization per epoch is enough to resolve all CUDA event timings.
        torch.cuda.synchronize()
        compute_seconds = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000.0

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        train_l1 = total_l1 / max(len(train_loader), 1)
        train_info_l1 = total_info_l1 / max(len(train_loader), 1)
        train_hard_l1 = total_hard_l1 / max(len(train_loader), 1)
        train_detail_score = total_detail_score / max(len(train_loader), 1)
        train_effective_hard_mix = total_effective_hard_mix / max(len(train_loader), 1)
        train_flat_black_fraction = total_flat_black_fraction / max(len(train_loader), 1)
        train_info_weight_std = total_info_weight_std / max(len(train_loader), 1)
        train_edge = total_edge / max(len(train_loader), 1)
        train_ssim_loss = total_ssim / max(ssim_evaluations, 1)
        train_hf_loss = total_hf / max(hf_evaluations, 1)
        train_ringing_loss = total_ringing / max(len(train_loader), 1)
        elapsed = time.perf_counter() - loop_start
        stats_getter = getattr(train_loader, "stats_snapshot", None)
        stream_stats = stats_getter() if stats_getter else {}
        if stream_stats.get("enabled"):
            stream_stats["read_mib_per_sec_wall"] = round(
                float(stream_stats.get("read_mib", 0.0)) / max(elapsed, 1e-6), 1
            )

        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            ema.copy_to(model)
        val_start = time.perf_counter()
        val_metrics, sample = validate(
            model, val_loader, device, amp, int(tcfg.get("val_batches", 0)), channels_last, tcfg
        )
        val_seconds = time.perf_counter() - val_start
        val_loss = float(val_metrics["val_loss"])
        psnr = float(val_metrics["psnr"])
        ssim_value = float(val_metrics["ssim"])
        trend = quality_trend(
            quality_history, epoch, psnr, ssim_value, int(tcfg.get("quality_trend_window", 5))
        )
        val_stats_getter = getattr(val_loader, "stats_snapshot", None)
        val_cache_stats = val_stats_getter() if val_stats_getter else {}
        model.load_state_dict(raw_state, strict=True)

        images = len(train_loader.dataset)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "train_l1": round(train_l1, 8),
            "train_info_l1": round(train_info_l1, 8),
            "train_hard_l1": round(train_hard_l1, 8),
            "train_edge": round(train_edge, 8),
            "train_ssim_loss": round(train_ssim_loss, 8),
            "train_high_frequency_loss": round(train_hf_loss, 8),
            "train_ringing_loss": round(train_ringing_loss, 8),
            "train_detail_score": round(train_detail_score, 6),
            "train_flat_black_fraction": round(train_flat_black_fraction, 6),
            "train_info_weight_std": round(train_info_weight_std, 6),
            "v18_effective_hard_mix": round(train_effective_hard_mix, 6),
            "v18_phase": policy["phase"],
            "v18_edge_weight": round(policy["edge_weight"], 6),
            "v18_ssim_weight": round(policy["ssim_weight"], 6),
            "v18_high_frequency_weight": round(policy["hf_weight"], 6),
            "v18_hard_fraction": round(policy["hard_fraction"], 6),
            "v18_hard_mix": round(policy["hard_mix"], 6),
            "v18_hard_weight_cap": round(policy["hard_weight_cap"], 6),
            "v18_degradation_strength": round(policy["degradation_strength"], 6),
            "v182_polish_ramp": round(policy["polish_ramp"], 6),
            "v182_information_strength": round(policy["information_strength"], 6),
            "val_loss": round(val_loss, 8),
            "psnr": round(psnr, 6),
            "ssim": round(ssim_value, 6),
            "val_l1": round(float(val_metrics["val_l1"]), 8),
            "val_edge": round(float(val_metrics["val_edge"]), 8),
            "val_ssim_loss": round(float(val_metrics["val_ssim_loss"]), 8),
            "val_objective": round(float(val_metrics["val_objective"]), 8),
            "baseline_psnr": round(float(val_metrics["baseline_psnr"]), 6),
            "delta_psnr": round(float(val_metrics["delta_psnr"]), 6),
            "mse_reduction_pct": round(float(val_metrics["mse_reduction_pct"]), 4),
            "psnr_std": round(float(val_metrics["psnr_std"]), 6),
            "psnr_p10": round(float(val_metrics["psnr_p10"]), 6),
            "psnr_min": round(float(val_metrics["psnr_min"]), 6),
            "ssim_p10": round(float(val_metrics["ssim_p10"]), 6),
            "edge_strength_ratio": round(float(val_metrics["edge_strength_ratio"]), 6),
            "detail_psnr": round(float(val_metrics["detail_psnr"]), 6),
            "smooth_psnr": round(float(val_metrics["smooth_psnr"]), 6),
            "baseline_detail_psnr": round(float(val_metrics["baseline_detail_psnr"]), 6),
            "detail_delta_psnr": round(float(val_metrics["detail_delta_psnr"]), 6),
            "detail_edge_error": round(float(val_metrics["detail_edge_error"]), 8),
            "oversharp_excess_ratio": round(float(val_metrics["oversharp_excess_ratio"]), 6),
            "undersharp_deficit_ratio": round(float(val_metrics["undersharp_deficit_ratio"]), 6),
            "psnr_gain_1": "" if trend["psnr_gain_1"] is None else round(float(trend["psnr_gain_1"]), 6),
            "psnr_gain_window": "" if trend["psnr_gain_window"] is None else round(float(trend["psnr_gain_window"]), 6),
            "psnr_gain_per_epoch": "" if trend["psnr_gain_per_epoch"] is None else round(float(trend["psnr_gain_per_epoch"]), 6),
            "quality_trend_epochs": int(trend["trend_epochs"] or 0),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(elapsed, 3),
            "loader_wait_seconds": round(wait_seconds, 3),
            "h2d_enqueue_seconds": round(h2d_seconds, 3),
            "h2d_and_degradation_seconds": round(h2d_seconds, 3),
            "compute_seconds": round(compute_seconds, 3),
            "gpu_stream_busy_pct": round(100.0 * compute_seconds / max(elapsed, 1e-6), 1),
            "images_per_sec": round(images / max(elapsed, 1e-6), 3),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "batch_size": int(config["data"]["batch_size"]),
            "stream_read_mib": float(stream_stats.get("read_mib", 0.0)),
            "stream_read_mib_s": float(stream_stats.get("read_mib_per_sec_wall", 0.0)),
            "stream_prefetch_hit_pct": float(stream_stats.get("prefetch_hit_pct", 0.0)),
            "stream_avg_ready_queue": float(stream_stats.get("avg_ready_queue", 0.0)),
            "stream_avg_batch_build_ms": float(stream_stats.get("avg_batch_build_ms", 0.0)),
            "stream_cache_hit_pct": float(stream_stats.get("cache_hit_pct", 0.0)),
            "stream_cache_mib": float(stream_stats.get("cache_mib", 0.0)),
            "stream_cache_budget_mib": float(stream_stats.get("cache_budget_mib", 0.0)),
            "stream_cache_entries": int(stream_stats.get("cache_entries", 0)),
            "stream_cache_evictions": int(stream_stats.get("cache_evictions", 0)),
            "stream_cache_pressure_evictions": int(stream_stats.get("cache_pressure_evictions", 0)),
            "stream_system_available_gib": float(stream_stats.get("system_available_gib", 0.0)),
            "stream_next_epoch_prewarm": int(stream_stats.get("next_epoch_prewarm_submitted", 0)),
            "validation_seconds": round(val_seconds, 3),
            "val_cache_read_mib": float(val_cache_stats.get("read_mib", 0.0)),
            "val_cache_network_service_mib_s": float(val_cache_stats.get("network_service_mib_s", 0.0)),
            "val_cache_decode_seconds": float(val_cache_stats.get("decode_service_seconds", 0.0)),
            "val_cache_hit_pct": float(val_cache_stats.get("cache_hit_pct", 0.0)),
            "val_cache_mib": float(val_cache_stats.get("val_cache_mib", 0.0)),
            "val_cache_budget_mib": float(val_cache_stats.get("val_cache_budget_mib", 0.0)),
            "val_cache_entries": int(val_cache_stats.get("val_cache_entries", 0)),
            "val_cache_pressure_evictions": int(val_cache_stats.get("val_cache_pressure_evictions", 0)),
            "val_cache_system_available_gib": float(val_cache_stats.get("val_system_available_gib", 0.0)),
        }
        append_csv(training_log_path, row)
        for key in (
            "train_loss", "train_l1", "train_info_l1", "train_hard_l1", "train_edge", "train_ssim_loss", "train_high_frequency_loss", "train_ringing_loss", "train_detail_score", "train_flat_black_fraction", "train_info_weight_std",
            "v18_edge_weight", "v18_ssim_weight", "v18_high_frequency_weight",
            "v18_hard_fraction", "v18_hard_mix", "v18_effective_hard_mix", "v18_hard_weight_cap", "v18_degradation_strength", "v182_polish_ramp", "v182_information_strength",
            "val_loss", "psnr", "ssim", "val_l1", "val_edge", "val_ssim_loss",
            "val_objective", "baseline_psnr", "delta_psnr", "mse_reduction_pct", "psnr_std",
            "psnr_p10", "psnr_min", "ssim_p10", "edge_strength_ratio", "detail_psnr", "smooth_psnr", "baseline_detail_psnr", "detail_delta_psnr", "detail_edge_error", "oversharp_excess_ratio", "undersharp_deficit_ratio", "images_per_sec",
            "loader_wait_seconds", "h2d_and_degradation_seconds", "compute_seconds", "gpu_stream_busy_pct",
        ):
            writer.add_scalar(key, row[key], epoch)
        for key in ("psnr_gain_1", "psnr_gain_window", "psnr_gain_per_epoch"):
            if row[key] != "":
                writer.add_scalar(key, row[key], epoch)
        writer.add_scalar("lr", row["lr"], epoch)
        writer.add_scalar("peak_vram_mib", row["peak_vram_mib"], epoch)
        if stream_stats.get("enabled"):
            for key in (
                "stream_read_mib_s", "stream_prefetch_hit_pct", "stream_avg_ready_queue",
                "stream_avg_batch_build_ms", "stream_cache_hit_pct", "stream_cache_mib",
                "stream_cache_evictions", "stream_system_available_gib",
            ):
                writer.add_scalar(key, row[key], epoch)

        if val_cache_stats.get("enabled"):
            for key in (
                "val_cache_read_mib", "val_cache_network_service_mib_s", "val_cache_decode_seconds",
                "val_cache_hit_pct", "val_cache_mib", "val_cache_entries",
                "val_cache_system_available_gib",
            ):
                writer.add_scalar(key, row[key], epoch)

        if sample and epoch % int(tcfg.get("preview_every", 2)) == 0:
            save_preview(*sample, output / "preview" / f"epoch-{epoch:04d}.png")

        is_best = psnr > best_psnr
        if is_best:
            best_psnr = psnr

        history_for_best = quality_history + [{"epoch": float(epoch), "psnr": psnr, "ssim": ssim_value}]
        best_entry = max(history_for_best, key=lambda item: float(item["psnr"]))
        quality_summary = {
            "version": "1.8.2",
            "epoch": epoch,
            "current": {
                "psnr_db": round(psnr, 6),
                "ssim": round(ssim_value, 6),
                "val_l1": round(float(val_metrics["val_l1"]), 8),
                "val_edge": round(float(val_metrics["val_edge"]), 8),
                "val_objective": round(float(val_metrics["val_objective"]), 8),
                "bilinear_psnr_db": round(float(val_metrics["baseline_psnr"]), 6),
                "delta_psnr_db": round(float(val_metrics["delta_psnr"]), 6),
                "mse_reduction_pct": round(float(val_metrics["mse_reduction_pct"]), 4),
                "psnr_p10_db": round(float(val_metrics["psnr_p10"]), 6),
                "psnr_min_db": round(float(val_metrics["psnr_min"]), 6),
                "psnr_std_db": round(float(val_metrics["psnr_std"]), 6),
                "ssim_p10": round(float(val_metrics["ssim_p10"]), 6),
                "edge_strength_ratio": round(float(val_metrics["edge_strength_ratio"]), 6),
                "detail_psnr_db": round(float(val_metrics["detail_psnr"]), 6),
                "smooth_psnr_db": round(float(val_metrics["smooth_psnr"]), 6),
                "baseline_detail_psnr_db": round(float(val_metrics["baseline_detail_psnr"]), 6),
                "detail_delta_psnr_db": round(float(val_metrics["detail_delta_psnr"]), 6),
                "detail_edge_error": round(float(val_metrics["detail_edge_error"]), 8),
                "oversharp_excess_ratio": round(float(val_metrics["oversharp_excess_ratio"]), 6),
                "undersharp_deficit_ratio": round(float(val_metrics["undersharp_deficit_ratio"]), 6),
            },
            "trend": trend,
            "best_psnr": {
                "epoch": int(best_entry["epoch"]),
                "psnr_db": round(float(best_entry["psnr"]), 6),
                "ssim": round(float(best_entry["ssim"]), 6),
            },
            "training_algorithm_v18": {
                "phase": policy["phase"],
                "progress": round(float(policy["progress"]), 6),
                "edge_weight": round(float(policy["edge_weight"]), 6),
                "ssim_weight": round(float(policy["ssim_weight"]), 6),
                "high_frequency_weight": round(float(policy["hf_weight"]), 6),
                "hard_fraction": round(float(policy["hard_fraction"]), 6),
                "hard_mix_requested": round(float(policy["hard_mix"]), 6),
                "hard_mix_effective_avg": round(float(train_effective_hard_mix), 6),
                "hard_weight_cap": round(float(policy["hard_weight_cap"]), 6),
                "detail_score_avg": round(float(train_detail_score), 6),
                "polish_ramp": round(float(policy["polish_ramp"]), 6),
                "information_strength": round(float(policy["information_strength"]), 6),
                "information_l1_avg": round(float(train_info_l1), 8),
                "flat_black_fraction_avg": round(float(train_flat_black_fraction), 6),
                "information_weight_std_avg": round(float(train_info_weight_std), 6),
                "detail_aware_hard_mining": True,
                "detail_aware_high_frequency": bool(policy["hf_detail_aware"]),
                "degradation_strength": round(float(policy["degradation_strength"]), 6),
            },
            "notes": {
                "val_loss": "Historical GameSR val_loss is L1 only. val_objective remains a legacy reconstruction diagnostic; v1.8.2 late-polish/information-aware curriculum is reported separately because its weights change by epoch.",
                "edge_strength_ratio": "1.0 means predicted edge energy matches target on average; below 1 tends softer, above 1 tends sharper. It is diagnostic, not a score to maximize.",
                "delta_psnr_db": "GameSR PSNR minus bilinear-upsample PSNR on the exact same validation pairs.",
                "detail_psnr_db": "PSNR weighted toward target high-detail regions; compare detail_delta_psnr_db against the bilinear baseline.",
                "oversharp_excess_ratio": "Positive predicted edge excess divided by target edge energy. Lower is better; interpret together with undersharp_deficit_ratio.",
            },
        }
        write_json(output / "quality_summary.json", quality_summary)
        quality_history.append({"epoch": float(epoch), "psnr": psnr, "ssim": ssim_value})

        save_start = time.perf_counter()
        save_checkpoint(output / "latest.pth", epoch, model, optimizer, scheduler, scaler, ema, best_psnr, config)
        if is_best:
            save_checkpoint(output / "best.pth", epoch, model, optimizer, scheduler, scaler, ema, best_psnr, config)
        if epoch % int(tcfg.get("save_every", 10)) == 0:
            save_checkpoint(output / f"epoch-{epoch:04d}.pth", epoch, model, optimizer, scheduler, scaler, ema, best_psnr, config)
        save_seconds = time.perf_counter() - save_start

        print(
            f"train={train_loss:.5f} val={val_loss:.5f} PSNR={psnr:.2f} dB SSIM={ssim_value:.4f} "
            f"time={elapsed:.1f}s img/s={row['images_per_sec']:.1f} wait={wait_seconds:.1f}s "
            f"gpu_stage={compute_seconds:.1f}s gpu_stream_busy~{row['gpu_stream_busy_pct']:.0f}% "
            f"val_time={val_seconds:.1f}s save_time={save_seconds:.1f}s peakVRAM={row['peak_vram_mib']:.0f}MiB"
        )
        print(
            f"[V1.8.2 LOSS] RawL1={train_l1:.5f} InfoL1={train_info_l1:.5f} HardL1={train_hard_l1:.5f} Edge={train_edge:.5f} "
            f"SSIMLoss={train_ssim_loss:.5f} HFLoss={train_hf_loss:.5f} Ring={train_ringing_loss:.5f} | phase={policy['phase']} "
            f"hard_top={100.0 * policy['hard_fraction']:.0f}% mix_req={100.0 * policy['hard_mix']:.0f}% "
            f"mix_eff={100.0 * train_effective_hard_mix:.1f}% detail={train_detail_score:.3f} "
            f"info={100.0 * policy['information_strength']:.0f}% flat_black={100.0 * train_flat_black_fraction:.1f}% "
            f"degrade={100.0 * policy['degradation_strength']:.0f}%"
        )
        print(
            f"[QUALITY] bilinear={val_metrics['baseline_psnr']:.2f}dB "
            f"delta={val_metrics['delta_psnr']:+.2f}dB MSE_reduction={val_metrics['mse_reduction_pct']:+.1f}% "
            f"P10={val_metrics['psnr_p10']:.2f}dB min={val_metrics['psnr_min']:.2f}dB "
            f"edge_err={val_metrics['val_edge']:.5f} edge_ratio={val_metrics['edge_strength_ratio']:.3f} "
            f"detail={val_metrics['detail_psnr']:.2f}dB dDelta={val_metrics['detail_delta_psnr']:+.2f}dB "
            f"smooth={val_metrics['smooth_psnr']:.2f}dB over={val_metrics['oversharp_excess_ratio']:.3f} "
            f"under={val_metrics['undersharp_deficit_ratio']:.3f} val_obj={val_metrics['val_objective']:.5f}"
        )
        if trend["psnr_gain_window"] is not None:
            print(
                f"[TREND] last {int(trend['trend_epochs'])} epoch(s): "
                f"PSNR {float(trend['psnr_gain_window']):+.3f}dB "
                f"({float(trend['psnr_gain_per_epoch']):+.3f}dB/epoch), "
                f"SSIM {float(trend['ssim_gain_window']):+.5f}; "
                f"best={float(best_entry['psnr']):.2f}dB@epoch{int(best_entry['epoch'])}"
            )
        if stream_stats.get("enabled"):
            cache_gib = float(stream_stats.get("cache_mib", 0.0)) / 1024.0
            cache_budget_gib = float(stream_stats.get("cache_budget_mib", 0.0)) / 1024.0
            print(
                f"[STREAM] net_read={stream_stats.get('read_mib', 0):.1f}MiB "
                f"avg_net={stream_stats.get('read_mib_per_sec_wall', 0):.1f}MiB/s "
                f"net_service={stream_stats.get('network_service_mib_s', 0):.1f}MiB/s "
                f"cache_hit={stream_stats.get('cache_hit_pct', 0):.1f}% "
                f"cache={cache_gib:.2f}/{cache_budget_gib:.2f}GiB "
                f"prefetch_hit={stream_stats.get('prefetch_hit_pct', 0):.1f}% "
                f"ready_q={stream_stats.get('avg_ready_queue', 0):.1f} "
                f"batch_build={stream_stats.get('avg_batch_build_ms', 0):.1f}ms"
            )
            print(
                f"[RAM] available_now={stream_stats.get('system_available_gib', 0):.2f}GiB "
                f"cache_entries={stream_stats.get('cache_entries', 0)} "
                f"evict={stream_stats.get('cache_evictions', 0)} "
                f"pressure_evict={stream_stats.get('cache_pressure_evictions', 0)} "
                f"next_epoch_prewarm={stream_stats.get('next_epoch_prewarm_submitted', 0)}"
            )
        if val_cache_stats.get("enabled"):
            print(
                f"[VALCACHE] net_read={val_cache_stats.get('read_mib', 0):.1f}MiB "
                f"net_service={val_cache_stats.get('network_service_mib_s', 0):.1f}MiB/s "
                f"decode={val_cache_stats.get('decode_service_seconds', 0):.1f}s "
                f"cache_hit={val_cache_stats.get('cache_hit_pct', 0):.1f}% "
                f"cache={val_cache_stats.get('val_cache_mib', 0):.0f}/{val_cache_stats.get('val_cache_budget_mib', 0):.0f}MiB "
                f"entries={val_cache_stats.get('val_cache_entries', 0)} "
                f"available={val_cache_stats.get('val_system_available_gib', 0):.2f}GiB"
            )
        torch.cuda.reset_peak_memory_stats()

    for loader in (train_loader, val_loader):
        close_loader = getattr(loader, "close", None)
        if close_loader:
            close_loader()
    writer.close()
    print(json.dumps({"status": "ok", "output": str(output), "best_psnr": best_psnr, "batch_size": config["data"]["batch_size"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
