from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from gamesr.data import GameSRDataset, edge_loss, gpu_game_degradation, ssim_loss
from gamesr.model import build_model
from gamesr.utils import load_config, manifest_fingerprint, seed_everything, seed_worker, stable_json_hash, write_json

SCHEMA_VERSION = 7


def resolve_config(config: dict, config_path: Path) -> dict:
    config = dict(config)
    base = config_path.parent.parent
    if not Path(config["data"]["root"]).is_absolute():
        config["data"]["root"] = str((base / config["data"]["root"]).resolve())
    return config


def build_optimizer(model, config: dict):
    kwargs = dict(
        lr=float(config["train"].get("learning_rate", 2e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    # Fused AdamW reduces kernel-launch overhead for these extremely small CNNs.
    try:
        return AdamW(model.parameters(), fused=True, **kwargs), True
    except (TypeError, RuntimeError):
        return AdamW(model.parameters(), **kwargs), False


def make_probe_pool(config: dict, pool_size: int) -> torch.Tensor:
    """Decode/crop a small, fixed pool once instead of rereading the full dataset
    for every candidate batch size. This stage intentionally measures data loading
    only once; the batch sweep itself measures H2D + GPU degradation + full train step.
    """
    dcfg = config["data"]
    dataset = GameSRDataset(
        Path(dcfg["root"]) / "train.csv",
        dcfg["hr_patch"],
        config["scale"],
        augment=True,
        return_lr=False,
    )
    count = min(int(pool_size), len(dataset))
    if count < 1:
        raise RuntimeError("Training dataset is empty.")
    workers = min(2, max(0, int(dcfg.get("workers", 0))))
    kwargs = {
        "batch_size": min(32, count),
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": False,
        "drop_last": False,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **kwargs)
    chunks = []
    got = 0
    print(f"[POOL] Loading only {count} HR patches once for the quick benchmark (workers={workers})...")
    for batch in tqdm(loader, desc="Quick benchmark pool", total=max(1, (count + kwargs['batch_size'] - 1)//kwargs['batch_size'])):
        take = min(batch.shape[0], count - got)
        chunks.append(batch[:take].contiguous())
        got += take
        if got >= count:
            break
    pool = torch.cat(chunks, dim=0)
    if bool(dcfg.get("pin_memory", True)):
        try:
            pool = pool.pin_memory()
        except RuntimeError:
            pass
    print(f"[POOL] Ready: {pool.shape[0]} patches; reused for every batch-size candidate.")
    return pool


def get_batch(pool: torch.Tensor, batch_size: int, offset: int) -> torch.Tensor:
    n = pool.shape[0]
    if batch_size > n:
        reps = (batch_size + n - 1) // n
        source = pool.repeat((reps, 1, 1, 1))
        return source[:batch_size]
    start = (offset * batch_size) % n
    end = start + batch_size
    if end <= n:
        return pool[start:end]
    first = pool[start:]
    second = pool[: end - n]
    return torch.cat((first, second), dim=0)


def train_step(model, optimizer, scaler, hr, config, device, amp, channels_last, step_index):
    if channels_last:
        hr = hr.contiguous(memory_format=torch.channels_last)
    lr = gpu_game_degradation(hr, config["scale"], config["data"].get("degradation", {}))
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        pred = model(lr)
        tcfg = config["train"]
        loss = float(tcfg.get("l1_weight", 1.0)) * nn.functional.l1_loss(pred, hr)
        loss = loss + float(tcfg.get("edge_loss_weight", 0.05)) * edge_loss(pred, hr)
        ssim_weight = float(tcfg.get("ssim_loss_weight", 0.02))
        every = max(1, int(tcfg.get("ssim_every_n_steps", 4)))
        if ssim_weight > 0 and step_index % every == 0:
            loss = loss + ssim_weight * ssim_loss(pred.float(), hr.float())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_clip = float(config["train"].get("grad_clip", 1.0))
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss


def try_batch(config: dict, pool: torch.Tensor, batch_size: int, steps: int, warmup: int) -> dict:
    device = torch.device("cuda")
    model = build_model(config).to(device)
    channels_last = bool(config["train"].get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    amp = bool(config["train"].get("amp", True))
    optimizer, fused = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"[BATCH {batch_size}] GPU warmup {warmup} + measured {steps} steps (fused_adamw={fused})...")
    try:
        for step in range(1, warmup + 1):
            hr = get_batch(pool, batch_size, step).to(device, non_blocking=True)
            train_step(model, optimizer, scaler, hr, config, device, amp, channels_last, step)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        completed = 0
        start = time.perf_counter()
        for step in range(1, steps + 1):
            hr = get_batch(pool, batch_size, warmup + step).to(device, non_blocking=True)
            train_step(model, optimizer, scaler, hr, config, device, amp, channels_last, step)
            completed += 1
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-6)
        result = {
            "batch_size": batch_size,
            "oom": False,
            "images_per_sec": round(completed * batch_size / elapsed, 2),
            "ms_per_step": round(elapsed * 1000 / completed, 2),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "fused_adamw": fused,
        }
        print(f"[BATCH {batch_size}] {result['images_per_sec']} img/s | {result['ms_per_step']} ms/step | {result['peak_vram_mib']} MiB")
        return result
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"[BATCH {batch_size}] OOM")
            return {"batch_size": batch_size, "oom": True, "error": str(exc)[:240]}
        raise
    finally:
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()


def cache_signature(config: dict, config_path: Path, steps: int, warmup: int, batches: list[int], pool_size: int) -> dict:
    dataset_root = Path(config["data"]["root"])
    important = {
        "schema": SCHEMA_VERSION,
        "config": config_path.name,
        "training_code_version": config.get("version", "unknown"),
        "model": config["model"],
        "architecture": config["model"].get("architecture", "lr_residual"),
        "scale": config["scale"],
        "hr_patch": config["data"]["hr_patch"],
        "amp": config["train"].get("amp", True),
        "channels_last": config["train"].get("channels_last", True),
        "degradation": config["data"].get("degradation", {}),
        "loss": {
            "l1": config["train"].get("l1_weight", 1.0),
            "edge": config["train"].get("edge_loss_weight", 0.05),
            "ssim": config["train"].get("ssim_loss_weight", 0.02),
            "ssim_every": config["train"].get("ssim_every_n_steps", 4),
        },
        "steps": steps,
        "warmup": warmup,
        "batches": batches,
        "pool_size": pool_size,
        "mode": "quick_reused_pool_v2",
    }
    important["dataset_fingerprint"] = manifest_fingerprint(dataset_root)
    important["signature"] = stable_json_hash(important)
    return important


def valid_cached_payload(payload: dict, signature: dict) -> bool:
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    if payload.get("benchmark_signature") != signature["signature"]:
        return False
    results = payload.get("results", [])
    valid = [r for r in results if isinstance(r.get("images_per_sec"), (int, float)) and not r.get("oom")]
    return bool(valid and payload.get("recommended_batch_size"))


def main() -> None:
    parser = argparse.ArgumentParser(description="GameSR v1.7.5 quick training batch benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", default="16,32,64,96,128")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--load-cache", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    config_path = Path(args.config).resolve()
    config = resolve_config(load_config(config_path), config_path)
    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    if not batches:
        raise SystemExit("No batch sizes were supplied.")
    pool_size = max(max(batches), int(args.pool_size))
    signature = cache_signature(config, config_path, args.steps, args.warmup, batches, pool_size)

    if args.load_cache and args.load_cache.exists() and not args.force:
        try:
            payload = json.loads(args.load_cache.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict) and valid_cached_payload(payload, signature):
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

    seed_everything(int(config.get("seed", 42)))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pool = make_probe_pool(config, pool_size)
    results = []
    for batch in batches:
        try:
            result = try_batch(config, pool, batch, args.steps, args.warmup)
        except Exception as exc:
            result = {"batch_size": batch, "oom": False, "error": f"{type(exc).__name__}: {exc}"}
            print(result)
        results.append(result)
        if result.get("oom"):
            break

    valid = [r for r in results if not r.get("oom") and isinstance(r.get("images_per_sec"), (int, float))]
    fastest = max(valid, key=lambda x: x["images_per_sec"]) if valid else None
    recommended = None
    if fastest:
        # Prefer the smallest batch that is already within 95% of peak throughput.
        # This preserves more optimizer updates per epoch instead of blindly choosing
        # the largest/fastest batch for a tiny model.
        threshold = fastest["images_per_sec"] * 0.95
        near_peak = [r for r in valid if r["images_per_sec"] >= threshold]
        recommended = min(near_peak, key=lambda x: x["batch_size"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "config": config_path.name,
        "training_code_version": config.get("version", "unknown"),
        "model": config["model"],
        "architecture": config["model"].get("architecture", "lr_residual"),
        "hr_patch": config["data"]["hr_patch"],
        "scale": config["scale"],
        "benchmark_mode": "quick_reused_pool_h2d_gpu_degradation_full_loss",
        "steps": args.steps,
        "warmup": args.warmup,
        "pool_size": pool_size,
        "benchmark_signature": signature["signature"],
        "dataset_fingerprint": signature["dataset_fingerprint"],
        "results": results,
        "fastest_batch_size": fastest["batch_size"] if fastest else None,
        "recommended_batch_size": recommended["batch_size"] if recommended else None,
        "recommendation_rule": "smallest batch within 95% of fastest throughput",
        "created_unix": time.time(),
    }
    if args.json:
        write_json(args.json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not valid:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
