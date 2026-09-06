from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODELS = {
    "1": ("tiny-npu-fast", "configs/tiny-npu-fast.yaml"),
    "2": ("tiny-npu-quality", "configs/tiny-npu-quality.yaml"),
    "3": ("tiny-c8", "configs/tiny-c8.yaml"),
    "fast": ("tiny-npu-fast", "configs/tiny-npu-fast.yaml"),
    "quality": ("tiny-npu-quality", "configs/tiny-npu-quality.yaml"),
}


def safe_gpu_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def run(cmd: list[str], log_file: Path, label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n\n[{datetime.now().isoformat()}] {label}\n")
        log.write("$ " + " ".join(cmd) + "\n")
        env = dict(__import__("os").environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(f"[FAIL] {label} (exit code {code})")


def python_cmd(root: Path, script: str, *args: str) -> list[str]:
    return [sys.executable, str(root / script), *args]


def quark_python_cmd(root: Path, script: str, *args: str) -> list[str]:
    import os
    override = os.environ.get("GAMESR_QUARK_PYTHON", "").strip().strip('"')
    candidates = [Path(override)] if override else []
    current = root.resolve()
    while True:
        candidates.extend((
            current / ".venv-quark" / "Scripts" / "python.exe",
            current / "venv-quark" / "Scripts" / "python.exe",
        ))
        if current.parent == current:
            break
        current = current.parent
    python = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if python is None:
        raise SystemExit(
            "AMD Quark environment was not found here or in a parent folder. "
            "Run 01_Install_RyzenAI_INT8.cmd once."
        )
    print(f"[ENV] Shared Quark Python: {python}")
    return [str(python), str(root / script), *args]


def _dataset_manifest_counts(dataset_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        path = dataset_root / f"{name}.csv"
        if not path.exists() or path.stat().st_size == 0:
            counts[name] = 0
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            counts[name] = sum(1 for row in csv.DictReader(handle) if (row.get("path") or "").strip())
    return counts


def _dataset_info(dataset_root: Path) -> dict:
    path = dataset_root / "dataset_info.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _dataset_recommendation(counts: dict[str, int], info: dict) -> tuple[bool, str]:
    total = sum(counts.values())
    min_eval = max(300, int((total * 0.05) + 0.999999)) if total else 300
    health = info.get("health") if isinstance(info.get("health"), dict) else {}
    if info and health.get("healthy") is False:
        return False, "dataset_info marks the split unhealthy"
    if counts.get("val", 0) < min_eval:
        return False, f"Val has only {counts.get('val', 0)} images; recommended minimum is {min_eval}"
    if counts.get("test", 0) < min_eval:
        return False, f"Test has only {counts.get('test', 0)} images; recommended minimum is {min_eval}"
    return True, "Val/Test image counts are healthy"


def _resolve_dataset_source(explicit_source: str | None, info: dict) -> str | None:
    if explicit_source:
        return explicit_source.strip().strip('"')
    stored = str(info.get("source") or "").strip()
    return stored or None


def _quick_source_inventory(dataset_root: Path, source: str | None) -> dict[str, int | str | None]:
    if not source:
        return {"source_files": None, "manifest_files": sum(_dataset_manifest_counts(dataset_root).values()), "new_files": None, "missing_files": None, "error": None}
    try:
        root = Path(source)
        if not root.exists():
            return {"source_files": None, "manifest_files": sum(_dataset_manifest_counts(dataset_root).values()), "new_files": None, "missing_files": None, "error": "source not reachable"}
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        source_paths = {
            str(p.resolve()).replace("\\", "/").casefold()
            for p in root.rglob("*")
            if p.suffix.lower() in extensions and p.is_file()
        }
        manifest_paths: set[str] = set()
        for name in ("train", "val", "test"):
            path = dataset_root / f"{name}.csv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw = (row.get("path") or "").strip()
                    if raw:
                        manifest_paths.add(str(Path(raw).resolve()).replace("\\", "/").casefold())
        return {
            "source_files": len(source_paths),
            "manifest_files": len(manifest_paths),
            "new_files": len(source_paths - manifest_paths),
            "missing_files": len(manifest_paths - source_paths),
            "error": None,
        }
    except Exception as exc:
        return {"source_files": None, "manifest_files": sum(_dataset_manifest_counts(dataset_root).values()), "new_files": None, "missing_files": None, "error": str(exc)}


def _run_dataset_action(root: Path, source: str, mode: str, log_file: Path, label: str) -> None:
    run(
        python_cmd(root, "prepare_dataset.py", "--source", source, "--output", "dataset", "--mode", mode),
        log_file,
        label,
    )


def ensure_dataset(root: Path, source: str | None, log_file: Path, reprepare: bool = False) -> dict:
    """Interactive v1.8.7 dataset manager.

    Existing manifests stay fixed unless the user explicitly chooses an update.
    Incremental mode keeps existing session ownership and sends new sessions to
    Train unless Val/Test need repair. Full mode rebuilds a balanced, session-
    isolated split by image count rather than session count.
    """
    dataset_root = root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    counts = _dataset_manifest_counts(dataset_root)
    info = _dataset_info(dataset_root)
    manifests_exist = all(counts.get(name, 0) > 0 for name in ("train", "val", "test"))
    resolved_source = _resolve_dataset_source(source, info)
    inventory = _quick_source_inventory(dataset_root, resolved_source) if manifests_exist else {}

    if not manifests_exist:
        resolved_source = resolved_source or input("Clean screenshot folder path: ").strip().strip('"')
        if not resolved_source:
            raise SystemExit("No source folder provided.")
        _run_dataset_action(root, resolved_source, "full", log_file, "[1/8] Full dataset rebuild")
        return {"action": "full", "changed": True, "source": resolved_source}

    if reprepare:
        resolved_source = resolved_source or input("Clean screenshot folder path: ").strip().strip('"')
        if not resolved_source:
            raise SystemExit("No source folder provided.")
        _run_dataset_action(root, resolved_source, "full", log_file, "[1/8] Full dataset rebuild (--reprepare)")
        return {"action": "full", "changed": True, "source": resolved_source}

    healthy, reason = _dataset_recommendation(counts, info)
    new_files = inventory.get("new_files") if inventory else None
    default_choice = "2" if healthy and resolved_source and isinstance(new_files, int) and new_files > 0 else ("1" if healthy else "3")
    while True:
        version = info.get("dataset_version")
        version_text = f"v{int(version):04d}" if isinstance(version, int) else "legacy"
        print("\n" + "=" * 72)
        print(" GameSR Dataset Manager v1.8.7")
        print("=" * 72)
        print(f"Current dataset: {version_text}")
        print(f"  Train: {counts.get('train', 0)}")
        print(f"  Val  : {counts.get('val', 0)}")
        print(f"  Test : {counts.get('test', 0)}")
        print(f"  Status: {'OK' if healthy else 'NEEDS REBUILD'} - {reason}")
        if resolved_source:
            print(f"  Source: {resolved_source}")
            if inventory.get("source_files") is not None:
                print(f"  Clean files found: {inventory.get('source_files')} | Manifest files: {inventory.get('manifest_files')} | New: {inventory.get('new_files')} | Missing: {inventory.get('missing_files')}")
            elif inventory.get("error"):
                print(f"  Fast inventory unavailable: {inventory.get('error')}")
        else:
            print("  Source: unknown (legacy manifests; update actions will ask once)")
        print("\nDataset action:")
        print("  1. Reuse existing manifests (no rescan; fastest)")
        print("  2. Incremental update (keep session ownership; add new data)")
        print("  3. Full rebuild from all clean images (balanced 80/10/10 by image count)")
        print("  4. Show detailed dataset statistics")
        if not healthy:
            print("  [RECOMMENDED] Full rebuild: the current Val/Test split is not statistically healthy.")
        elif resolved_source and isinstance(inventory.get("new_files"), int) and inventory.get("new_files", 0) > 0:
            print("  [RECOMMENDED] Incremental update: preserves session ownership while admitting new data.")
        elif resolved_source:
            print("  [RECOMMENDED] Reuse: no new source files were detected by the fast inventory.")
        else:
            print("  [RECOMMENDED] Reuse, or choose 2/3 and provide the clean source folder.")
        raw = input(f"Select [{default_choice}]: ").strip() or default_choice

        if raw == "1":
            if not healthy:
                print("[WARN] Reusing an unhealthy dataset. Training is allowed, but final Test metrics may be unreliable.")
            print(f"[DATA] Reusing fixed manifests: {counts}")
            return {"action": "reuse", "changed": False, "source": resolved_source}

        if raw in {"2", "3", "4"}:
            if not resolved_source:
                resolved_source = input("Clean screenshot folder path: ").strip().strip('"')
                if not resolved_source:
                    print("[DATA] No source provided; returning to dataset menu.")
                    continue
            mode = "incremental" if raw == "2" else ("full" if raw == "3" else "stats")
            label = {
                "incremental": "[1/8] Incremental dataset update",
                "full": "[1/8] Full dataset rebuild",
                "stats": "[DATA] Detailed dataset statistics",
            }[mode]
            if mode == "stats":
                run(
                    python_cmd(root, "prepare_dataset.py", "--source", resolved_source, "--output", "dataset", "--mode", "stats"),
                    log_file,
                    label,
                )
                counts = _dataset_manifest_counts(dataset_root)
                info = _dataset_info(dataset_root)
                healthy, reason = _dataset_recommendation(counts, info)
                inventory = _quick_source_inventory(dataset_root, resolved_source)
                new_files = inventory.get("new_files")
                default_choice = "2" if healthy and isinstance(new_files, int) and new_files > 0 else ("1" if healthy else "3")
                continue
            _run_dataset_action(root, resolved_source, mode, log_file, label)
            counts = _dataset_manifest_counts(dataset_root)
            info = _dataset_info(dataset_root)
            healthy, reason = _dataset_recommendation(counts, info)
            return {"action": mode, "changed": True, "source": resolved_source, "healthy": healthy}

        print("Please select 1, 2, 3, or 4.")

def _parse_positive_int(raw: str, label: str, minimum: int) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{label} must be an integer.") from exc
    if value < minimum:
        raise SystemExit(f"{label} must be >= {minimum}.")
    return value


def make_custom_config(root: Path, channels: int, layers: int, output_channels: int = 3, epochs: int | None = None, name: str | None = None) -> tuple[str, Path]:
    """Create a reproducible npu_sequential training profile for one C/L/O combination.

    The generated YAML lives directly under configs/ so the existing v1.7.x
    relative-path semantics (dataset/output relative to project root) remain
    unchanged. Re-running the same C/L pair reuses the same output directory
    and therefore can auto-resume safely.
    """
    import copy
    import yaml

    channels = int(channels)
    layers = int(layers)
    output_channels = int(output_channels)
    if channels < 1:
        raise SystemExit("Channels (C) must be >= 1.")
    if layers < 2:
        raise SystemExit("Conv layers (L) must be >= 2.")
    if output_channels < 3:
        raise SystemExit("Output channels (O) must be >= 3.")

    base_path = root / "configs" / "tiny-npu-fast.yaml"
    config = copy.deepcopy(load_yaml(base_path))
    default_profile = f"gamesr-c{channels}-l{layers}-o{output_channels}"
    if output_channels > 3:
        # v2 has a full bilinear skip and a zero-initialized residual decoder.
        # Use a new name so an incompatible v1 direct-RGB checkpoint is never
        # resumed automatically.
        default_profile += "-phase3x3r-v4"
    profile = (name or default_profile).strip()
    if output_channels > 3:
        profile = profile.replace("phase3x3r-v2", "phase3x3r-v4").replace("phase3x3r-v3", "phase3x3r-v4")
    profile = re.sub(r"[^A-Za-z0-9._-]+", "-", profile).strip("-.")
    if not profile:
        profile = f"gamesr-c{channels}-l{layers}-o{output_channels}"

    config["version"] = "1.7.4"
    config["trainer_version"] = "1.8.7"
    config["project"] = profile
    config["model"] = {
        "channels": channels,
        "architecture": "npu_sequential",
        "conv_layers": layers,
        "output_channels": output_channels,
    }
    if epochs is not None:
        if int(epochs) < 1:
            raise SystemExit("Epochs must be >= 1.")
        config["train"]["epochs"] = int(epochs)
    config["output"] = f"outputs/{profile}"
    # v1.7.6 remote-training defaults. Training caches compressed
    # source bytes in RAM only; it never copies the dataset to NVMe.
    config.setdefault("data", {})["val_batch_size"] = 16
    # v1.8.1 Game-aware degradation profile derived from real GameSR screenshots:
    # digital game capture needs far less sensor noise/compression than photo SR,
    # while resize diversity remains useful. Weighted kernels track the actual
    # real-time scaling path more closely than equal-probability sampling.
    config["data"]["degradation"] = {
        "resize_kernels": ["bicubic", "bilinear", "area"],
        "resize_kernel_weights": [0.50, 0.40, 0.10],
        "blur_probability": 0.10,
        "sharpen_probability": 0.08,
        "post_sharpen_probability": 0.08,
        "jpeg_probability": 0.0,
        "noise_probability": 0.04,
        "quantization_probability": 0.025,
        "brightness_probability": 0.03,
        "contrast_probability": 0.05,
        "jpeg_quality": [80, 96],
        "noise_std": [0.2, 1.0],
    }
    config["data"]["streaming_prefetch"] = {
        "enabled": "auto",
        "read_ahead_samples": 256,
        "fetch_threads": 24,
        "decode_threads": 6,
        "pin_memory": True,
        "batch_queue": 4,
        "cuda_prefetch": True,
        "persistent_cache": True,
        "cache_budget_gib": "auto",
        "cache_reserve_gib": 8.0,
        "cache_cap_gib": 12.0,
        "cache_auto_fraction": 0.50,
        "cache_floor_gib": 1.0,
        "cache_protected_ratio": 0.80,
        "emergency_free_gib": 4.0,
        "next_epoch_prefetch_samples": 256,
        "validation_cache": "auto",
        "validation_cache_budget_gib": "auto",
        "validation_cache_reserve_gib": 7.0,
        "validation_cache_cap_gib": 1.5,
        "validation_cache_auto_fraction": 0.50,
        "validation_cache_floor_mib": 64.0,
        "validation_read_ahead_samples": 96,
        "validation_threads": 12,
        "validation_pin_memory": True,
    }
    # v1.8 training-algorithm upgrade. O3 keeps config.version=1.7.4 so its
    # compatible checkpoints can continue. The checkpoint signature also
    # includes the deployment decoder contract, isolating phase v1 from v2.
    config["data"]["hard_patch_mining"] = {
        # v1.8.1: rank by reconstruction error * target-detail value, not error
        # alone. This protects intentional DOF/Bloom/VFX frames from monopolizing
        # the tiny model capacity. Selected-sample gradient gain is capped.
        "enabled": True,
        "min_fraction": 0.10,
        "max_fraction": 0.25,
        "min_mix": 0.06,
        "max_mix": 0.20,
        "weight_cap": 1.50,
        "detail_floor": 0.45,
        "detail_low": 0.006,
        "detail_high": 0.030,
        # v1.8.2 late-polish targets: reduce hard-example pressure after ~50%.
        "polish_fraction": 0.15,
        "polish_mix": 0.09,
    }
    config["data"]["degradation_curriculum"] = {
        "start_strength": 0.85,
        "full_strength_at": 0.35,
    }
    config.setdefault("train", {})["algorithm_v18"] = {
        "enabled": True,
        "game_aware_profile": "v1.8.2",
        "foundation_end": 0.06,
        "detail_end": 0.30,
        # Base Edge=0.05 -> max 0.0725; Base SSIM=0.02 -> max 0.030.
        "edge_boost": 0.45,
        "ssim_boost": 0.50,
        "high_frequency_weight": 0.03,
        "high_frequency_every_n_steps": 2,
        # HF is weighted by target detail: UI/text/geometry high, intentional
        # soft DOF/Bloom low, with a floor that still suppresses ringing.
        "high_frequency_detail_aware": True,
        "high_frequency_detail_floor": 0.15,
        "high_frequency_detail_low": 0.006,
        "high_frequency_detail_high": 0.050,
        # v1.8.2: after the model has recovered detail, gradually lower HF/hard
        # pressure and spend base-L1 capacity on information-rich game patches.
        "polish_start": 0.50,
        "polish_full_at": 0.75,
        "polish_edge_weight": 0.0675,
        "polish_ssim_weight": 0.030,
        "polish_high_frequency_weight": 0.012,
        "information_aware_l1": {
            "enabled": True,
            "low_information_weight": 0.80,
            "high_information_weight": 1.15,
            "flat_black_weight": 0.80,
            "dark_mean_threshold": 0.12,
            "flat_std_threshold": 0.035,
            "min_weight": 0.75,
            "max_weight": 1.25,
        },
    }

    config.setdefault("deployment", {})["core_only"] = True
    config["deployment"]["postprocess"] = (
        "magpie_bilinear_residual_v1" if output_channels == 3 else "phase_aware_3x3_v1"
    )
    if output_channels > 3:
        config["training_contract"] = "phase_residual_stable_v187"
        config["deployment"]["decoder"] = {
            "kernel_size": 3,
            "phase_x": 3,
            "phase_y": 3,
            "scale_numerator": 3,
            "scale_denominator": 2,
            "padding_mode": "replicate",
            "base_skip": 1.0,
            "output": "rgb_residual",
        }
        phase_train = config.setdefault("train", {})
        phase_train["phase_aware_loss"] = {
            "enabled": True,
            "charbonnier_weight": 1.0,
            "charbonnier_epsilon": 0.001,
            "ms_ssim_weight": 0.03,
            "gradient_weight": 0.02,
            "frequency_weight": 0.004,
            "ringing_weight": 0.02,
            "ms_ssim_every_n_steps": 1,
            "frequency_every_n_steps": 1,
            "detail_ramp_start": 0.15,
            "detail_ramp_full_at": 0.55,
            "polish_start": 0.70,
            "polish_full_at": 0.90,
            "polish_ms_ssim_weight": 0.020,
            "polish_gradient_weight": 0.012,
            "polish_frequency_weight": 0.002,
            "polish_ringing_weight": 0.020,
        }
        # Stable residual-SR optimizer/controller. Respect the requested epoch
        # count exactly; if no epoch count is supplied, keep the legacy/default 100.
        # Early stopping may still terminate earlier when quality has plateaued.
        if epochs is None:
            phase_train["epochs"] = 100
        phase_train["learning_rate"] = 0.00015
        phase_train["warmup_epochs"] = 3
        phase_train["lr_schedule"] = {
            "mode": "stable_piecewise",
            "hold_fraction": 0.10,
            "half_fraction": 0.30,
            "half_ratio": 0.50,
            "min_ratio": 0.0666667,
        }
        phase_train["quality_control"] = {
            "enabled": True,
            "edge_soft": 1.08,
            "edge_hard": 1.12,
            "edge_recover": 1.06,
            "psnr_drop_brake_db": 0.10,
            "oversharp_soft": 0.24,
            "detail_recover_step": 0.10,
            "lr_plateau_patience": 4,
            "lr_plateau_min_delta": 0.02,
            "lr_reduce_factor": 0.50,
            "lr_reduce_start": 0.15,
            "min_lr": 0.00001,
        }
        phase_train["early_stopping"] = {
            "enabled": True,
            "patience": 15,
            "min_delta_psnr": 0.02,
            "start_progress": 0.70,
        }
        phase_train["quality_checkpoint"] = {
            "max_edge_ratio": 1.10,
            "min_delta_psnr": 0.0,
            "min_detail_delta_psnr": 0.0,
            "max_oversharp_ratio": 0.26,
        }
        # Keep hard mining gentle: it should focus a small share of the batch,
        # not replace the reconstruction objective.
        config["data"]["hard_patch_mining"].update({
            "enabled": True,
            "min_fraction": 0.08,
            "max_fraction": 0.12,
            "min_mix": 0.02,
            "max_mix": 0.04,
            "polish_fraction": 0.08,
            "polish_mix": 0.01,
        })
        # Real low-to-high degradation curriculum with a permanent clean anchor.
        config["data"]["degradation"].update({
            "curriculum_mode": "stable_v2",
            "clean_resize_kernel": "bicubic",
            "clean_anchor_start": 1.00,
            "clean_anchor_end": 0.75,
            "resize_kernel_weights": [0.75, 0.20, 0.05],
        })
        config["data"]["degradation_curriculum"] = {
            "start_strength": 0.10,
            "full_strength_at": 0.88,
        }
    config["architecture_selection"] = {
        "channels": channels,
        "conv_layers": layers,
        "output_channels": output_channels,
        "source": "interactive_clo_selector",
    }

    path = root / "configs" / f"{profile}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[ARCH] Generated profile: {path}")
    print(f"[ARCH] C={channels} | L={layers} | O={output_channels} | epochs={config['train']['epochs']} | output={config['output']}")
    print("[V1.8.7] Stable phase-residual training + versioned session-aware dataset manager")
    if channels not in (8, 16, 24, 32):
        print("[ARCH] Note: current XDNA sweep found C8/C16/C24/C32 especially attractive, but this value is allowed.")
    return profile, path


def choose_custom_model(root: Path, args: argparse.Namespace) -> list[tuple[str, Path]]:
    default_c = 24
    default_l = 3
    default_o = 3
    if args.channels is None:
        print("\nCustom NPU architecture (npu_sequential)")
        print("  Sweep sweet-spot widths: C8 / C16 / C24 / C32")
        raw_c = input(f"Channels C [{default_c}]: ").strip() or str(default_c)
        channels = _parse_positive_int(raw_c, "Channels C", 1)
    else:
        channels = int(args.channels)

    if args.layers is None:
        raw_l = input(f"Conv layers L [{default_l}]: ").strip() or str(default_l)
        layers = _parse_positive_int(raw_l, "Conv layers L", 2)
    else:
        layers = int(args.layers)

    if args.output_channels is None:
        print("  O3 = original RGB residual; O4/O6/O8 = phase-aware 3x3 learned reconstruction")
        raw_o = input(f"Output channels O [{default_o}]: ").strip() or str(default_o)
        output_channels = _parse_positive_int(raw_o, "Output channels O", 3)
    else:
        output_channels = int(args.output_channels)

    epochs = args.epochs
    if epochs is None and args.channels is None and args.layers is None:
        raw_epochs = input("Epochs [100]: ").strip()
        epochs = _parse_positive_int(raw_epochs, "Epochs", 1) if raw_epochs else 100

    name, cfg = make_custom_config(root, channels, layers, output_channels, epochs=epochs, name=args.name)
    return [(name, cfg)]


def choose_models(choice: str | None, root: Path, args: argparse.Namespace) -> list[tuple[str, Path]]:
    if choice:
        normalized = choice.lower()
        if normalized in {"custom", "5"}:
            return choose_custom_model(root, args)
        if normalized == "all" or normalized == "4":
            keys = ("1", "2")
            return [(MODELS[key][0], Path(MODELS[key][1])) for key in keys]
        if normalized in MODELS:
            name, cfg = MODELS[normalized]
            return [(name, Path(cfg))]
        raise SystemExit("--model must be fast, quality, custom, 1, 2, 3, 4, 5, or all.")
    print("\nSelect model:")
    print("  1. NPU Fast    (legacy preset: C8-L8)")
    print("  2. NPU Quality (legacy preset: C8-L14)")
    print("  3. Legacy Tiny-C8 (compatibility only)")
    print("  4. Train both legacy NPU presets")
    print("  5. Custom NPU architecture (choose C, L and O)  <-- recommended")
    selected = input("Model [5]: ").strip() or "5"
    return choose_models(selected, root, args)


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def torch_info(root: Path) -> tuple[str, str]:
    script = "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.cuda or '')"
    result = subprocess.run([sys.executable, "-c", script], cwd=root, check=True, capture_output=True, text=True)
    lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    return (lines[0] if lines else "unknown_gpu", lines[1] if len(lines) > 1 else "")


def benchmark_path(root: Path, config_path: Path, gpu: str) -> Path:
    return root / "outputs" / "benchmarks" / safe_gpu_name(gpu) / f"{config_path.stem}.json"


def print_benchmark(payload: dict) -> None:
    print("\nBenchmark results")
    print("  Batch | images/s | ms/step | VRAM MiB | Status")
    print("  ------+----------+----------+----------+--------")
    for row in payload.get("results", []):
        if row.get("oom"):
            status = "OOM"
        elif "images_per_sec" in row:
            status = "OK"
        else:
            status = "ERROR"
        ips = row.get("images_per_sec", "-")
        ms = row.get("ms_per_step", "-")
        vram = row.get("peak_vram_mib", "-")
        print(f"  {row.get('batch_size', '-'):>5} | {ips:>8} | {ms:>8} | {vram:>8} | {status}")
    fastest = payload.get("fastest_batch_size")
    recommended = payload.get("recommended_batch_size")
    if fastest:
        print(f"  Fastest raw throughput: batch {fastest}")
    if recommended:
        print(f"  Recommended: batch {recommended} (smallest batch within 95% of peak throughput)")
    if not fastest and not recommended:
        print("  Recommendation unavailable")


def cached_payload(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("schema_version") != 7:
        return None
    results = [r for r in payload.get("results", []) if not r.get("oom") and isinstance(r.get("images_per_sec"), (int, float))]
    if not results or not payload.get("recommended_batch_size"):
        return None
    return payload


def load_or_run_benchmark(root: Path, config_path: Path, log_file: Path, force: bool = False) -> tuple[dict, Path]:
    gpu, _ = torch_info(root)
    path = benchmark_path(root, config_path, gpu)
    if not force:
        cached = cached_payload(path)
        if cached:
            print(f"\n[BENCHMARK] Loaded valid cached result: {path}")
            print_benchmark(cached)
            return cached, path
        if path.exists():
            print(f"[BENCHMARK] Existing cache is missing/invalid. Re-running: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        root, "benchmark_speed.py", "--config", str(config_path), "--json", str(path),
        "--steps", "6", "--warmup", "2", "--pool-size", "256",
    )
    if force:
        cmd += ["--force"]
    run(cmd, log_file, "[2/8] Benchmarking training batch sizes")
    payload = cached_payload(path)
    if not payload:
        raise SystemExit("[FAIL] Benchmark produced no valid result. Training will not start.")
    print(f"[BENCHMARK] Saved valid result: {path}")
    print_benchmark(payload)
    return payload, path


def select_batch(payload: dict) -> int:
    results = [r for r in payload.get("results", []) if not r.get("oom") and isinstance(r.get("images_per_sec"), (int, float))]
    if not results:
        raise SystemExit("No valid benchmark result is available.")
    fastest = max(results, key=lambda r: r["images_per_sec"])
    recommended_value = int(payload.get("recommended_batch_size") or fastest["batch_size"])
    choices = {int(r["batch_size"]): r for r in results}
    raw = input(f"Select batch size [{recommended_value} = recommended, fastest={fastest['batch_size']}]: ").strip()
    if not raw:
        return recommended_value
    try:
        value = int(raw)
    except ValueError:
        print(f"[BATCH] Invalid input; using recommended batch {recommended_value}.")
        return recommended_value
    if value not in choices:
        print(f"[BATCH] Batch {value} was not benchmarked; using recommended batch {recommended_value}.")
        return recommended_value
    return value


def one_model(root: Path, name: str, config_path: Path, log_file: Path, args: argparse.Namespace) -> None:
    config = load_yaml(config_path)
    output = Path(config["output"])
    if not output.is_absolute():
        output = root / output
    latest = output / "latest.pth"
    best_quality = output / "best_quality.pth"
    best_psnr = output / "best_psnr.pth"
    best = output / "best.pth"

    payload, bench_path_value = load_or_run_benchmark(root, config_path, log_file, force=args.force_benchmark)
    batch = select_batch(payload)
    print(f"[BATCH] Selected for {name}: {batch}")

    train_args = ["--config", str(config_path), "--batch-size", str(batch)]
    if not args.no_auto_resume and latest.exists():
        train_args += ["--auto-resume"]
    run(python_cmd(root, "train.py", *train_args), log_file, f"[3/8] Training {name}")

    checkpoint = best_quality if best_quality.exists() else (best_psnr if best_psnr.exists() else (best if best.exists() else latest))
    if not checkpoint.exists():
        raise SystemExit(f"[FAIL] No checkpoint produced for {name}: {checkpoint}")
    run(python_cmd(root, "evaluate.py", "--config", str(config_path), "--checkpoint", str(checkpoint), "--limit", "0"), log_file, f"[4/8] Full test evaluation {name}")
    onnx_path = root / "models" / f"{name}-720p-to-1080p.onnx"
    run(python_cmd(root, "export_onnx.py", "--config", str(config_path), "--checkpoint", str(checkpoint), "--output", str(onnx_path), "--use-ema"), log_file, f"[5/8] Exporting {name} to ONNX")

    fp16_path = root / "models" / f"{name}-720p-to-1080p-fp16.onnx"
    fp16_ok = False
    try:
        run(python_cmd(root, "fp16_convert.py", "--model", str(onnx_path), "--output", str(fp16_path)), log_file, f"[6/8] Building FP16 {name}")
        fp16_ok = fp16_path.exists()
    except SystemExit as exc:
        print(f"[WARN] FP16 build skipped for {name}: {exc}")

    int8_path = root / "models" / f"{name}-720p-to-1080p-int8-qdq.onnx"
    int8_ok = False
    calibration = args.source or str(root / "dataset" / "train.csv")
    try:
        int8_arguments = [
            "--model", str(onnx_path), "--output", str(int8_path),
            "--calibration", str(calibration), "--count", "128", "--config", "XINT8",
        ]
        if bool(config.get("deployment", {}).get("core_only", False)):
            int8_arguments.append("--require-core-only")
            int8_arguments += ["--output-channels", str(int(config.get("model", {}).get("output_channels", 3)))]
            int8_arguments += ["--deployment-contract", str(config.get("deployment", {}).get("postprocess", "magpie_bilinear_residual_v1"))]
        run(
            quark_python_cmd(root, "int8_ptq.py", *int8_arguments),
            log_file,
            f"[7/8] Building AMD Quark Ryzen AI XINT8/QDQ {name}",
        )
        int8_ok = int8_path.exists()
    except SystemExit as exc:
        if bool(config.get("deployment", {}).get("core_only", False)):
            raise SystemExit(f"[FAIL] Required NPU INT8 build failed for {name}: {exc}") from exc
        print(f"[WARN] INT8/QDQ build failed for legacy profile {name}: {exc}")

    run(
        python_cmd(root, "build_deployment.py", "--model", name, "--config", str(config_path), "--root", str(root)),
        log_file,
        f"[8/8] Packaging deployment bundle {name}",
    )

    output.mkdir(parents=True, exist_ok=True)
    selection = {
        "created": datetime.now().isoformat(),
        "gpu": payload.get("gpu"),
        "model": name,
        "config": str(config_path),
        "batch_size": batch,
        "benchmark_path": str(bench_path_value),
        "benchmark_created_unix": payload.get("created_unix"),
        "float_onnx": str(onnx_path),
        "fp16_onnx": str(fp16_path) if fp16_ok else None,
        "int8_qdq": str(int8_path) if int8_ok else None,
        "deployment_bundle": str(root / "deployment" / name),
    }
    (output / "batch_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GameSR v1.7.6 one-click NPU training pipeline with train + validation RAM cache")
    parser.add_argument("--model", default=None, help="fast/quality/custom/1/2/3/4/5/all")
    parser.add_argument("--channels", type=int, default=None, help="custom NPU width C; use together with --layers")
    parser.add_argument("--layers", type=int, default=None, help="custom NPU total Conv count L; use together with --channels")
    parser.add_argument("--output-channels", type=int, default=None, help="custom NPU output channels O (3, 4, 6, etc.)")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for a custom C/L run")
    parser.add_argument("--name", default=None, help="optional custom profile name; default gamesr-c{C}-l{L}")
    parser.add_argument("--source", default=None)
    parser.add_argument("--config", default=None, help="config for --benchmark-only")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--force-benchmark", action="store_true")
    parser.add_argument("--reprepare", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()
    supplied = (args.channels is not None, args.layers is not None, args.output_channels is not None)
    if any(supplied) and not all(supplied):
        raise SystemExit("--channels, --layers and --output-channels must be supplied together.")
    if args.channels is not None and args.model is None:
        args.model = "custom"

    root = Path(__file__).resolve().parent
    log_root = root / "outputs" / "pipeline_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    print("=" * 72)
    print(" GameSR v1.8.7 | Dataset Manager + C/L/O train/export/package")
    print("=" * 72)
    run(python_cmd(root, "check_env.py"), log_file, "[0/8] Checking training environment")
    dataset_state = ensure_dataset(root, args.source, log_file, reprepare=args.reprepare)
    if dataset_state.get("changed"):
        print("[DATA] Dataset manifest contract changed; incompatible old auto-resume checkpoints will be skipped automatically.")

    if args.config:
        p = Path(args.config)
        if not p.is_absolute():
            p = root / p
        models = [(p.stem, p)]
    else:
        models = choose_models(args.model, root, args)

    if args.benchmark_only:
        for name, config_path in models:
            payload, path = load_or_run_benchmark(root, config_path, log_file, force=args.force_benchmark)
            print(f"\n[BENCHMARK ONLY] {name}")
            print_benchmark(payload)
            print(f"Saved: {path}")
        return

    for name, config_rel in models:
        config_path = root / config_rel if not config_rel.is_absolute() else config_rel
        one_model(root, name, config_path, log_file, args)

    print("\n[OK] Full GameSR pipeline completed.")


if __name__ == "__main__":
    main()
