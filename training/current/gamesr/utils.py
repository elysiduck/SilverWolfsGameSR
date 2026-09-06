from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config, Path(path))
    return config


def validate_config(config: object, source: Path | None = None) -> None:
    label = str(source) if source else "configuration"
    if not isinstance(config, dict):
        raise ValueError(f"{label}: YAML root must be a mapping")
    for key in ("model", "data", "train", "output", "scale"):
        if key not in config:
            raise ValueError(f"{label}: missing required field '{key}'")
    model = config["model"]
    data = config["data"]
    train = config["train"]
    if not isinstance(model, dict) or not isinstance(data, dict) or not isinstance(train, dict):
        raise ValueError(f"{label}: model, data and train must be mappings")
    architecture = str(model.get("architecture", "lr_residual"))
    if architecture not in {"lr_residual", "hr_residual", "npu_sequential"}:
        raise ValueError(f"{label}: unsupported model architecture '{architecture}'")
    if int(model.get("channels", 0)) < 1:
        raise ValueError(f"{label}: model.channels must be at least 1")
    if architecture == "npu_sequential":
        if int(model.get("conv_layers", 0)) < 2:
            raise ValueError(f"{label}: npu_sequential requires model.conv_layers >= 2")
    elif int(model.get("blocks", -1)) < 0:
        raise ValueError(f"{label}: residual architectures require model.blocks >= 0")
    scale = float(config["scale"])
    if scale <= 1.0:
        raise ValueError(f"{label}: scale must be greater than 1")
    hr_patch = int(data.get("hr_patch", 0))
    if hr_patch < 1 or abs(round(hr_patch / scale) * scale - hr_patch) > 1e-6:
        raise ValueError(f"{label}: data.hr_patch must produce an integer LR patch at scale {scale}")
    if int(data.get("batch_size", 0)) < 1:
        raise ValueError(f"{label}: data.batch_size must be at least 1")
    if int(train.get("epochs", 0)) < 1:
        raise ValueError(f"{label}: train.epochs must be at least 1")


def checkpoint_signature(config: dict) -> dict:
    """Fields that must match before checkpoint weights/state may be reused."""
    return {
        "version": config.get("version", "unknown"),
        "model": config.get("model", {}),
        "scale": config.get("scale"),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8), "RGB")


def save_preview(lr: torch.Tensor, pred: torch.Tensor, hr: torch.Tensor, path: Path) -> None:
    hr_img = tensor_image(hr)
    lr_img = tensor_image(lr).resize(hr_img.size, Image.Resampling.BILINEAR)
    pred_img = tensor_image(pred)
    canvas = Image.new("RGB", (hr_img.width * 3, hr_img.height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate((("Bilinear", lr_img), ("GameSR", pred_img), ("Target", hr_img))):
        canvas.paste(image, (idx * hr_img.width, 24))
        draw.text((idx * hr_img.width + 5, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def append_csv(path: Path, row: dict) -> None:
    """Append a row and migrate older/malformed GameSR CSV schemas safely.

    Some pre-v1.7.5 training logs were written after new metrics had been
    introduced without expanding the CSV header first. ``csv.DictReader``
    represents those surplus values under the special ``None`` key.  When a
    newer trainer later tries to rewrite the file with an expanded schema, a
    normal ``DictWriter`` rejects that key.

    This migration keeps the old rows, maps surplus values onto newly-added
    fields when their order can be inferred, and preserves any remaining
    unknown values in ``legacy_extra_N`` columns.  The original file is only
    replaced after the rewritten temporary file is complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        existing_fields = [field for field in (reader.fieldnames or []) if field is not None]
        raw_rows = list(reader)

    missing = [key for key in row if key not in existing_fields]

    # Detect surplus columns left by older writers.  DictReader stores all
    # such values in row[None] as a list.
    max_extra = 0
    for old_row in raw_rows:
        extras = old_row.get(None) or []
        if not isinstance(extras, (list, tuple)):
            extras = [extras]
        max_extra = max(max_extra, len(extras))

    # The common migration case is that those surplus values are exactly the
    # metrics that are now present in ``missing``.  Map them in insertion
    # order.  If an older row contains still more values, preserve them in
    # explicit legacy columns rather than silently dropping them.
    mapped_extra_count = min(max_extra, len(missing))
    legacy_extra_count = max(0, max_extra - mapped_extra_count)
    legacy_fields = [f"legacy_extra_{idx + 1}" for idx in range(legacy_extra_count)]

    needs_rewrite = bool(missing or max_extra or legacy_fields)
    if not needs_rewrite:
        with path.open("a", newline="", encoding="utf-8") as out:
            csv.DictWriter(out, fieldnames=existing_fields, extrasaction="ignore").writerow(row)
        return

    fields = existing_fields + missing + [f for f in legacy_fields if f not in existing_fields and f not in missing]
    repaired_rows = []
    for old_row in raw_rows:
        clean = {key: value for key, value in old_row.items() if key is not None}
        extras = old_row.get(None) or []
        if not isinstance(extras, (list, tuple)):
            extras = [extras]

        for idx, value in enumerate(extras):
            if idx < len(missing):
                key = missing[idx]
            else:
                key = f"legacy_extra_{idx - len(missing) + 1}"
                if key not in fields:
                    fields.append(key)
            # Do not overwrite an explicitly parsed value from a newer row.
            if clean.get(key, "") in ("", None):
                clean[key] = value
        repaired_rows.append(clean)

    temporary = path.with_name(path.name + ".schema.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(repaired_rows)
        writer.writerow(row)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(dataset_root: Path) -> str:
    """Stable fingerprint of train/val/test manifests plus metadata."""
    digest = hashlib.sha256()
    for name in ("train.csv", "val.csv", "test.csv"):
        path = dataset_root / name
        if not path.exists():
            return "missing"
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def stable_json_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
