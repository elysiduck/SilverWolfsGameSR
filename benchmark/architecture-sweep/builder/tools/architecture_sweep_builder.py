from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# This tool deliberately reuses the GameSR project model definition and
# int8_ptq.py. It is not a second training/quantization stack.
import numpy as np
import onnx
from PIL import Image
import torch
from torch import nn

# This file is installed as <GameSR root>/tools/architecture_sweep_builder.py.
# When Python launches a script from tools/, sys.path[0] is tools/ rather than
# the project root, so explicitly add the parent before importing gamesr.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gamesr.model import GameSR

BUILDER_VERSION = "1.0"
DEFAULT_CHANNELS = "4,6,8,10,12,14,16,18,20,22,24,26,28,30,32"
DEFAULT_LAYERS = "2,3,4,5,6,7,8"


class CoreOnly(nn.Module):
    def __init__(self, model: GameSR) -> None:
        super().__init__()
        self.model = model

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model.forward_core(lr)


def parse_int_list(text: str, *, name: str, minimum: int) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {name} contains a non-integer value: {raw!r}") from exc
        if value < minimum:
            raise SystemExit(f"[FAIL] {name} values must be >= {minimum}, got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise SystemExit(f"[FAIL] {name} list is empty")
    return values


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parameter_count(channels: int, layers: int) -> int:
    # npu_sequential: 3->C head, (L-2) C->C hidden, C->3 tail, all 3x3+bias.
    head = 3 * channels * 9 + channels
    tail = channels * 3 * 9 + 3
    hidden = max(0, layers - 2) * (channels * channels * 9 + channels)
    return head + hidden + tail


def gmac_per_frame(channels: int, layers: int, height: int, width: int) -> float:
    mac_per_pixel = 3 * channels * 9 + max(0, layers - 2) * channels * channels * 9 + channels * 3 * 9
    return height * width * mac_per_pixel / 1e9


def make_calibration(dst: Path, width: int, height: int, count: int, seed: int) -> None:
    """Create deterministic broad-range RGB frames for probe quantization.

    These probes are for compiler/hardware latency, not image-quality scoring.
    Synthetic calibration keeps all architecture builds reproducible and avoids
    coupling the sweep to a particular game dataset.
    """
    dst.mkdir(parents=True, exist_ok=True)
    expected = [dst / f"cal_{i:02d}.png" for i in range(count)]
    if all(p.is_file() and p.stat().st_size > 0 for p in expected):
        return

    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    grad = np.stack(
        [
            np.broadcast_to(xx, (height, width)),
            np.broadcast_to(yy, (height, width)),
            np.broadcast_to((xx + yy) / 2.0, (height, width)),
        ],
        axis=2,
    )
    for i, path in enumerate(expected):
        noise = rng.random((height, width, 3), dtype=np.float32)
        # Add both smooth and high-frequency content so calibration sees a wide
        # activation range. The seed makes this byte-stable between reruns.
        checker = (((np.indices((height, width)).sum(axis=0) // (8 + (i % 5) * 4)) & 1).astype(np.float32))[..., None]
        arr = np.clip(0.45 * noise + 0.40 * np.roll(grad, i % 3, axis=2) + 0.15 * checker, 0.0, 1.0)
        Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGB").save(path)


def validate_float_graph(path: Path, layers: int, height: int, width: int) -> dict[str, int]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if counts.get("Conv", 0) != layers:
        raise RuntimeError(f"Float graph {path.name}: expected {layers} Conv nodes, found {counts.get('Conv', 0)}")
    forbidden = {op: counts.get(op, 0) for op in ("Resize", "Add", "Clip") if counts.get(op, 0)}
    if forbidden:
        raise RuntimeError(f"Float core graph contains forbidden post-processing nodes: {forbidden}")
    graph_input = model.graph.input[0]
    dims = [int(d.dim_value) if d.HasField("dim_value") else None for d in graph_input.type.tensor_type.shape.dim]
    expected = [1, 3, height, width]
    if dims != expected:
        raise RuntimeError(f"Float graph input mismatch: expected {expected}, got {dims}")
    return counts


def export_float(path: Path, channels: int, layers: int, width: int, height: int, seed: int) -> dict[str, int]:
    torch.manual_seed(seed + channels * 1000 + layers)
    model = GameSR(
        channels=channels,
        blocks=0,
        scale=1.5,
        architecture="npu_sequential",
        conv_layers=layers,
    ).cpu().eval()
    export_model = CoreOnly(model).cpu().eval()
    example = torch.zeros(1, 3, height, width, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        example,
        str(path),
        input_names=["lr"],
        output_names=["residual_lr"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    return validate_float_graph(path, layers, height, width)


def quantize(root: Path, float_path: Path, int8_path: Path, calibration: Path, cal_count: int, width: int, height: int, seed: int) -> None:
    script = root / "int8_ptq.py"
    cmd = [
        sys.executable,
        str(script),
        "--model", str(float_path),
        "--output", str(int8_path),
        "--calibration", str(calibration),
        "--count", str(cal_count),
        "--seed", str(seed),
        "--width", str(width),
        "--height", str(height),
        "--config", "XINT8",
        "--require-core-only",
    ]
    completed = subprocess.run(cmd, cwd=root)
    if completed.returncode != 0:
        raise RuntimeError(f"int8_ptq.py exited with code {completed.returncode}")


def validate_int8(int8_path: Path, meta_path: Path, expected_layers: int) -> dict:
    if not int8_path.is_file() or int8_path.stat().st_size == 0:
        raise RuntimeError("INT8 ONNX was not created")
    if not meta_path.is_file():
        raise RuntimeError("INT8 metadata was not created")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    report = meta.get("graph_validation") or {}
    if int(report.get("qlinearconv_count", -1)) != 0:
        raise RuntimeError("QLinearConv found; expected XINT8 Q/DQ Conv graph")
    if int(report.get("conv_count", -1)) != expected_layers:
        raise RuntimeError(f"INT8 graph expected {expected_layers} Conv nodes, found {report.get('conv_count')}")
    if int(report.get("quantize_count", 0)) <= 0 or int(report.get("dequantize_count", 0)) <= 0:
        raise RuntimeError("INT8 graph has no complete Q/DQ structure")
    for key in ("resize_count", "add_count", "clip_count"):
        if int(report.get(key, 0)) != 0:
            raise RuntimeError(f"INT8 core graph contains forbidden {key}: {report.get(key)}")
    return meta


def copy_payload_model(src_model: Path, src_meta: Path, payload_models: Path, arch: str) -> tuple[Path, Path]:
    dst_dir = payload_models / arch
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_model = dst_dir / src_model.name
    dst_meta = dst_dir / src_meta.name
    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_meta, dst_meta)
    return dst_model, dst_meta


def write_zip(payload_root: Path, exports_dir: Path, model_count: int) -> Path:
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = exports_dir / f"GameSR-NPU-Architecture-Sweep-Models-{model_count}-{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(payload_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(payload_root.parent))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GameSR XDNA architecture-sweep INT8 probe models")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--channels", default=DEFAULT_CHANNELS)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cal-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-float", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / "int8_ptq.py").is_file() or not (root / "gamesr" / "model.py").is_file():
        raise SystemExit("[FAIL] Run this Builder from the GameSR-training-code-v1.7.1 project root")
    if args.width < 64 or args.height < 64:
        raise SystemExit("[FAIL] width/height are implausibly small")
    if args.cal_count < 1:
        raise SystemExit("[FAIL] --cal-count must be >= 1")

    channels_list = parse_int_list(args.channels, name="channels", minimum=1)
    layers_list = parse_int_list(args.layers, name="layers", minimum=2)
    combinations = [(c, l) for c in channels_list for l in layers_list]

    build_root = root / "architecture_sweep"
    float_root = build_root / "float_models"
    int8_root = build_root / "int8_models"
    calibration = build_root / "calibration_synthetic"
    payload_root = build_root / "handheld_payload"
    payload_models = payload_root / "models"
    exports_dir = build_root / "exports"
    for d in (float_root, int8_root, payload_models, exports_dir):
        d.mkdir(parents=True, exist_ok=True)
    make_calibration(calibration, args.width, args.height, args.cal_count, args.seed)

    started = time.time()
    rows: list[dict] = []
    failures: list[dict] = []
    print(f"[SWEEP] Builder v{BUILDER_VERSION}")
    print(f"[SWEEP] Python: {sys.executable}")
    print(f"[SWEEP] Input: 1x3x{args.height}x{args.width}")
    print(f"[SWEEP] Channels: {channels_list}")
    print(f"[SWEEP] Layers: {layers_list}")
    print(f"[SWEEP] Models: {len(combinations)}")
    print(f"[SWEEP] Calibration: deterministic synthetic x{args.cal_count}")

    for index, (channels, layers) in enumerate(combinations, 1):
        arch = f"C{channels}-L{layers}"
        print(f"\n===== [{index}/{len(combinations)}] {arch} =====", flush=True)
        float_path = float_root / f"{arch}.onnx"
        out_dir = int8_root / arch
        int8_path = out_dir / f"{arch}-int8-qdq.onnx"
        meta_path = int8_path.with_suffix(int8_path.suffix + ".meta.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.skip_existing and int8_path.is_file() and meta_path.is_file():
                print(f"[SKIP] Existing INT8 model: {int8_path}")
                meta = validate_int8(int8_path, meta_path, layers)
                if not float_path.is_file():
                    # Float model is optional once validated INT8 exists.
                    float_counts = {"Conv": layers}
                else:
                    float_counts = validate_float_graph(float_path, layers, args.height, args.width)
            else:
                print("[1/2] Exporting GameSR npu_sequential core...")
                float_counts = export_float(float_path, channels, layers, args.width, args.height, args.seed)
                print(f"[PASS] Float graph: Conv={float_counts.get('Conv', 0)}, Resize/Add/Clip=0/0/0")
                print("[2/2] AMD Quark XINT8/QDQ...")
                quantize(root, float_path, int8_path, calibration, args.cal_count, args.width, args.height, args.seed)
                meta = validate_int8(int8_path, meta_path, layers)

            payload_model, payload_meta = copy_payload_model(int8_path, meta_path, payload_models, arch)
            row = {
                "architecture": arch,
                "channels": channels,
                "conv_layers": layers,
                "parameters": parameter_count(channels, layers),
                "gmac_per_frame": round(gmac_per_frame(channels, layers, args.height, args.width), 6),
                "input_shape": f"1x3x{args.height}x{args.width}",
                "model_relpath": str(payload_model.relative_to(payload_root)).replace("\\", "/"),
                "meta_relpath": str(payload_meta.relative_to(payload_root)).replace("\\", "/"),
                "sha256": sha256_file(payload_model),
                "bytes": payload_model.stat().st_size,
                "conv_count": int((meta.get("graph_validation") or {}).get("conv_count", layers)),
                "quantize_count": int((meta.get("graph_validation") or {}).get("quantize_count", 0)),
                "dequantize_count": int((meta.get("graph_validation") or {}).get("dequantize_count", 0)),
                "status": "READY",
            }
            rows.append(row)
            print(f"[READY] {arch}: {row['gmac_per_frame']:.4f} GMAC, {row['parameters']} params")
        except Exception as exc:
            failure = {"architecture": arch, "channels": channels, "conv_layers": layers, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[FAIL] {arch}: {failure['error']}", file=sys.stderr, flush=True)

    manifest = {
        "schema_version": 2,
        "builder": "GameSR NPU Architecture Sweep Builder",
        "builder_version": BUILDER_VERSION,
        "created": datetime.now().isoformat(),
        "source_environment": "GameSR-training-code-v1.7.1/.venv-quark",
        "architecture": "gamesr.model.GameSR / npu_sequential / forward_core",
        "quantizer": "existing GameSR int8_ptq.py / AMD Quark XINT8 QDQ",
        "input_shape": [1, 3, args.height, args.width],
        "calibration": {"type": "deterministic_synthetic", "count": args.cal_count, "seed": args.seed},
        "channels": channels_list,
        "layers": layers_list,
        "requested_models": len(combinations),
        "ready_models": len(rows),
        "failed_models": len(failures),
        "models": rows,
        "failures": failures,
    }
    manifest_json = payload_root / "sweep_manifest.json"
    manifest_csv = payload_root / "sweep_manifest.csv"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys()) if rows else ["architecture", "channels", "conv_layers", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    readme = payload_root / "README_HANDHELD.txt"
    readme.write_text(
        "GameSR NPU Architecture Sweep - prebuilt model payload\n"
        "======================================================\n\n"
        "This payload was built on the GameSR training laptop using the existing\n"
        ".venv-quark + AMD Quark XINT8/QDQ environment.\n\n"
        "The handheld does NOT need Python, PyTorch, ONNX Python packages, or AMD Quark.\n"
        "A separate Sweep Runner will consume sweep_manifest.json and models/.\n\n"
        f"Input: 1x3x{args.height}x{args.width}\n"
        f"Ready models: {len(rows)} / {len(combinations)}\n",
        encoding="utf-8",
    )

    zip_path: Path | None = None
    if args.package and rows:
        zip_path = write_zip(payload_root, exports_dir, len(rows))

    if not args.keep_float and float_root.exists():
        shutil.rmtree(float_root, ignore_errors=True)

    elapsed = time.time() - started
    print("\n============================================================")
    print(f"[DONE] Ready: {len(rows)}/{len(combinations)}   Failed: {len(failures)}")
    print(f"[DONE] Elapsed: {elapsed/60.0:.1f} min")
    print(f"[OUT]  {payload_root}")
    if zip_path:
        print(f"[ZIP]  {zip_path}")
    if failures:
        print(f"[WARN] {len(failures)} model(s) failed. Rerun with --skip-existing to resume.")
        # Partial output is deliberately kept; return nonzero so the launcher
        # makes the failure visible instead of silently treating it as complete.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
