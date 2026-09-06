from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image

BUILDER_VERSION = "1.7.1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class ImageCalibrationDataReader(CalibrationDataReader):
    """Deterministic full-frame reader accepted by AMD Quark and ORT."""

    def __init__(self, paths: list[Path], input_name: str, width: int, height: int) -> None:
        self.paths = list(paths)
        self.input_name = input_name
        self.width = int(width)
        self.height = int(height)
        self.index = 0

    def __len__(self) -> int:
        return len(self.paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        self.index += 1
        if self.index == 1 or self.index % 8 == 0 or self.index == len(self.paths):
            print(f"[CAL] {self.index}/{len(self.paths)}: {path.name}", flush=True)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != (self.width, self.height):
                rgb = rgb.resize((self.width, self.height), Image.Resampling.BICUBIC)
            array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
        nchw = np.transpose(array, (2, 0, 1))[None, ...].copy()
        return {self.input_name: nchw}

    def rewind(self) -> None:
        self.index = 0


def read_image_paths(source: Path, count: int, seed: int) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".csv":
        paths: list[Path] = []
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = next((row[key] for key in ("path", "file", "image", "hr", "filepath") if row.get(key)), None)
                if raw:
                    candidate = Path(raw).expanduser()
                    if not candidate.is_absolute():
                        candidate = source.parent / candidate
                    paths.append(candidate.resolve())
    elif source.is_dir():
        paths = sorted(
            (path.resolve() for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: str(path).lower(),
        )
    else:
        paths = []
    paths = [path for path in paths if path.is_file()]
    random.Random(seed).shuffle(paths)
    return paths[:count]


def valid_xint8_zero_point(dtype: str, value: int) -> bool:
    """Validate symmetric XINT8 values according to ONNX storage dtype."""
    if dtype == "UINT8":
        return value in (0, 128)
    if dtype in ("INT8", "INT16", "INT32", "INT64"):
        return value == 0
    return False


def graph_report(model: onnx.ModelProto) -> dict[str, object]:
    op_counts = Counter(node.op_type for node in model.graph.node)
    domains = Counter((node.domain or "ai.onnx") for node in model.graph.node)
    initializers = {item.name: item for item in model.graph.initializer}
    zero_points: list[int] = []
    zero_point_histogram: Counter[str] = Counter()
    invalid_zero_points = 0
    for node in model.graph.node:
        if "QuantizeLinear" not in node.op_type and "DequantizeLinear" not in node.op_type:
            continue
        if len(node.input) >= 3 and node.input[2] in initializers:
            initializer = initializers[node.input[2]]
            dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
            values = numpy_helper.to_array(initializer).reshape(-1)
            for raw_value in values:
                value = int(raw_value)
                zero_points.append(value)
                zero_point_histogram[f"{dtype}:{value}"] += 1
                # Quark XINT8 uses signed symmetric math, but may encode
                # activations in UINT8 storage. UINT8 value 128 is the
                # signed-zero centre; value 0 is also valid for ranges
                # constrained by Relu/Clip. Signed tensors must use zero.
                invalid_zero_points += int(not valid_xint8_zero_point(dtype, value))
    return {
        "node_count": len(model.graph.node),
        "op_counts": dict(sorted(op_counts.items())),
        "domains": dict(sorted(domains.items())),
        "qlinearconv_count": int(op_counts.get("QLinearConv", 0)),
        "conv_count": int(op_counts.get("Conv", 0)),
        "quantize_count": sum(count for name, count in op_counts.items() if "QuantizeLinear" in name),
        "dequantize_count": sum(count for name, count in op_counts.items() if "DequantizeLinear" in name),
        "zero_point_value_count": len(zero_points),
        "nonzero_zero_point_count": sum(value != 0 for value in zero_points),
        "zero_point_histogram": dict(sorted(zero_point_histogram.items())),
        "invalid_xint8_zero_point_count": invalid_zero_points,
        "resize_count": int(op_counts.get("Resize", 0)),
        "add_count": int(op_counts.get("Add", 0)),
        "clip_count": int(op_counts.get("Clip", 0)),
    }


def validate_xint8_graph(model: onnx.ModelProto, report: dict[str, object], require_core_only: bool = False) -> None:
    onnx.checker.check_model(model)
    if int(report["qlinearconv_count"]):
        raise RuntimeError(
            f"output contains {report['qlinearconv_count']} QLinearConv nodes; "
            "Ryzen AI XINT8 does not support QLinearConv"
        )
    if int(report["conv_count"]) == 0:
        raise RuntimeError("output contains no Conv nodes")
    if int(report["quantize_count"]) == 0 or int(report["dequantize_count"]) == 0:
        raise RuntimeError("output has no complete Q/DQ structure")
    if int(report["invalid_xint8_zero_point_count"]):
        raise RuntimeError(
            "XINT8 zero-point encoding check failed: "
            f"{report['invalid_xint8_zero_point_count']} values are outside the accepted encoding; "
            f"histogram={report['zero_point_histogram']}"
        )
    if require_core_only:
        forbidden = {name: int(report[f"{name.lower()}_count"]) for name in ("Resize", "Add", "Clip") if int(report[f"{name.lower()}_count"])}
        if forbidden:
            raise RuntimeError(
                "core-only deployment graph contains post-processing operators; "
                f"move them to Magpie: {forbidden}"
            )


def import_quark():
    try:
        from quark.onnx import ModelQuantizer
    except Exception as exc:
        raise SystemExit(
            "[FAIL] AMD Quark is unavailable.\n"
            "       Double-click 01_Install_RyzenAI_INT8.cmd once, then retry.\n"
            f"       Import error: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        from quark.onnx import QConfig

        return ModelQuantizer, QConfig.get_default_config, "QConfig"
    except ImportError:
        try:
            from quark.onnx.quantization.config import Config, get_default_config

            def make_config(name: str):
                return Config(global_quant_config=get_default_config(name))

            return ModelQuantizer, make_config, "Config/get_default_config"
        except Exception as exc:
            raise SystemExit(
                "[FAIL] AMD Quark ONNX API is incompatible. Re-run 01_Install_RyzenAI_INT8.cmd.\n"
                f"       API error: {type(exc).__name__}: {exc}"
            ) from exc


def remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def static_nchw_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape") or len(tensor_type.shape.dim) != 4:
        return None
    dims = [int(dim.dim_value) for dim in tensor_type.shape.dim]
    return dims if all(dim > 0 for dim in dims) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="GameSR AMD Quark Ryzen AI INT8 builder")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--config", default="XINT8", choices=("XINT8",))
    parser.add_argument("--require-core-only", action="store_true")
    args = parser.parse_args()

    model_path = args.model.resolve()
    out_path = args.output.resolve()
    calibration_path = args.calibration.resolve()
    if not model_path.is_file():
        raise SystemExit(f"[FAIL] Float ONNX not found: {model_path}")
    if not calibration_path.exists():
        raise SystemExit(f"[FAIL] Calibration source not found: {calibration_path}")
    if args.count < 1:
        raise SystemExit("[FAIL] --count must be at least 1")

    float_model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(float_model)
    graph_input = next((value for value in float_model.graph.input if value.name), None)
    if graph_input is None:
        raise SystemExit("[FAIL] Model has no graph input")
    input_shape = static_nchw_shape(graph_input)
    expected_shape = [1, 3, args.height, args.width]
    if input_shape != expected_shape:
        raise SystemExit(
            f"[FAIL] Model input must be static NCHW {expected_shape}, found {input_shape}. "
            "Do not quantize a dynamic or differently sized model for the handheld."
        )
    paths = read_image_paths(calibration_path, args.count, args.seed)
    if not paths:
        raise SystemExit("[FAIL] No readable calibration images found")

    ModelQuantizer, make_config, api_style = import_quark()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_name(f".{out_path.stem}.quark-building.onnx")
    staged_path = out_path.with_name(f".{out_path.stem}.validated.onnx")
    for stale in (temporary_path, staged_path):
        remove_if_present(stale)
        remove_if_present(Path(str(stale) + ".data"))
        remove_if_present(Path(str(stale) + "_data"))

    print(f"[INT8] GameSR Ryzen AI builder v{BUILDER_VERSION}")
    print(f"[INT8] Quantizer: AMD Quark {package_version('amd-quark')} ({api_style})")
    print(f"[INT8] Configuration: {args.config} (NPU CNN, symmetric INT8)")
    print(f"[INT8] Float model: {model_path}")
    print(f"[INT8] Calibration images: {len(paths)}")
    print(f"[INT8] Input: {graph_input.name} -> {args.width}x{args.height} RGB NCHW FP32 [0,1]")

    reader = ImageCalibrationDataReader(paths, graph_input.name, args.width, args.height)
    quantizer = ModelQuantizer(make_config(args.config))
    try:
        quantizer.quantize_model(str(model_path), str(temporary_path), reader)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("AMD Quark did not create a non-empty ONNX file")
        quantized_model = onnx.load(str(temporary_path), load_external_data=True)
        report = graph_report(quantized_model)
        validate_xint8_graph(quantized_model, report, require_core_only=args.require_core_only)
        # GameSR models are small; publish a single self-contained ONNX.
        onnx.save_model(quantized_model, str(staged_path), save_as_external_data=False)
        onnx.checker.check_model(onnx.load(str(staged_path), load_external_data=True))
        os.replace(staged_path, out_path)
        # The published model is deliberately self-contained. Remove stale
        # sidecars left by an older export so they cannot be packaged by mistake.
        remove_if_present(Path(str(out_path) + ".data"))
        remove_if_present(Path(str(out_path) + "_data"))
    except Exception as exc:
        remove_if_present(staged_path)
        raise SystemExit(f"[FAIL] Ryzen AI INT8 build rejected: {type(exc).__name__}: {exc}") from exc
    finally:
        remove_if_present(temporary_path)
        remove_if_present(Path(str(temporary_path) + ".data"))
        remove_if_present(Path(str(temporary_path) + "_data"))

    meta = {
        "schema_version": 3,
        "created": datetime.now().isoformat(),
        "builder_version": BUILDER_VERSION,
        "builder": "AMD Quark for ONNX",
        "quark_version": package_version("amd-quark"),
        "onnx_version": package_version("onnx"),
        "onnxruntime_version": package_version("onnxruntime"),
        "python": sys.version.split()[0],
        "source_model": str(model_path),
        "source_sha256": sha256_file(model_path),
        "output_model": str(out_path),
        "output_sha256": sha256_file(out_path),
        "calibration_source": str(calibration_path),
        "calibration_count": len(paths),
        "seed": args.seed,
        "input_shape": [1, 3, args.height, args.width],
        "quantization_config": args.config,
        "target": "AMD Ryzen AI NPU CNN",
        "graph_validation": report,
        "requires_handheld_profile_validation": True,
        "deployment_contract": (
            "magpie_bilinear_residual_v1" if args.require_core_only else "complete_image_v1"
        ),
        "core_only": bool(args.require_core_only),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] ONNX checker: {out_path}")
    print(f"[PASS] QLinearConv: {report['qlinearconv_count']} (XINT8 requires 0)")
    print(f"[PASS] Conv/Q/DQ: {report['conv_count']}/{report['quantize_count']}/{report['dequantize_count']}")
    print(f"[PASS] XINT8 zero-point encoding: {report['zero_point_histogram']}")
    print(f"[PASS] Invalid XINT8 zero points: {report['invalid_xint8_zero_point_count']}")
    if args.require_core_only:
        print("[PASS] Core-only graph: Resize/Add/Clip = 0/0/0")
    print(f"[OK] Ryzen AI INT8 model: {out_path}")
    print(f"[OK] Metadata: {meta_path}")
    print("[NEXT] Copy this ONNX to Benchmark v2.4.1 and run Benchmark-INT8.cmd")


if __name__ == "__main__":
    main()
