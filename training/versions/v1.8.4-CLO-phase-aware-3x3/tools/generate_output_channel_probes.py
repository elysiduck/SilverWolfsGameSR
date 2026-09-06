from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

BUILDER_VERSION = "1.0.1"


class OutputChannelProbe(nn.Module):
    """GameSR npu_sequential core with a variable final output width."""

    def __init__(self, channels: int, layers: int, output_channels: int):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be >= 2")
        if channels < 1 or output_channels < 1:
            raise ValueError("channel counts must be >= 1")
        seq: list[nn.Module] = [
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.ReLU(inplace=False),
        ]
        for _ in range(layers - 2):
            seq += [nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(inplace=False)]
        seq += [nn.Conv2d(channels, output_channels, 3, 1, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(c: int, layers: int, out_c: int) -> int:
    first = 3 * c * 9 + c
    middle = max(0, layers - 2) * (c * c * 9 + c)
    last = c * out_c * 9 + out_c
    return first + middle + last


def gmac_per_frame(c: int, layers: int, out_c: int, h: int, w: int) -> float:
    mac_per_pixel = 3 * c * 9 + max(0, layers - 2) * c * c * 9 + c * out_c * 9
    return h * w * mac_per_pixel / 1e9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_calibration(dst: Path, width: int, height: int, count: int, seed: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # Reuse existing files when the requested count is already present.
    existing = sorted(dst.glob("cal_*.png"))
    if len(existing) >= count:
        return
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    for i in range(count):
        noise = rng.random((height, width, 3), dtype=np.float32)
        grad = np.stack(
            [
                np.broadcast_to(xx, (height, width)),
                np.broadcast_to(yy, (height, width)),
                np.broadcast_to((xx + yy) / 2, (height, width)),
            ],
            axis=2,
        )
        arr = np.clip(0.55 * noise + 0.45 * np.roll(grad, i % 3, axis=2), 0, 1)
        Image.fromarray((arr * 255).astype(np.uint8), "RGB").save(dst / f"cal_{i:02d}.png")


def parse_ints(value: str) -> list[int]:
    vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    if len(vals) != len(set(vals)):
        raise ValueError("duplicate values are not allowed")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GameSR C16-L4 output-channel NPU probes")
    ap.add_argument("--root", type=Path, required=True, help="GameSR XDNA benchmark root")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--outputs", default="3,4,6,8")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cal-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-package", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    # The FIX patch bundles the exact v1.3 quantizer beside this builder, so
    # model generation no longer depends on where the patch was extracted.
    qscript = Path(__file__).with_name("int8_ptq.py").resolve()
    if not qscript.is_file():
        raise SystemExit(
            f"[FAIL] Missing bundled GameSR quantizer: {qscript}\n"
            "       Re-extract the OutputChannelProbe v1.0.1 FIX patch."
        )

    outputs = parse_ints(args.outputs)
    if 3 not in outputs:
        print("[WARN] O3 baseline is not in --outputs; delta-vs-O3 analysis will be unavailable.")

    build_root = root / "output_channel_build"
    payload = build_root / "handheld_payload"
    models_root = payload / "models"
    float_root = build_root / "float_models"
    cal_root = build_root / "calibration_synthetic"
    models_root.mkdir(parents=True, exist_ok=True)
    float_root.mkdir(parents=True, exist_ok=True)
    make_calibration(cal_root, args.width, args.height, args.cal_count, args.seed)

    manifest_models: list[dict[str, object]] = []
    baseline_output_elements = args.width * args.height * 3

    for out_c in outputs:
        arch = f"C{args.channels}-L{args.layers}-O{out_c}"
        out_dir = models_root / arch
        out_dir.mkdir(parents=True, exist_ok=True)
        float_model = float_root / f"{arch}.onnx"
        quant_model = out_dir / f"{arch}-int8-qdq.onnx"

        if not (args.skip_existing and quant_model.is_file()):
            # Reset the seed for every O variant so all common layers have the
            # same initial weights. Only the final Conv shape changes.
            torch.manual_seed(args.seed)
            model = OutputChannelProbe(args.channels, args.layers, out_c).eval()
            x = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
            print(f"[EXPORT] {arch}", flush=True)
            torch.onnx.export(
                model,
                x,
                str(float_model),
                input_names=["input"],
                output_names=["feature" if out_c != 3 else "residual"],
                opset_version=17,
                do_constant_folding=True,
            )
            cmd = [
                sys.executable,
                str(qscript),
                "--model",
                str(float_model),
                "--output",
                str(quant_model),
                "--calibration",
                str(cal_root),
                "--count",
                str(args.cal_count),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--require-core-only",
            ]
            print(f"[QUANT]  {arch}", flush=True)
            subprocess.run(cmd, check=True)

        if not quant_model.is_file():
            raise SystemExit(f"[FAIL] Quantized model missing after build: {quant_model}")

        output_elements = args.width * args.height * out_c
        rel = quant_model.relative_to(payload).as_posix()
        manifest_models.append(
            {
                "architecture": arch,
                "channels": args.channels,
                "conv_layers": args.layers,
                "output_channels": out_c,
                "parameters": parameter_count(args.channels, args.layers, out_c),
                "gmac_per_frame": round(
                    gmac_per_frame(args.channels, args.layers, out_c, args.height, args.width), 6
                ),
                "output_elements": output_elements,
                "output_fp32_mib": round(output_elements * 4 / (1024 * 1024), 6),
                "output_ratio_vs_o3": round(output_elements / baseline_output_elements, 6),
                "model_relpath": rel,
                "sha256": sha256_file(quant_model),
                "status": "READY",
            }
        )

    created = datetime.now().isoformat()
    manifest = {
        "schema_version": 1,
        "probe": "GameSR NPU Output Channel Probe",
        "probe_version": BUILDER_VERSION,
        "builder": "GameSR Output Channel Probe Builder",
        "builder_version": BUILDER_VERSION,
        "architecture": f"C{args.channels}-L{args.layers}-O{{3,4,6,8}}",
        "purpose": "Isolate final output-channel width cost while keeping the C/L body fixed",
        "created": created,
        "input_shape": [1, 3, args.height, args.width],
        "output_spatial_shape": [args.height, args.width],
        "fixed_channels": args.channels,
        "fixed_conv_layers": args.layers,
        "output_channels": outputs,
        "requested_models": len(outputs),
        "models": manifest_models,
        "notes": [
            "O3 is the current RGB residual baseline.",
            "O4/O6/O8 are feature-output probes only; they are not trained quality models.",
            "This payload measures NPU graph/runtime cost. It does not measure Magpie CPU/GPU transfer or learned-decoder cost.",
        ],
    }
    payload.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (payload / "output_channel_manifest.json").write_text(manifest_text, encoding="utf-8")
    # Compatibility copy: the existing architecture runner can still discover it,
    # although the dedicated output-channel runner gives the useful O-column report.
    (payload / "sweep_manifest.json").write_text(manifest_text, encoding="utf-8")
    (payload / "README_HANDHELD.txt").write_text(
        "GameSR NPU Output Channel Probe payload\n"
        "========================================\n\n"
        f"Fixed body: C{args.channels}-L{args.layers}\n"
        f"Output channels: {','.join(map(str, outputs))}\n"
        f"Input: 1x3x{args.height}x{args.width}\n\n"
        "Run this payload with RUN-GAMESR-OUTPUT-CHANNEL-PROBE.cmd in the patched benchmark root.\n"
        "The handheld does not need Python when using this prebuilt payload.\n",
        encoding="utf-8",
    )

    print(f"[OK] Payload: {payload}")
    if args.no_package:
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = root / f"GameSR-NPU-Output-Channel-Probe-Models-{len(outputs)}-{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(payload.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(build_root).as_posix())
    print(f"[OK] Handheld ZIP: {zip_path}")
    print("[NEXT] Copy that ZIP next to RUN-GAMESR-OUTPUT-CHANNEL-PROBE.cmd on the handheld and run it.")


if __name__ == "__main__":
    main()
