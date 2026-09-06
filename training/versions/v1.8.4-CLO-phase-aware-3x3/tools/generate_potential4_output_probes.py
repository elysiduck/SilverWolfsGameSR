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

BUILDER_VERSION = "1.1.0"
DEFAULT_OUTPUTS = [3, 4, 6, 8]


class OutputChannelProbe(nn.Module):
    """Exact GameSR npu_sequential core with a variable final output width."""

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


def parse_outputs(text: str) -> list[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals or len(vals) != len(set(vals)) or any(v < 1 for v in vals):
        raise ValueError("--outputs must be a unique comma-separated list of positive integers")
    return vals


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bbs = list(data.get("backbones", []))
    if len(bbs) != 4:
        raise SystemExit(f"[FAIL] Expected exactly four selected backbones, found {len(bbs)}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GameSR Potential-4 output-channel probes")
    ap.add_argument("--root", type=Path, required=True, help="PC probe package root")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--outputs", default="3,4,6,8")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cal-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-package", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    config_path = (args.config or (root / "config" / "potential_backbones.json")).resolve()
    qscript = Path(__file__).with_name("int8_ptq.py").resolve()
    if not qscript.is_file():
        raise SystemExit(f"[FAIL] Missing bundled quantizer: {qscript}")
    if not config_path.is_file():
        raise SystemExit(f"[FAIL] Missing backbone config: {config_path}")

    cfg = load_config(config_path)
    outputs = parse_outputs(args.outputs)
    if 3 not in outputs:
        print("[WARN] O3 is absent, so per-backbone delta-vs-O3 will be unavailable.")

    build_root = root / "output_potential4"
    payload = build_root / "handheld_payload"
    models_root = payload / "models"
    float_root = build_root / "float_models"
    cal_root = build_root / "calibration_synthetic"
    models_root.mkdir(parents=True, exist_ok=True)
    float_root.mkdir(parents=True, exist_ok=True)
    make_calibration(cal_root, args.width, args.height, args.cal_count, args.seed)

    baseline_output_elements = args.width * args.height * 3
    manifest_models: list[dict[str, object]] = []

    for bb in sorted(cfg["backbones"], key=lambda x: int(x["order"])):
        c = int(bb["channels"])
        layers = int(bb["conv_layers"])
        backbone = str(bb["name"])
        # Same seed inside one backbone => common layers are identical across O variants.
        backbone_seed = int(args.seed) + c * 100 + layers
        for out_c in outputs:
            arch = f"{backbone}-O{out_c}"
            out_dir = models_root / arch
            out_dir.mkdir(parents=True, exist_ok=True)
            float_model = float_root / f"{arch}.onnx"
            quant_model = out_dir / f"{arch}-int8-qdq.onnx"

            if not (args.skip_existing and quant_model.is_file()):
                torch.manual_seed(backbone_seed)
                model = OutputChannelProbe(c, layers, out_c).eval()
                x = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
                print(f"[EXPORT] {arch}", flush=True)
                torch.onnx.export(
                    model,
                    x,
                    str(float_model),
                    input_names=["input"],
                    output_names=["residual" if out_c == 3 else "feature"],
                    opset_version=17,
                    do_constant_folding=True,
                )
                cmd = [
                    sys.executable,
                    str(qscript),
                    "--model", str(float_model),
                    "--output", str(quant_model),
                    "--calibration", str(cal_root),
                    "--count", str(args.cal_count),
                    "--width", str(args.width),
                    "--height", str(args.height),
                    "--require-core-only",
                ]
                print(f"[QUANT]  {arch}", flush=True)
                subprocess.run(cmd, check=True)

            if not quant_model.is_file():
                raise SystemExit(f"[FAIL] Quantized model missing: {quant_model}")

            output_elements = args.width * args.height * out_c
            manifest_models.append({
                "candidate_order": int(bb["order"]),
                "backbone": backbone,
                "tier": bb.get("tier", ""),
                "architecture": arch,
                "channels": c,
                "conv_layers": layers,
                "output_channels": out_c,
                "parameters": parameter_count(c, layers, out_c),
                "gmac_per_frame": round(gmac_per_frame(c, layers, out_c, args.height, args.width), 6),
                "output_elements": output_elements,
                "output_fp32_mib": round(output_elements * 4 / (1024 * 1024), 6),
                "output_ratio_vs_o3": round(output_elements / baseline_output_elements, 6),
                "source_o3_average_ms": float(bb["source_average_ms"]),
                "source_o3_parameters": int(bb["source_parameters_o3"]),
                "source_o3_gmac": float(bb["source_gmac_o3"]),
                "model_relpath": quant_model.relative_to(payload).as_posix(),
                "sha256": sha256_file(quant_model),
                "status": "READY",
            })

    manifest = {
        "schema_version": 2,
        "probe": "GameSR Potential-4 Output Channel Probe",
        "probe_version": BUILDER_VERSION,
        "source_sweep": cfg.get("source_sweep"),
        "selection_policy": cfg.get("selection_policy"),
        "created": datetime.now().isoformat(),
        "input_shape": [1, 3, args.height, args.width],
        "output_spatial_shape": [args.height, args.width],
        "output_channels": outputs,
        "candidate_backbones": cfg["backbones"],
        "requested_models": len(manifest_models),
        "models": manifest_models,
        "notes": [
            "Only four surviving C/L candidates are probed: C8-L5, C16-L4, C24-L3, C32-L3.",
            "Each backbone gets O3/O4/O6/O8 using the same common-layer random weights within that backbone.",
            "O3 is a controlled baseline generated in the same build, not reused from the old 105-model payload.",
            "This probe measures pure NPU graph/runtime cost only; it does not measure Magpie Map/stitch/upload or learned GPU decoder cost.",
        ],
    }
    payload.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (payload / "potential4_output_manifest.json").write_text(text, encoding="utf-8")
    (payload / "README_HANDHELD_PAYLOAD.txt").write_text(
        "GameSR Potential-4 Output Channel Probe payload\n"
        "================================================\n\n"
        "Backbones: C8-L5, C16-L4, C24-L3, C32-L3\n"
        f"Outputs: {','.join(map(str, outputs))}\n"
        f"Models: {len(manifest_models)}\n"
        f"Input: 1x3x{args.height}x{args.width}\n\n"
        "Copy this ZIP to the handheld benchmark root and run the separate HANDHELD patch.\n",
        encoding="utf-8",
    )

    if args.no_package:
        print(f"[OK] Payload folder: {payload}")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    zip_path = build_root / f"GameSR-NPU-Potential4-Output-Probe-Models-{len(manifest_models)}-{stamp}-{digest}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(payload.rglob("*")):
            if p.is_file():
                zf.write(p, Path("handheld_payload") / p.relative_to(payload))
    print("")
    print(f"[OK] Built {len(manifest_models)} probes")
    print(f"[OK] Payload ZIP: {zip_path}")
    print("[NEXT] Copy ONLY that generated Models-16 ZIP to the handheld benchmark root.")


if __name__ == "__main__":
    main()
