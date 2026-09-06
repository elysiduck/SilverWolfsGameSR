from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows console compatibility: torch.onnx may emit Unicode symbols.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import onnx
import torch

from gamesr.model import build_model
from gamesr.utils import checkpoint_signature, load_config


class CoreOnlyExport(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model.forward_core(lr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GameSR checkpoint to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--core-only", action="store_true", help="export the LR residual core for Ryzen AI")
    mode.add_argument("--full-output", action="store_true", help="export the legacy complete 1080p graph")
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config).cpu().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = state.get("config") or {}
    if saved_config and checkpoint_signature(saved_config) != checkpoint_signature(config):
        raise SystemExit(
            "[FAIL] Checkpoint belongs to a different code version/model configuration. "
            "Use the matching config or train this profile from scratch."
        )
    weights = state.get("ema") if args.use_ema and state.get("ema") is not None else state["model"]
    model.load_state_dict(weights, strict=True)

    deployment = config.get("deployment", {})
    core_only = bool(deployment.get("core_only", False))
    if args.core_only:
        core_only = True
    elif args.full_output:
        core_only = False
    export_model = CoreOnlyExport(model) if core_only else model
    output_channels = int(config.get("model", {}).get("output_channels", 3))
    output_name = ("residual_lr" if output_channels == 3 else "latent_lr") if core_only else "sr"

    lr_h, lr_w = 720, 1280
    example = torch.rand(1, 3, lr_h, lr_w)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "lr": {2: "height", 3: "width"},
            output_name: {2: "height" if core_only else "out_height", 3: "width" if core_only else "out_width"},
        }
    try:
        torch.onnx.export(
            export_model,
            example,
            args.output,
            input_names=["lr"],
            output_names=[output_name],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    except UnicodeEncodeError as exc:
        raise SystemExit(
            "ONNX export hit a console encoding error. v1.4.2 forces UTF-8; "
            "run from GameSR_Export.cmd or set PYTHONUTF8=1."
        ) from exc

    checked = onnx.load(args.output, load_external_data=True)
    onnx.checker.check_model(checked)
    if core_only:
        op_types = [node.op_type for node in checked.graph.node]
        forbidden = sorted({name for name in ("Resize", "Add", "Clip") if name in op_types})
        if forbidden:
            args.output.unlink(missing_ok=True)
            raise SystemExit(f"[FAIL] Core-only export contains forbidden deployment ops: {forbidden}")
    print(f"ONNX OK: {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")
    print("Input:  [1, 3, 720, 1280] RGB FP32 [0,1]")
    if core_only:
        print(f"Output: [1, {output_channels}, 720, 1280] FP32 " + ("RGB residual" if output_channels == 3 else "latent"))
        if output_channels == 3:
            print("Contract: display = clamp(bilinear(lr, 1.5) + bilinear(residual_lr, 1.5), 0, 1)")
        else:
            decoder_path = args.output.with_suffix(args.output.suffix + ".decoder.json")
            decoder = model.decoder
            payload = {
                "schema_version": 1,
                "type": "bilinear_then_1x1",
                "input_channels": output_channels,
                "output_channels": 3,
                "coordinate_mode": "half_pixel_align_corners_false",
                "weight": decoder.weight.detach().cpu().view(3, output_channels).tolist(),
                "bias": decoder.bias.detach().cpu().tolist(),
            }
            decoder_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Decoder: {decoder_path}")
            print("Contract: display = clamp(bilinear(lr, 1.5) + decoder_1x1(bilinear(latent_lr, 1.5)), 0, 1)")
        print("[PASS] Core-only export: Resize/Add/Clip are outside the ONNX graph")
    else:
        print("Output: [1, 3, 1080, 1920] RGB FP32 [0,1]")
    data_path = Path(str(args.output) + ".data")
    if data_path.exists():
        print(f"External data: {data_path} ({data_path.stat().st_size / 1024**2:.2f} MiB)")
    print("Keep .onnx and .onnx.data together when using external data.")


if __name__ == "__main__":
    main()
