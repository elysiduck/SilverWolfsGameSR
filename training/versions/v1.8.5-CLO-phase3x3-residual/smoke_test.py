from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from gamesr.data import (
    charbonnier_loss, edge_loss, gpu_game_degradation, high_frequency_loss,
    multi_scale_ssim_loss, ringing_loss, ssim_loss,
)
from gamesr.model import build_model
from gamesr.utils import load_config


def main() -> None:
    root = Path(__file__).resolve().parent
    profiles = {
        "tiny-npu-fast.yaml": 3947,
        "tiny-npu-quality.yaml": 7451,
    }
    models = {}
    for filename, expected_parameters in profiles.items():
        config = load_config(root / "configs" / filename)
        candidate = build_model(config)
        actual_parameters = sum(parameter.numel() for parameter in candidate.parameters())
        if actual_parameters != expected_parameters:
            raise SystemExit(
                f"[FAIL] {filename}: expected {expected_parameters} parameters, found {actual_parameters}"
            )
        models[filename] = candidate
    print("[PASS] NPU Fast and NPU Quality YAML configs load correctly")

    phase_config = {
        "scale": 1.5,
        "model": {"channels": 8, "architecture": "npu_sequential", "conv_layers": 3, "output_channels": 4},
        "deployment": {"postprocess": "phase_aware_3x3_v1", "decoder": {"base_skip": 1.0}},
    }
    phase_model = build_model(phase_config).train()
    phase_lr = torch.rand(2, 3, 8, 10)
    with torch.no_grad():
        initial = phase_model(phase_lr)
        baseline = torch.nn.functional.interpolate(
            phase_lr, scale_factor=1.5, mode="bilinear", align_corners=False
        )
    if not torch.equal(initial, baseline):
        raise SystemExit("[FAIL] Zero-initialized phase decoder is not an exact bilinear baseline.")
    print("[PASS] Phase-aware v2 starts as the exact bilinear baseline")
    phase_target = torch.rand(2, 3, 12, 15)
    phase_prediction = phase_model(phase_lr)
    phase_loss = (
        charbonnier_loss(phase_prediction, phase_target)
        + 0.10 * multi_scale_ssim_loss(phase_prediction, phase_target)
        + 0.05 * edge_loss(phase_prediction, phase_target)
        + 0.02 * high_frequency_loss(phase_prediction, phase_target)
        + 0.01 * ringing_loss(phase_prediction, phase_target)
    )
    phase_loss.backward()
    if phase_prediction.shape != phase_target.shape or phase_model.decoder.weight.grad is None:
        raise SystemExit("[FAIL] Phase-aware 3x3 decoder forward/backward contract failed.")
    print("[PASS] Phase-aware 3x3 decoder + new composite loss forward/backward")

    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA GPU unavailable. Run check_env.py first.")
    device = torch.device("cuda")
    model = models["tiny-npu-fast.yaml"].to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    lr = torch.rand(2, 3, 128, 128, device=device)
    target = torch.rand(2, 3, 192, 192, device=device)
    optimizer.zero_grad(set_to_none=True)
    lr_from_hr = gpu_game_degradation(target, 1.5, {"resize_kernels": ["bicubic"], "blur_probability": 0.0, "sharpen_probability": 0.0, "post_sharpen_probability": 0.0, "brightness_probability": 0.0, "contrast_probability": 0.0, "quantization_probability": 0.0, "noise_probability": 0.0})
    prediction = model(lr_from_hr)
    residual_core = model.forward_core(lr_from_hr)
    loss = torch.nn.functional.l1_loss(prediction, target) + 0.05 * edge_loss(prediction, target) + 0.02 * ssim_loss(prediction, target)
    loss.backward()
    optimizer.step()
    if prediction.shape != target.shape or residual_core.shape != lr_from_hr.shape or not torch.isfinite(loss):
        raise SystemExit("[FAIL] Invalid output from training step.")
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "smoke.pth"
        torch.save({"model": model.state_dict()}, checkpoint)
        if checkpoint.stat().st_size == 0:
            raise SystemExit("[FAIL] Checkpoint save failed.")
    allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[PASS] NPU sequential full-training output + LR core + backward + checkpoint | loss={loss.item():.5f} | peak={allocated:.1f} MiB")


if __name__ == "__main__":
    main()
