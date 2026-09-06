from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from gamesr.data import edge_loss, gpu_game_degradation, ssim_loss
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
