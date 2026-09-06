from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from gamesr.data import GameSRDataset, read_manifest, ssim_loss, to_tensor
from gamesr.model import build_model
from gamesr.utils import load_config, save_preview
from PIL import Image


def resolve_config(config: dict, config_path: Path) -> dict:
    base = config_path.parent.parent
    config["data"]["root"] = str((base / config["data"]["root"]).resolve()) if not Path(config["data"]["root"]).is_absolute() else config["data"]["root"]
    config["output"] = str((base / config["output"]).resolve()) if not Path(config["output"]).is_absolute() else config["output"]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Full GameSR test benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all test images")
    parser.add_argument("--preview-count", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = resolve_config(load_config(config_path), config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("ema") if state.get("ema") is not None else state["model"]
    model.load_state_dict(weights, strict=True)

    files = read_manifest(Path(config["data"]["root"]) / "test.csv")
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise SystemExit("Test manifest is empty.")

    preview_dir = Path(config["output"]) / "test_preview"
    psnrs, ssims, bilinear_psnrs = [], [], []
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    elapsed_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    elapsed_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if elapsed_start is not None:
        elapsed_start.record()

    with torch.inference_mode():
        for index, path in enumerate(tqdm(files, desc="test")):
            with Image.open(path) as image:
                hr_img = image.convert("RGB")
            w, h = hr_img.size
            w -= w % 3
            h -= h % 3
            hr_img = hr_img.crop((0, 0, w, h))
            lr_img = hr_img.resize((round(w / config["scale"]), round(h / config["scale"])), Image.Resampling.LANCZOS)
            lr = to_tensor(lr_img).unsqueeze(0).to(device)
            hr = to_tensor(hr_img).unsqueeze(0).to(device)
            base = lr_img.resize(hr_img.size, Image.Resampling.BILINEAR)
            base_tensor = to_tensor(base).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(lr)
            mse = torch.nn.functional.mse_loss(pred.float(), hr.float()).item()
            base_mse = torch.nn.functional.mse_loss(base_tensor.float(), hr.float()).item()
            psnrs.append(-10 * math.log10(max(mse, 1e-12)))
            bilinear_psnrs.append(-10 * math.log10(max(base_mse, 1e-12)))
            ssims.append(float((1.0 - ssim_loss(pred.float(), hr.float())).item()))
            if index < args.preview_count:
                save_preview(lr[0].cpu(), pred[0].cpu(), hr[0].cpu(), preview_dir / f"{index:03d}.png")

    if elapsed_end is not None:
        elapsed_end.record()
        torch.cuda.synchronize()
        total_seconds = elapsed_start.elapsed_time(elapsed_end) / 1000.0
    else:
        total_seconds = float("nan")

    mean_psnr = sum(psnrs) / len(psnrs)
    mean_bilinear = sum(bilinear_psnrs) / len(bilinear_psnrs)
    result = {
        "test_images": len(files),
        "psnr_db": round(mean_psnr, 4),
        "ssim": round(sum(ssims) / len(ssims), 6),
        "bilinear_psnr_db": round(mean_bilinear, 4),
        "delta_psnr_db": round(mean_psnr - mean_bilinear, 4),
        "inference_seconds": None if math.isnan(total_seconds) else round(total_seconds, 4),
        "inference_fps": None if math.isnan(total_seconds) else round(len(files) / max(total_seconds, 1e-9), 3),
    }
    result_path = Path(config["output"]) / "test_results.json"
    result_path.write_text(__import__("json").dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Test images: {len(files)}")
    print(f"GameSR PSNR: {mean_psnr:.3f} dB")
    print(f"GameSR SSIM: {sum(ssims) / len(ssims):.4f}")
    print(f"Bilinear PSNR: {mean_bilinear:.3f} dB")
    print(f"Delta PSNR: {mean_psnr - mean_bilinear:+.3f} dB")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
