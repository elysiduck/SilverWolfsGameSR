from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=20
        ).strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] PyTorch import failed: {exc}")
        return 2

    in_project_venv = Path(sys.prefix).name.lower() == ".venv"
    cuda_ok = torch.cuda.is_available()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "project_venv": in_project_venv,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_ok,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if shutil.which("nvidia-smi")
        else "not found",
        "cuda_home_env": os.environ.get("CUDA_PATH", "not set (normally OK)"),
    }

    if cuda_ok:
        report.update(
            {
                "gpu_count": torch.cuda.device_count(),
                "gpu_0": torch.cuda.get_device_name(0),
                "gpu_0_vram_gib": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                ),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        x = torch.randn((1024, 1024), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        report["cuda_matrix_test"] = float(y[0, 0].item())

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not in_project_venv:
        print("[WARN] Python is not running from the project .venv")
    if not cuda_ok:
        print("[FAIL] PyTorch cannot use an NVIDIA CUDA GPU.")
        return 3
    print("[PASS] Isolated environment and CUDA training test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

