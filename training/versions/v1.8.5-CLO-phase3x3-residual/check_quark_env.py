from __future__ import annotations

import importlib.metadata
import json
import platform
import sys


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "amd_quark": version("amd-quark"),
        "onnx": version("onnx"),
        "onnxruntime": version("onnxruntime"),
        "numpy": version("numpy"),
        "Pillow": version("Pillow"),
        "torch": version("torch"),
    }
    try:
        import torch
        from quark.onnx import ModelQuantizer

        if not torch.__version__.startswith("2.11."):
            raise RuntimeError(
                f"AMD Quark CPU wheel requires PyTorch 2.11.x, found {torch.__version__}"
            )
        try:
            from quark.onnx import QConfig

            config = QConfig.get_default_config("XINT8")
            api = "QConfig"
        except ImportError:
            from quark.onnx.quantization.config import Config, get_default_config

            config = Config(global_quant_config=get_default_config("XINT8"))
            api = "Config/get_default_config"
        if ModelQuantizer(config) is None:
            raise RuntimeError("ModelQuantizer construction returned None")
        report["quark_onnx_api"] = api
        report["xint8_config"] = "OK"
    except Exception as exc:
        report["xint8_config"] = f"FAIL: {type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("[FAIL] AMD Quark ONNX/XINT8 verification failed")
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[PASS] AMD Quark ONNX/XINT8 environment is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
