from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Convert GameSR ONNX float model to FP16')
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    try:
        from onnxconverter_common import float16
    except Exception as exc:
        raise SystemExit('[FAIL] Missing onnxconverter-common. Install deployment dependencies first.') from exc
    import onnx
    src = Path(args.model).resolve(); out = Path(args.output).resolve()
    if not src.exists(): raise SystemExit(f'[FAIL] Model not found: {src}')
    model = onnx.load(str(src), load_external_data=True)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(converted, str(out))
    print(f'[OK] FP16 ONNX written: {out}')

if __name__ == '__main__': main()
