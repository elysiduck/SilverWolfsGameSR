from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a GameSR ONNX graph and its external data")
    p.add_argument("onnx_path", type=Path)
    args = p.parse_args()
    path = args.onnx_path.resolve()
    if not path.exists():
        raise SystemExit(f"ONNX not found: {path}")
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model)
    print(f"[OK] ONNX checker passed: {path}")
    print(f"Nodes: {len(model.graph.node)}")
    if model.graph.input:
        print("Input:", model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
    if model.graph.output:
        print("Output:", model.graph.output[0].name, model.graph.output[0].type.tensor_type.shape)
    data_path = Path(str(path) + ".data")
    print(f"External data: {'FOUND' if data_path.exists() else 'not required / missing'}")


if __name__ == "__main__":
    main()
