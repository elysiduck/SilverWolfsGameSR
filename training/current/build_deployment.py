from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime

MODELS = {
    'tiny-npu-fast': 'configs/tiny-npu-fast.yaml',
    'tiny-npu-quality': 'configs/tiny-npu-quality.yaml',
    'tiny-c8': 'configs/tiny-c8.yaml',
    'balanced-c12': 'configs/balanced-c12.yaml',
    'quality-c16': 'configs/quality-c16.yaml',
}


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validated_xint8(model_path: Path, float_model_path: Path) -> tuple[bool, dict | None, str]:
    meta_path = model_path.with_suffix(model_path.suffix + '.meta.json')
    if not model_path.is_file():
        return False, None, 'model is missing'
    if not meta_path.is_file():
        return False, None, 'XINT8 validation metadata is missing'
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception as exc:
        return False, None, f'metadata cannot be read: {type(exc).__name__}'
    graph = meta.get('graph_validation') or {}
    if meta.get('builder') != 'AMD Quark for ONNX':
        return False, meta, 'model was not built by AMD Quark'
    if meta.get('quantization_config') != 'XINT8':
        return False, meta, 'model is not XINT8'
    if graph.get('qlinearconv_count') != 0:
        return False, meta, 'QLinearConv is present'
    if graph.get('invalid_xint8_zero_point_count') != 0:
        return False, meta, 'XINT8 zero-point encoding is invalid or was not checked'
    if not graph.get('conv_count') or not graph.get('quantize_count') or not graph.get('dequantize_count'):
        return False, meta, 'required Conv/Q/DQ structure is missing'
    if meta.get('output_sha256') != sha256_file(model_path):
        return False, meta, 'ONNX hash does not match metadata'
    if not float_model_path.is_file():
        return False, meta, 'source FP32 ONNX is missing'
    if meta.get('source_sha256') != sha256_file(float_model_path):
        return False, meta, 'INT8 metadata belongs to an older FP32 ONNX'
    return True, meta, 'validated'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='profile/model name; custom C/L names are supported')
    ap.add_argument('--config', default=None, help='YAML config path for custom profiles')
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    name = args.model
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = root / config_path
    elif name in MODELS:
        config_path = root / MODELS[name]
    else:
        candidate = root / 'configs' / f'{name}.yaml'
        if not candidate.is_file():
            raise SystemExit(f'[FAIL] Config not found for custom profile {name}: pass --config explicitly.')
        config_path = candidate
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    core_only = bool(config.get('deployment', {}).get('core_only', False))
    src_dir = root / 'models'
    out = root / 'deployment' / name
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for suffix in [f'{name}-720p-to-1080p.onnx', f'{name}-720p-to-1080p.onnx.data', f'{name}-720p-to-1080p-fp16.onnx', f'{name}-720p-to-1080p-fp16.onnx.data']:
        if copy_if_exists(src_dir / suffix, out / suffix):
            files.append(suffix)

    int8_name = f'{name}-720p-to-1080p-int8-qdq.onnx'
    int8_path = src_dir / int8_name
    float_model_path = src_dir / f'{name}-720p-to-1080p.onnx'
    int8_ok, int8_meta, int8_reason = validated_xint8(int8_path, float_model_path)
    if int8_ok and core_only and not bool(int8_meta.get('core_only', False)):
        int8_ok, int8_reason = False, 'metadata does not declare a core-only NPU graph'
    if int8_ok and core_only:
        graph = int8_meta.get('graph_validation') or {}
        forbidden = {op: int(graph.get(f'{op}_count', 0)) for op in ('resize', 'add', 'clip')}
        if any(forbidden.values()):
            int8_ok, int8_reason = False, f'core-only graph contains post-processing ops: {forbidden}'
    if int8_ok:
        for suffix in [int8_name, int8_name + '.data', int8_name + '.meta.json']:
            if copy_if_exists(src_dir / suffix, out / suffix):
                files.append(suffix)
    else:
        for suffix in [int8_name, int8_name + '.data', int8_name + '.meta.json']:
            stale = out / suffix
            if stale.exists():
                stale.unlink()
        print(f'[WARN] INT8 was not packaged: {int8_reason}')
        if core_only:
            raise SystemExit(f'[FAIL] NPU deployment requires a validated core-only XINT8 model: {int8_reason}')

    result = {
        'schema_version': 3,
        'created': datetime.now().isoformat(),
        'model': name,
        'input': {'shape': [1, 3, 720, 1280], 'layout': 'NCHW', 'dtype': 'float32'},
        'output': {
            'shape': [1, 3, 720, 1280] if core_only else [1, 3, 1080, 1920],
            'layout': 'NCHW',
            'dtype': 'float32',
            'meaning': 'low_resolution_rgb_residual' if core_only else 'display_rgb',
        },
        'deployment_contract': config.get('deployment', {}).get('postprocess', 'complete_image_v1'),
        'core_only': core_only,
        'postprocess': {
            'owner': 'Magpie shader' if core_only else 'ONNX graph',
            'formula': 'clamp(bilinear(lr, 1.5) + bilinear(residual_lr, 1.5), 0, 1)' if core_only else None,
            'coordinate_mode': 'half_pixel / align_corners=false' if core_only else None,
        },
        'models': files,
        'int8': {
            'packaged': int8_ok,
            'validation': int8_reason,
            'builder': int8_meta.get('builder') if int8_meta else None,
            'quark_version': int8_meta.get('quark_version') if int8_meta else None,
            'quantization_config': int8_meta.get('quantization_config') if int8_meta else None,
            'sha256': int8_meta.get('output_sha256') if int8_meta else None,
        },
        'npu_benchmark_note': 'Benchmark measures the core ONNX only for core-only models. Final Magpie FPS also includes capture, transfers, shader postprocess and display.',
    }
    (out / 'deployment_manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] Deployment bundle: {out}')
    for item in files:
        print(f'  - {item}')

if __name__ == '__main__':
    main()
