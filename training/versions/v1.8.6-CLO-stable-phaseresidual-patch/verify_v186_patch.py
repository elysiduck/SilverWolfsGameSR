from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checks = []

def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(': ' + detail) if detail else ''}")

required = [
    ROOT / "train.py",
    ROOT / "pipeline.py",
    ROOT / "gamesr" / "data.py",
    ROOT / "gamesr" / "utils.py",
    ROOT / "configs" / "gamesr-c16-l4-o4-phase3x3.yaml",
]
for path in required:
    check(f"file {path.relative_to(ROOT)}", path.exists())

for path in required[:4]:
    if path.exists():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            check(f"syntax {path.relative_to(ROOT)}", True)
        except Exception as exc:
            check(f"syntax {path.relative_to(ROOT)}", False, str(exc))

if required[-1].exists():
    text = required[-1].read_text(encoding="utf-8")
    expected = {
        "trainer_version: 1.8.6": "trainer version",
        "phase_residual_stable_v186": "training contract",
        "base_skip: 1.0": "bilinear skip",
        "output: rgb_residual": "residual decoder",
        "ms_ssim_weight: 0.03": "MS-SSIM weight",
        "gradient_weight: 0.02": "gradient weight",
        "frequency_weight: 0.004": "frequency weight",
        "ringing_weight: 0.02": "ringing weight",
        "clean_anchor_end: 0.75": "clean anchor",
        "start_strength: 0.1": "degradation start",
        "full_strength_at: 0.88": "degradation full point",
        "max_mix: 0.04": "hard mining max mix",
        "start_progress: 0.7": "early-stop start",
        "epochs: 100": "default max epochs",
    }
    for needle, label in expected.items():
        check(label, needle in text, needle)

source = (ROOT / "train.py").read_text(encoding="utf-8") if (ROOT / "train.py").exists() else ""
for needle, label in [
    ("build_phase_loss_policy", "phase polish scheduler"),
    ("update_quality_controller", "quality brake controller"),
    ("best_quality.pth", "quality checkpoint"),
    ("best_psnr.pth", "PSNR checkpoint"),
    ("[V1.8.6 PHASE]", "v1.8.6 phase logging"),
]:
    check(label, needle in source)

data_source = (ROOT / "gamesr" / "data.py").read_text(encoding="utf-8") if (ROOT / "gamesr" / "data.py").exists() else ""
for needle, label in [
    ('curriculum_mode", "legacy")).lower() == "stable_v2"', "stable degradation mode"),
    ("probability_scale = strength", "true degradation probability ramp"),
    ("clean_anchor_end", "per-batch clean anchor"),
]:
    check(label, needle in data_source)

utils_source = (ROOT / "gamesr" / "utils.py").read_text(encoding="utf-8") if (ROOT / "gamesr" / "utils.py").exists() else ""
check("checkpoint training contract", '"training_contract": config.get("training_contract")' in utils_source)
check("no hidden 100-to-80 epoch rewrite", 'tcfg["epochs"] = 80' not in source)
pipeline_source = (ROOT / "pipeline.py").read_text(encoding="utf-8") if (ROOT / "pipeline.py").exists() else ""
check("pipeline fallback max epochs is 100", 'phase_train["epochs"] = 100' in pipeline_source)
check("interactive epoch default is 100", 'input("Epochs [100]: ")' in pipeline_source)

if all(checks):
    print("\nGameSR v1.8.6 stable-training patch verification PASSED.")
    sys.exit(0)
print("\nGameSR v1.8.6 patch verification FAILED. Re-extract the patch and overwrite files.")
sys.exit(1)
