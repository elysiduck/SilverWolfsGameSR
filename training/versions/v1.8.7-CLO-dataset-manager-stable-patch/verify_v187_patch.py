from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
checks: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append(bool(ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(': ' + detail) if detail else ''}")


required = [
    ROOT / "train.py",
    ROOT / "pipeline.py",
    ROOT / "prepare_dataset.py",
    ROOT / "gamesr" / "data.py",
    ROOT / "gamesr" / "utils.py",
    ROOT / "configs" / "gamesr-c16-l4-o4-phase3x3.yaml",
]
for path in required:
    check(f"file {path.relative_to(ROOT)}", path.exists())

for path in required[:5]:
    if path.exists():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            check(f"syntax {path.relative_to(ROOT)}", True)
        except Exception as exc:
            check(f"syntax {path.relative_to(ROOT)}", False, str(exc))

config_text = required[-1].read_text(encoding="utf-8") if required[-1].exists() else ""
expected_config = {
    "trainer_version: 1.8.7": "trainer version",
    "phase_residual_stable_v187": "training contract",
    "phase3x3r-v4": "new experiment suffix",
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
for needle, label in expected_config.items():
    check(label, needle in config_text, needle)

train_source = (ROOT / "train.py").read_text(encoding="utf-8") if (ROOT / "train.py").exists() else ""
for needle, label in [
    ("build_phase_loss_policy", "phase polish scheduler"),
    ("update_quality_controller", "quality brake controller"),
    ("best_quality.pth", "quality checkpoint"),
    ("best_psnr.pth", "PSNR checkpoint"),
    ("[V1.8.7 PHASE]", "v1.8.7 phase logging"),
    ("attach_dataset_contract", "dataset checkpoint contract"),
]:
    check(label, needle in train_source)
check("no hidden 100-to-80 epoch rewrite", 'tcfg["epochs"] = 80' not in train_source)

pipeline_source = (ROOT / "pipeline.py").read_text(encoding="utf-8") if (ROOT / "pipeline.py").exists() else ""
for needle, label in [
    ('input("Epochs [100]: ")', "interactive epoch default is 100"),
    ("GameSR Dataset Manager v1.8.7", "dataset menu"),
    ("Incremental update", "incremental option"),
    ("Full rebuild", "full rebuild option"),
    ("_quick_source_inventory", "new-file inventory"),
    ("phase3x3r-v4", "pipeline v4 experiment isolation"),
]:
    check(label, needle in pipeline_source)

prepare_source = (ROOT / "prepare_dataset.py").read_text(encoding="utf-8") if (ROOT / "prepare_dataset.py").exists() else ""
for needle, label in [
    ("balanced_group_assignment", "image-count-aware session split"),
    ("incremental_update", "incremental dataset update"),
    ("dataset_info.json", "dataset metadata"),
    ("manifest_fingerprint", "manifest fingerprint"),
    ("archive_existing", "manifest history backup"),
]:
    check(label, needle in prepare_source)

utils_source = (ROOT / "gamesr" / "utils.py").read_text(encoding="utf-8") if (ROOT / "gamesr" / "utils.py").exists() else ""
check("checkpoint includes dataset_contract", '"dataset_contract": config.get("dataset_contract")' in utils_source)

data_source = (ROOT / "gamesr" / "data.py").read_text(encoding="utf-8") if (ROOT / "gamesr" / "data.py").exists() else ""
for needle, label in [
    ('curriculum_mode", "legacy")).lower() == "stable_v2"', "stable degradation mode"),
    ("probability_scale = strength", "true degradation probability ramp"),
    ("clean_anchor_end", "per-batch clean anchor"),
]:
    check(label, needle in data_source)

# End-to-end dataset-manager smoke test with intentionally uneven sessions.
try:
    with tempfile.TemporaryDirectory(prefix="gamesr-v187-verify-") as td:
        temp = Path(td)
        source = temp / "clean" / "GameA"
        dataset = temp / "dataset"
        session_counts = [15, 7, 5, 4, 4, 2, 2, 1, 1]
        for i, count in enumerate(session_counts):
            folder = source / f"session{i:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            for j in range(count):
                Image.new("RGB", (32, 32), (i * 17 % 255, j * 11 % 255, 100)).save(folder / f"{j:04d}.png")

        common = [
            sys.executable, str(ROOT / "prepare_dataset.py"),
            "--source", str(temp / "clean"), "--output", str(dataset),
            "--min-width", "16", "--min-height", "16",
            "--min-eval-images", "3", "--min-eval-fraction", "0.05",
            "--min-eval-groups", "2", "--seed", "42",
        ]
        subprocess.run(common + ["--mode", "full"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        info1 = json.loads((dataset / "dataset_info.json").read_text(encoding="utf-8"))
        check("full rebuild health", bool(info1["health"]["healthy"]))
        check("full rebuild has >=2 val groups", int(info1["groups"]["val"]) >= 2)
        check("full rebuild has >=2 test groups", int(info1["groups"]["test"]) >= 2)
        check("full rebuild is image-count balanced", abs(int(info1["counts"]["train"]) - round(sum(session_counts) * 0.8)) <= 10)
        fp1 = info1["manifest_fingerprint"]

        new_session = source / "session09"
        new_session.mkdir(parents=True, exist_ok=True)
        for j in range(3):
            Image.new("RGB", (32, 32), (9, j, 9)).save(new_session / f"{j:04d}.png")
        subprocess.run(common + ["--mode", "incremental"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        info2 = json.loads((dataset / "dataset_info.json").read_text(encoding="utf-8"))
        check("incremental version increments", int(info2["dataset_version"]) == int(info1["dataset_version"]) + 1)
        check("incremental fingerprint changes", info2["manifest_fingerprint"] != fp1)
        check("incremental keeps healthy eval", bool(info2["health"]["healthy"]))

        train_paths: set[str] = set()
        with (dataset / "train.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            train_paths = {(row.get("path") or "").replace("\\", "/").casefold() for row in csv.DictReader(handle)}
        check("healthy incremental sends new session to train", all(str(p.resolve()).replace("\\", "/").casefold() in train_paths for p in new_session.glob("*.png")))
        check("previous manifests archived", (dataset / "history" / "v0001" / "train.csv").exists())
except Exception as exc:
    check("dataset manager smoke test", False, repr(exc))

if all(checks):
    print("\nGameSR v1.8.7 integrated patch verification PASSED.")
    raise SystemExit(0)
print("\nGameSR v1.8.7 verification FAILED. Re-extract the patch over v1.8.5 and retry.")
raise SystemExit(1)
