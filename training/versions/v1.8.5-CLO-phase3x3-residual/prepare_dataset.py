from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def valid_image(path: Path, minimum_width: int, minimum_height: int) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.width >= minimum_width and image.height >= minimum_height
    except Exception:
        return False


def group_key(path: Path, source: Path) -> str:
    relative = path.relative_to(source)
    # Collector V3.2 stores clean images as game/session/image.png.  Splitting
    # only by the first folder would group an entire game together; splitting
    # individual files would leak adjacent frames across train/val/test.
    if len(relative.parts) >= 3:
        return "/".join(relative.parts[:2])
    if len(relative.parts) == 2:
        return relative.parts[0]
    stem = path.stem
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep, 1)[0]
    return stem


def write_manifest(path: Path, files: list[Path]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path"])
        writer.writeheader()
        for item in files:
            writer.writerow({"path": str(item.resolve())})


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GameSR train/val/test manifests")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", default="dataset", type=Path)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--min-height", type=int, default=720)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source folder does not exist: {args.source}")
    candidates = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    valid, rejected = [], []
    for path in tqdm(candidates, desc="Scanning images", unit="img"):
        if valid_image(path, args.min_width, args.min_height):
            valid.append(path)
        else:
            rejected.append(path)
    if not valid:
        raise SystemExit("No valid images found.")

    groups: dict[str, list[Path]] = {}
    for path in valid:
        groups.setdefault(group_key(path, args.source), []).append(path)

    sessions = sorted(groups)
    random.Random(args.seed).shuffle(sessions)
    if len(sessions) < 3:
        # With no useful session structure, keep a deterministic file split but warn loudly.
        print("[WARN] Fewer than 3 capture groups detected. Collect more distinct sessions for a leakage-safe test set.")
        ordered = sorted(valid, key=lambda p: str(p).lower())
        n_test = max(1, round(len(ordered) * args.test))
        n_val = max(1, round(len(ordered) * args.val))
        splits = {
            "test": ordered[:n_test],
            "val": ordered[n_test:n_test + n_val],
            "train": ordered[n_test + n_val:],
        }
    else:
        n_test = max(1, round(len(sessions) * args.test))
        n_val = max(1, round(len(sessions) * args.val))
        selected = {
            "test": sessions[:n_test],
            "val": sessions[n_test:n_test + n_val],
            "train": sessions[n_test + n_val:],
        }
        splits = {name: [p for session in chosen for p in groups[session]] for name, chosen in selected.items()}

    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise SystemExit("One of train/val/test is empty. Provide more images/capture groups.")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, split_files in splits.items():
        write_manifest(args.output / f"{name}.csv", split_files)
        print(f"{name}: {len(split_files)} images")
    print(f"valid: {len(valid)} | rejected: {len(rejected)} | groups: {len(groups)}")


if __name__ == "__main__":
    main()
