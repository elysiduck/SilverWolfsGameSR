from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
SCHEMA_VERSION = 2


def valid_image(path: Path, minimum_width: int, minimum_height: int) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.width >= minimum_width and image.height >= minimum_height
    except Exception:
        return False


def norm_path(path: Path) -> str:
    # Case-folding keeps Windows/SMB path comparisons stable while remaining
    # harmless on case-sensitive systems for the manifest bookkeeping purpose.
    return str(path.resolve()).replace("\\", "/").casefold()


def group_key(path: Path, source: Path) -> str:
    try:
        relative = path.resolve().relative_to(source.resolve())
    except Exception:
        relative = path
    # Collector V3.2 stores clean images as game/session/image.png. Splitting
    # only by game leaks adjacent frames; splitting individual files destroys
    # session isolation. Keep game/session together whenever possible.
    if len(relative.parts) >= 3:
        return "/".join(relative.parts[:2])
    if len(relative.parts) == 2:
        return relative.parts[0]
    stem = path.stem
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep, 1)[0]
    return stem


def scan_candidates(source: Path) -> list[Path]:
    return sorted(
        (p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file()),
        key=lambda p: str(p).casefold(),
    )


def read_manifest(path: Path) -> list[Path]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    items: list[Path] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("path") or "").strip()
            if raw:
                items.append(Path(raw))
    return items


def read_splits(output: Path) -> dict[str, list[Path]]:
    return {split: read_manifest(output / f"{split}.csv") for split in SPLITS}


def write_manifest(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path"])
        writer.writeheader()
        for item in sorted(files, key=lambda p: str(p).casefold()):
            writer.writerow({"path": str(item.resolve())})


def group_files(files: list[Path], source: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[group_key(path, source)].append(path)
    return dict(groups)


def split_group_counts(splits: dict[str, list[Path]], source: Path) -> dict[str, int]:
    return {name: len(group_files(files, source)) for name, files in splits.items()}


def manifest_fingerprint(splits: dict[str, list[Path]]) -> str:
    digest = hashlib.sha256()
    for split in SPLITS:
        digest.update((split + "\n").encode("utf-8"))
        for path in sorted((norm_path(p) for p in splits.get(split, []))):
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def minimum_eval_images(total: int, absolute_min: int, fraction_min: float) -> int:
    if total <= 0:
        return 1
    return max(1, min(total, max(int(absolute_min), math.ceil(total * max(0.0, fraction_min)))))


def evaluate_health(
    splits: dict[str, list[Path]],
    source: Path,
    *,
    min_eval_images: int,
    min_eval_groups: int,
) -> dict:
    counts = {name: len(splits.get(name, [])) for name in SPLITS}
    groups = split_group_counts(splits, source)
    group_owner: dict[str, str] = {}
    leaked: set[str] = set()
    for split, files in splits.items():
        for key in group_files(files, source):
            previous = group_owner.setdefault(key, split)
            if previous != split:
                leaked.add(key)
    healthy = (
        counts["train"] > 0
        and counts["val"] >= min_eval_images
        and counts["test"] >= min_eval_images
        and groups["val"] >= min_eval_groups
        and groups["test"] >= min_eval_groups
        and not leaked
    )
    return {
        "healthy": healthy,
        "counts": counts,
        "groups": groups,
        "min_eval_images": min_eval_images,
        "min_eval_groups": min_eval_groups,
        "leaked_groups": sorted(leaked),
    }


def split_score(
    assignment: dict[str, str],
    group_sizes: dict[str, int],
    total: int,
    val_ratio: float,
    test_ratio: float,
    min_eval_images: int,
    min_eval_groups: int,
) -> float:
    targets = {
        "train": total * max(0.0, 1.0 - val_ratio - test_ratio),
        "val": total * val_ratio,
        "test": total * test_ratio,
    }
    counts = {name: 0 for name in SPLITS}
    group_counts = {name: 0 for name in SPLITS}
    for group, split in assignment.items():
        counts[split] += group_sizes[group]
        group_counts[split] += 1
    score = 0.0
    for split in SPLITS:
        target = max(1.0, targets[split])
        score += ((counts[split] - target) / target) ** 2
    # Hardly ever accept tiny validation/test sets when a better assignment is
    # possible. The large coefficient intentionally dominates ratio niceties.
    for split in ("val", "test"):
        if counts[split] < min_eval_images:
            score += 1000.0 * ((min_eval_images - counts[split]) / max(1, min_eval_images)) ** 2
        if group_counts[split] < min_eval_groups:
            score += 500.0 * (min_eval_groups - group_counts[split]) ** 2
    if counts["train"] <= 0:
        score += 10000.0
    return score


def balanced_group_assignment(
    groups: dict[str, list[Path]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_eval_images: int,
    min_eval_groups: int,
) -> dict[str, str]:
    names = sorted(groups)
    sizes = {name: len(groups[name]) for name in names}
    total = sum(sizes.values())
    if len(names) < 3:
        raise ValueError("Fewer than 3 groups")

    rng = random.Random(seed)
    best: dict[str, str] | None = None
    best_score = float("inf")
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)

    # Randomized search is robust to highly uneven session sizes (1 image vs
    # 1500+ images). Deterministic seed keeps the chosen split reproducible.
    trials = max(5000, min(30000, 1200 * len(names)))
    for trial in range(trials):
        assignment: dict[str, str] = {}
        for name in names:
            r = rng.random()
            if r < train_ratio:
                split = "train"
            elif r < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"
            assignment[name] = split
        # Inject a few deterministic-ish alternatives so tiny group counts do
        # not depend on lucky random draws.
        if trial < len(names):
            order = names[trial:] + names[:trial]
            assignment = {}
            counts = {name: 0 for name in SPLITS}
            targets = {
                "train": total * train_ratio,
                "val": total * val_ratio,
                "test": total * test_ratio,
            }
            for group in order:
                # Fill the most under-target split, normalized by its target.
                split = min(SPLITS, key=lambda s: counts[s] / max(1.0, targets[s]))
                assignment[group] = split
                counts[split] += sizes[group]

        score = split_score(
            assignment, sizes, total, val_ratio, test_ratio,
            min_eval_images, min_eval_groups,
        )
        if score < best_score:
            best, best_score = dict(assignment), score

    assert best is not None
    return best


def deterministic_file_split(
    files: list[Path], val_ratio: float, test_ratio: float
) -> dict[str, list[Path]]:
    ordered = sorted(files, key=lambda p: str(p).casefold())
    n_test = max(1, round(len(ordered) * test_ratio))
    n_val = max(1, round(len(ordered) * val_ratio))
    if n_test + n_val >= len(ordered):
        n_test = 1
        n_val = 1
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test:n_test + n_val],
        "train": ordered[n_test + n_val:],
    }


def archive_existing(output: Path, info: dict | None) -> None:
    manifests = [output / f"{split}.csv" for split in SPLITS]
    if not any(path.exists() for path in manifests) and not (output / "dataset_info.json").exists():
        return
    history = output / "history"
    history.mkdir(parents=True, exist_ok=True)
    if info and isinstance(info.get("dataset_version"), int):
        name = f"v{int(info['dataset_version']):04d}"
    else:
        name = "legacy-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    target = history / name
    suffix = 1
    while target.exists():
        target = history / f"{name}-{suffix}"
        suffix += 1
    target.mkdir(parents=True, exist_ok=True)
    for path in manifests + [output / "dataset_info.json"]:
        if path.exists():
            shutil.copy2(path, target / path.name)


def load_info(output: Path) -> dict | None:
    path = output / "dataset_info.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_info(
    output: Path,
    *,
    source: Path,
    action: str,
    splits: dict[str, list[Path]],
    valid_count: int,
    rejected_count: int,
    total_groups: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    min_eval_images: int,
    min_eval_groups: int,
    previous_info: dict | None,
    extra: dict | None = None,
) -> dict:
    previous_version = int(previous_info.get("dataset_version", 0)) if previous_info else 0
    version = previous_version + 1
    health = evaluate_health(
        splits, source,
        min_eval_images=min_eval_images,
        min_eval_groups=min_eval_groups,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": version,
        "created_at": previous_info.get("created_at") if previous_info else datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "action": action,
        "source": str(source.resolve()),
        "seed": seed,
        "ratios": {
            "train": round(max(0.0, 1.0 - val_ratio - test_ratio), 6),
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "source_valid": valid_count,
            "source_rejected": rejected_count,
            **health["counts"],
        },
        "groups": {
            "source_total": total_groups,
            **health["groups"],
        },
        "health": {
            "healthy": bool(health["healthy"]),
            "min_eval_images": min_eval_images,
            "min_eval_groups": min_eval_groups,
            "leaked_groups": health["leaked_groups"],
        },
        "manifest_fingerprint": manifest_fingerprint(splits),
    }
    if extra:
        payload["update"] = extra
    (output / "dataset_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def print_summary(info: dict) -> None:
    counts = info.get("counts", {})
    groups = info.get("groups", {})
    health = info.get("health", {})
    print("\n[DATASET]")
    print(f"  Version : v{int(info.get('dataset_version', 0)):04d}")
    print(f"  Source  : {info.get('source', '-')}")
    print(f"  Train   : {counts.get('train', 0)} images / {groups.get('train', 0)} groups")
    print(f"  Val     : {counts.get('val', 0)} images / {groups.get('val', 0)} groups")
    print(f"  Test    : {counts.get('test', 0)} images / {groups.get('test', 0)} groups")
    print(f"  Source  : {counts.get('source_valid', 0)} valid / {counts.get('source_rejected', 0)} rejected")
    print(f"  Health  : {'OK' if health.get('healthy') else 'INVALID / NEEDS ATTENTION'}")
    print(f"  Contract: {str(info.get('manifest_fingerprint', ''))[:16]}...")


def full_rebuild(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    candidates = scan_candidates(source)
    valid: list[Path] = []
    rejected: list[Path] = []
    for path in tqdm(candidates, desc="Scanning images", unit="img"):
        (valid if valid_image(path, args.min_width, args.min_height) else rejected).append(path)
    if not valid:
        raise SystemExit("No valid images found.")

    groups = group_files(valid, source)
    min_eval = minimum_eval_images(len(valid), args.min_eval_images, args.min_eval_fraction)
    min_groups = min(args.min_eval_groups, max(1, (len(groups) - 1) // 2))

    if len(groups) < 3:
        print("[WARN] Fewer than 3 capture groups detected; falling back to deterministic file split.")
        splits = deterministic_file_split(valid, args.val, args.test)
        min_groups = 1
    else:
        assignment = balanced_group_assignment(
            groups,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
            min_eval_images=min_eval,
            min_eval_groups=min_groups,
        )
        splits = {name: [] for name in SPLITS}
        for group, files in groups.items():
            splits[assignment[group]].extend(files)

    previous = load_info(args.output)
    archive_existing(args.output, previous)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_manifest(args.output / f"{split}.csv", splits[split])

    info = write_info(
        args.output,
        source=source,
        action="full_rebuild",
        splits=splits,
        valid_count=len(valid),
        rejected_count=len(rejected),
        total_groups=len(groups),
        seed=args.seed,
        val_ratio=args.val,
        test_ratio=args.test,
        min_eval_images=min_eval,
        min_eval_groups=min_groups,
        previous_info=previous,
        extra={"candidate_files": len(candidates)},
    )
    print_summary(info)
    if not info["health"]["healthy"]:
        print("[WARN] Best available split still misses one or more health targets; collect more distinct sessions.")
    return info


def incremental_update(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    existing = read_splits(args.output)
    if not all(existing[name] for name in SPLITS):
        raise SystemExit("Incremental update requires non-empty train/val/test manifests. Use full rebuild first.")

    # Detect session leakage before adding anything. Repairing an already leaked
    # manifest safely requires a full rebuild.
    all_existing = [p for files in existing.values() for p in files]
    existing_group_owner: dict[str, str] = {}
    for split, files in existing.items():
        for key in group_files(files, source):
            old = existing_group_owner.setdefault(key, split)
            if old != split:
                raise SystemExit(
                    f"Existing manifests leak session '{key}' across {old}/{split}. Use full rebuild."
                )

    candidates = scan_candidates(source)
    candidate_by_norm = {norm_path(p): p for p in candidates}
    existing_norm_to_split = {
        norm_path(p): split for split, files in existing.items() for p in files
    }

    # Remove manifest entries that no longer exist under the selected source.
    cleaned = {name: [] for name in SPLITS}
    removed = 0
    for split, files in existing.items():
        for p in files:
            n = norm_path(p)
            if n in candidate_by_norm:
                cleaned[split].append(candidate_by_norm[n])
            else:
                removed += 1

    new_candidates = [p for n, p in candidate_by_norm.items() if n not in existing_norm_to_split]
    valid_new: list[Path] = []
    rejected_new: list[Path] = []
    for path in tqdm(new_candidates, desc="Validating new images", unit="img"):
        (valid_new if valid_image(path, args.min_width, args.min_height) else rejected_new).append(path)

    # Preserve session ownership. New frames belonging to an existing Val/Test
    # session remain in that split; they are never leaked into Train.
    group_owner: dict[str, str] = {}
    for split, files in cleaned.items():
        for key in group_files(files, source):
            group_owner[key] = split

    new_groups = group_files(valid_new, source)
    unassigned: dict[str, list[Path]] = {}
    for key, files in new_groups.items():
        owner = group_owner.get(key)
        if owner:
            cleaned[owner].extend(files)
        else:
            unassigned[key] = files

    unseen_group_count = len(unassigned)
    projected_total = sum(len(v) for v in cleaned.values()) + sum(len(v) for v in unassigned.values())
    min_eval = minimum_eval_images(projected_total, args.min_eval_images, args.min_eval_fraction)
    existing_group_counts = split_group_counts(cleaned, source)
    min_groups = min(args.min_eval_groups, max(1, (len(set(group_owner) | set(unassigned)) - 1) // 2))

    # Healthy evaluation sets are frozen by session: all brand-new sessions go
    # to Train. If Val/Test are invalid, use the smallest suitable new sessions
    # to repair them first, then send the rest to Train.
    for split in ("val", "test"):
        while unassigned and (
            len(cleaned[split]) < min_eval or existing_group_counts[split] < min_groups
        ):
            need = max(1, min_eval - len(cleaned[split]))
            # Prefer a session that closes the deficit with the least overshoot;
            # if none is large enough, take the largest remaining one.
            items = list(unassigned.items())
            large_enough = [(k, f) for k, f in items if len(f) >= need]
            if large_enough:
                key, files = min(large_enough, key=lambda item: (len(item[1]) - need, len(item[1]), item[0]))
            else:
                key, files = max(items, key=lambda item: (len(item[1]), item[0]))
            cleaned[split].extend(files)
            existing_group_counts[split] += 1
            del unassigned[key]

    for files in unassigned.values():
        cleaned["train"].extend(files)

    previous = load_info(args.output)
    archive_existing(args.output, previous)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_manifest(args.output / f"{split}.csv", cleaned[split])

    all_valid_now = sum(len(v) for v in cleaned.values())
    all_groups_now = len(group_files([p for files in cleaned.values() for p in files], source))
    info = write_info(
        args.output,
        source=source,
        action="incremental_update",
        splits=cleaned,
        valid_count=all_valid_now,
        rejected_count=len(rejected_new),
        total_groups=all_groups_now,
        seed=args.seed,
        val_ratio=args.val,
        test_ratio=args.test,
        min_eval_images=min_eval,
        min_eval_groups=min_groups,
        previous_info=previous,
        extra={
            "new_candidates": len(new_candidates),
            "new_valid": len(valid_new),
            "new_rejected": len(rejected_new),
            "removed_missing": removed,
            "new_unseen_sessions": unseen_group_count,
        },
    )
    print_summary(info)
    if not info["health"]["healthy"]:
        print("[WARN] Incremental update could not fully repair Val/Test; choose full rebuild when practical.")
    return info


def stats(args: argparse.Namespace) -> None:
    splits = read_splits(args.output)
    counts = {name: len(splits[name]) for name in SPLITS}
    info = load_info(args.output)
    source = args.source
    if source is None and info and info.get("source"):
        source = Path(info["source"])
    print("\n[DATASET STATS]")
    print(f"  Train manifest: {counts['train']}")
    print(f"  Val manifest  : {counts['val']}")
    print(f"  Test manifest : {counts['test']}")
    if info:
        print_summary(info)
    else:
        print("  Dataset metadata: legacy / not available")
    if source is not None and source.exists():
        candidates = scan_candidates(source)
        manifest_norm = {norm_path(p) for files in splits.values() for p in files}
        candidate_norm = {norm_path(p) for p in candidates}
        print(f"  Source candidate images: {len(candidates)}")
        print(f"  New/unmanifested files : {len(candidate_norm - manifest_norm)}")
        print(f"  Missing manifest files : {len(manifest_norm - candidate_norm)}")
    elif source is not None:
        print(f"  [WARN] Stored/provided source is not reachable: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GameSR v1.8.7 session-aware dataset manager")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", default="dataset", type=Path)
    parser.add_argument("--mode", choices=("full", "incremental", "stats"), default="full")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--min-height", type=int, default=720)
    parser.add_argument("--min-eval-images", type=int, default=300)
    parser.add_argument("--min-eval-fraction", type=float, default=0.05)
    parser.add_argument("--min-eval-groups", type=int, default=2)
    args = parser.parse_args()

    if args.val <= 0 or args.test <= 0 or args.val + args.test >= 1:
        raise SystemExit("--val and --test must be >0 and sum to <1.")
    args.output = args.output.resolve()
    if args.mode in {"full", "incremental"}:
        if args.source is None:
            raise SystemExit("--source is required for full/incremental mode.")
        if not args.source.exists():
            raise SystemExit(f"Source folder does not exist: {args.source}")

    if args.mode == "full":
        full_rebuild(args)
    elif args.mode == "incremental":
        incremental_update(args)
    else:
        stats(args)


if __name__ == "__main__":
    main()
