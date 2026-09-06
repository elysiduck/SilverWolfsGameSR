from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from cleaner import Cleaner, image_metrics
from dataset_audit import audit


SOURCE = Path(__file__).resolve().parent


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), "RGB").save(path)


def scene(seed: int, border: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:1080, :1920]
    a = np.empty((1080, 1920, 3), dtype=np.uint8)
    a[..., 0] = (x / 8 + seed * 17) % 256
    a[..., 1] = (y / 5 + seed * 31) % 256
    a[..., 2] = ((x + y) / 11 + seed * 47) % 256
    a = np.clip(a.astype(np.int16) + rng.integers(-8, 9, a.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    if border:
        a[:80] = 0; a[-80:] = 0; a[:, :80] = 0; a[:, -80:] = 0
    return a


def status_map(db: Path) -> dict[str, tuple[str, str]]:
    con = sqlite3.connect(db)
    result = {p: (s, r) for p, s, r in con.execute("SELECT path,status,reason FROM files")}
    con.close(); return result


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="gamesr_v3_test_") as td:
        root = Path(td)
        (root / "cleaner_config.json").write_text((SOURCE / "cleaner_config.json").read_text(encoding="utf-8"), encoding="utf-8")
        raw = root / "GameSR_Dataset/raw"
        normal = scene(1)
        save_rgb(raw / "GameA/S1/normal.png", normal)
        save_rgb(raw / "GameB/S9/cross_game_duplicate.png", normal)
        save_rgb(raw / "GameA/S1/unique.png", scene(7))
        save_rgb(raw / "GameA/S1/border.png", scene(11, border=True))
        save_rgb(raw / "GameA/S1/black.png", np.zeros((1080, 1920, 3), dtype=np.uint8))
        save_rgb(raw / "GameA/S1/wrong.png", np.full((720, 1280, 3), 120, dtype=np.uint8))

        first = Cleaner(root).run()
        states = status_map(root / "GameSR_Dataset/reports/cleaner_state.sqlite3")
        assert first["processed_this_run"] == 6, first
        assert states["GameB\\S9\\cross_game_duplicate.png"][1] == "global_near_duplicate", states
        assert states["GameA\\S1\\black.png"][1] == "low_information", states
        assert states["GameA\\S1\\wrong.png"][1] == "wrong_resolution", states
        assert states["GameA\\S1\\border.png"][0] == "REVIEW", states
        assert first["storage_this_run"]["hardlinks_created"] == 3, first
        assert first["storage_this_run"]["copies_created"] == 0, first

        second = Cleaner(root).run()
        assert second["processed_this_run"] == 0 and second["skipped_unchanged"] == 6, second

        save_rgb(raw / "GameC/S2/new.png", scene(23))
        third = Cleaner(root).run()
        assert third["processed_this_run"] == 1 and third["skipped_unchanged"] == 6, third

        # A changed RAW at the same path must invalidate its old fingerprint,
        # rather than being falsely matched as a duplicate of itself.
        save_rgb(raw / "GameA/S1/unique.png", scene(19))
        fourth = Cleaner(root).run()
        assert fourth["processed_this_run"] == 1 and fourth["skipped_unchanged"] == 6, fourth
        assert status_map(root / "GameSR_Dataset/reports/cleaner_state.sqlite3")["GameA\\S1\\unique.png"][0] in {"KEEP", "REVIEW"}

        result = audit(root)
        report = root / "GameSR_Dataset/reports"
        assert result["processed_total"] == 7
        assert (report / "dataset_audit.json").exists()
        assert (report / "dataset_audit.html").exists()
        assert (report / "audit_contact_sheet.jpg").exists()
        assert result["storage"]["hardlinks"] == 4, result["storage"]
        assert result["storage"]["copies"] == 0, result["storage"]

        # Force hard-link failure and verify the safe copy fallback is counted.
        fallback = Cleaner(root)
        fallback_src = raw / "fallback-source.bin"
        fallback_dst = root / "fallback-output.bin"
        fallback_src.write_bytes(b"storage-fallback-test")
        with patch("cleaner.os.link", side_effect=OSError("simulated cross-device link")):
            method = fallback.link_copy(fallback_src, fallback_dst)
        assert method == "copy" and fallback_dst.read_bytes() == fallback_src.read_bytes()
        assert fallback.storage_stats["copies_created"] == 1

        # Directly verify the density/novelty mechanism independently of hash rejection.
        cleaner = Cleaner(root)
        emb = image_metrics(Image.fromarray(scene(31), "RGB"))["embedding"]
        cleaner.memory = np.stack([emb] * 12)
        novelty, density = cleaner.novelty(emb)
        assert novelty < 1e-5 and density == 12

        # Simulate an existing V2.4 database and verify in-place KEEP backfill.
        legacy = root / "legacy"
        legacy.mkdir(parents=True)
        (legacy / "cleaner_config.json").write_text((SOURCE / "cleaner_config.json").read_text(encoding="utf-8"), encoding="utf-8")
        legacy_raw = legacy / "GameSR_Dataset/raw/GameOld/Session1/old.png"
        save_rgb(legacy_raw, scene(44))
        legacy_clean = legacy / "GameSR_Dataset/clean/GameOld/Session1/old.png"
        legacy_clean.parent.mkdir(parents=True, exist_ok=True)
        legacy_clean.write_bytes(legacy_raw.read_bytes())
        legacy_db = legacy / "GameSR_Dataset/reports/cleaner_state.sqlite3"
        legacy_db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(legacy_db)
        con.execute("""CREATE TABLE files(path TEXT PRIMARY KEY,game TEXT,session TEXT,timestamp TEXT,size INTEGER,mtime_ns INTEGER,status TEXT,reason TEXT,mean REAL,std REAL,dhash TEXT,dhash_distance INTEGER,clean_path TEXT,processed_at TEXT,pipeline_version TEXT,config_id TEXT)""")
        st = legacy_raw.stat()
        con.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("GameOld\\Session1\\old.png", "GameOld", "Session1", "", st.st_size, st.st_mtime_ns, "KEEP", "keep", 100.0, 40.0, "0", None, "GameOld\\Session1\\old.png", "", "2.4", "legacy"))
        con.commit(); con.close()
        migrated = Cleaner(legacy).run()
        con = sqlite3.connect(legacy_db)
        fp_count = con.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        quality = con.execute("SELECT quality_score FROM files").fetchone()[0]
        con.close()
        assert migrated["processed_this_run"] == 0 and migrated["skipped_unchanged"] == 1
        assert fp_count == 1 and quality is not None

    print("ALL TESTS PASSED: incremental/changed-file, global dedup, quality/review, novelty density, audit, V2.4 migration")


if __name__ == "__main__":
    run()
