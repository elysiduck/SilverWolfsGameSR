from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PIPELINE_VERSION = "3.2.1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_bytes(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


class Progress:
    def __init__(self, label: str, total: int, update_seconds: float = 0.5):
        self.label = label
        self.total = max(0, int(total))
        self.current = 0
        self.counts = {"KEEP": 0, "REVIEW": 0, "REJECT": 0, "ERROR": 0, "SKIP": 0}
        self.started = time.monotonic()
        self.last_update = self.started
        self.last_width = 0
        self.update_seconds = max(0.05, float(update_seconds))
        self.interactive = sys.stdout.isatty()

    def advance(self, status: str = "") -> None:
        self.current += 1
        if status in self.counts:
            self.counts[status] += 1
        now = time.monotonic()
        if self.current >= self.total or now - self.last_update >= self.update_seconds:
            self.render(now, force=self.current >= self.total)

    def render(self, now: float | None = None, force: bool = False) -> None:
        now = now or time.monotonic()
        # A small floor avoids meaningless million-images/second readings when
        # an incremental run only skips a handful of already indexed files.
        elapsed = max(0.05, now - self.started)
        fraction = self.current / self.total if self.total else 1.0
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 else float("inf")
        width = 24
        filled = min(width, int(fraction * width))
        bar = "#" * filled + "-" * (width - filled)
        c = self.counts
        line = (f"[{bar}] {fraction * 100:6.2f}%  {self.current:,}/{self.total:,} | "
                f"KEEP {c['KEEP']:,}  REVIEW {c['REVIEW']:,}  REJECT {c['REJECT']:,}  "
                f"ERROR {c['ERROR']:,}  SKIP {c['SKIP']:,} | {rate:5.1f} img/s | ETA {format_duration(eta)}")
        if self.interactive:
            padding = " " * max(0, self.last_width - len(line))
            print("\r" + line + padding, end="\n" if force else "", flush=True)
            self.last_width = len(line)
        else:
            print(f"{self.label}: {line}", flush=True)
        self.last_update = now


def stable_fraction(text: str) -> float:
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value / float(2**64 - 1)


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def dhash64(im: Image.Image) -> int:
    a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = a[:, 1:] > a[:, :-1]
    out = 0
    for bit in bits.ravel():
        out = (out << 1) | int(bit)
    return out


def phash64(im: Image.Image) -> int:
    a = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    n = 32
    x = np.arange(n, dtype=np.float32)
    k = np.arange(8, dtype=np.float32)[:, None]
    basis = np.cos((math.pi / n) * (x + 0.5) * k)
    low = (basis @ a @ basis.T)[:8, :8].ravel()
    median = float(np.median(low[1:]))
    out = 0
    for bit in low > median:
        out = (out << 1) | int(bit)
    return out


def hash_chunks(value: int) -> list[tuple[int, int]]:
    return [(slot, (int(value) >> (slot * 8)) & 0xFF) for slot in range(8)]


def entropy8(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def image_metrics(im: Image.Image) -> dict:
    rgb = np.asarray(im.convert("RGB").resize((256, 144), Image.Resampling.BILINEAR), dtype=np.uint8)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    g = gray.astype(np.float32)
    dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    sharpness = float((dx.mean() + dy.mean()) * 0.5)
    edge_density = float(((dx[:-1, :] + dy[:, :-1]) > 24.0).mean())
    mean, std, entropy = float(g.mean()), float(g.std()), entropy8(gray)
    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    border = max(2, min(gray.shape) // 30)
    bp = np.concatenate((gray[:border].ravel(), gray[-border:].ravel(), gray[:, :border].ravel(), gray[:, -border:].ravel()))
    border_dark = float((bp < 8).mean())
    channel_means = rgb.reshape(-1, 3).mean(axis=0)
    color_cast = float((channel_means.max() - channel_means.min()) / max(20.0, channel_means.mean()))
    quality = 100.0 * (
        0.15 * max(0.0, 1.0 - abs(mean - 118.0) / 150.0)
        + 0.15 * min(1.0, std / 55.0)
        + 0.25 * min(1.0, entropy / 7.5)
        + 0.25 * min(1.0, sharpness / 15.0)
        + 0.10 * max(0.0, 1.0 - clipped * 3.0)
        + 0.10 * max(0.0, 1.0 - border_dark * 1.4)
    )
    flags: list[str] = []
    if sharpness < 2.5: flags.append("soft_frame")
    if clipped > 0.30: flags.append("heavy_clipping")
    if border_dark > 0.72: flags.append("heavy_black_border")
    if color_cast > 0.52: flags.append("extreme_color_cast")
    if entropy < 3.2: flags.append("low_entropy")
    gh = np.histogram(gray, bins=16, range=(0, 256))[0].astype(np.float32)
    gh /= max(1.0, float(gh.sum()))
    color = np.concatenate((rgb.reshape(-1, 3).mean(axis=0), rgb.reshape(-1, 3).std(axis=0))).astype(np.float32) / 255.0
    gx, gy = np.diff(g, axis=1)[:-1, :], np.diff(g, axis=0)[:, :-1]
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + math.pi) % math.pi
    orient = np.histogram(angle, bins=8, range=(0, math.pi), weights=mag)[0].astype(np.float32)
    orient /= max(1.0, float(orient.sum()))
    embedding = np.concatenate((gh, color, orient)).astype(np.float32)
    embedding /= max(1e-8, float(np.linalg.norm(embedding)))
    return {"mean": mean, "std": std, "entropy": entropy, "sharpness": sharpness,
            "edge_density": edge_density, "clipped_fraction": clipped,
            "border_dark_fraction": border_dark, "color_cast": color_cast,
            "quality_score": float(np.clip(quality, 0, 100)), "flags": flags,
            "embedding": embedding}


class Cleaner:
    def __init__(self, root: Path, config_path: Path | None = None):
        self.root = root.resolve()
        self.cfg = json.loads((config_path or self.root / "cleaner_config.json").read_text(encoding="utf-8"))
        self.raw = self.root / self.cfg["raw_root"]
        self.clean = self.root / self.cfg["clean_root"]
        self.review = self.root / self.cfg.get("review_root", "GameSR_Dataset/review")
        self.report = self.root / self.cfg["report_root"]
        for p in (self.raw, self.clean, self.review, self.report): p.mkdir(parents=True, exist_ok=True)
        self.db_path = self.report / "cleaner_state.sqlite3"
        self.config_id = hashlib.sha256(json.dumps({"version": PIPELINE_VERSION, **self.cfg}, sort_keys=True).encode()).hexdigest()[:16]
        self.memory = np.empty((0, 30), dtype=np.float32)
        self.storage_stats = {
            "hardlinks_created": 0,
            "copies_created": 0,
            "hardlink_bytes_saved": 0,
            "copied_bytes": 0,
        }

    def connect(self, rebuild: bool = False) -> sqlite3.Connection:
        if rebuild and self.db_path.exists(): self.db_path.unlink()
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")
        con.executescript("""
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY, game TEXT, session TEXT, timestamp TEXT,
          size INTEGER, mtime_ns INTEGER, status TEXT, reason TEXT,
          mean REAL, std REAL, dhash TEXT, dhash_distance INTEGER,
          clean_path TEXT, processed_at TEXT, pipeline_version TEXT, config_id TEXT);
        CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
        CREATE INDEX IF NOT EXISTS idx_files_game ON files(game);
        CREATE TABLE IF NOT EXISTS fingerprints (
          path TEXT PRIMARY KEY, game TEXT NOT NULL, dhash64 TEXT NOT NULL,
          phash64 TEXT NOT NULL, embedding BLOB NOT NULL,
          quality_score REAL NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fingerprint_chunks (
          path TEXT NOT NULL, kind TEXT NOT NULL, slot INTEGER NOT NULL, value INTEGER NOT NULL,
          PRIMARY KEY(path, kind, slot));
        CREATE INDEX IF NOT EXISTS idx_fp_chunks ON fingerprint_chunks(kind,slot,value);
        """)
        additions = {"entropy": "REAL", "sharpness": "REAL", "edge_density": "REAL",
                     "clipped_fraction": "REAL", "border_dark_fraction": "REAL",
                     "color_cast": "REAL", "quality_score": "REAL", "novelty_score": "REAL",
                     "density": "INTEGER", "flags": "TEXT", "phash": "TEXT", "duplicate_of": "TEXT"}
        existing = {r[1] for r in con.execute("PRAGMA table_info(files)")}
        for name, typ in additions.items():
            if name not in existing: con.execute(f"ALTER TABLE files ADD COLUMN {name} {typ}")
        con.commit(); return con

    def rows(self) -> list[dict[str, str]]:
        rows, seen = [], set()
        meta = self.raw / "metadata.csv"
        if meta.exists():
            with meta.open("r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    rel = str(row.get("file", "")).replace("/", "\\")
                    if rel and rel not in seen:
                        row["file"] = rel; seen.add(rel); rows.append(row)
        for p in sorted(x for x in self.raw.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS):
            rel = str(p.relative_to(self.raw)).replace("/", "\\")
            if rel in seen: continue
            parts = p.relative_to(self.raw).parts
            rows.append({"file": rel, "game": parts[0] if parts else "UnknownGame",
                         "session": parts[1] if len(parts) > 1 else "UnknownSession", "timestamp": ""})
        return rows

    def info(self, row: dict[str, str]) -> tuple[Path, str, str, str, str]:
        rp = Path(str(row.get("file", "")).replace("\\", os.sep).replace("/", os.sep))
        rel = str(rp).replace(os.sep, "\\")
        game = str(row.get("game") or (rp.parts[0] if rp.parts else "UnknownGame"))
        session = str(row.get("session") or (rp.parts[1] if len(rp.parts) > 1 else "UnknownSession"))
        return self.raw / rp, rel, game, session, str(row.get("timestamp") or "")

    def processed(self, con: sqlite3.Connection, rel: str, src: Path) -> bool:
        row = con.execute("SELECT size,mtime_ns,status,clean_path FROM files WHERE path=?", (rel,)).fetchone()
        if not row or row[2] not in {"KEEP", "REVIEW", "REJECT", "ERROR"}: return False
        st = src.stat()
        if int(row[0]) != st.st_size or int(row[1]) != st.st_mtime_ns: return False
        if row[2] in {"KEEP", "REVIEW"} and row[3]:
            output_root = self.clean if row[2] == "KEEP" else self.review
            if not (output_root / str(row[3]).replace("\\", os.sep)).exists(): return False
        return True

    def invalidate_changed(self, con: sqlite3.Connection, rel: str) -> None:
        con.execute("DELETE FROM fingerprint_chunks WHERE path=?", (rel,))
        con.execute("DELETE FROM fingerprints WHERE path=?", (rel,))
        rp = Path(rel.replace("\\", os.sep))
        for output_root in (self.clean, self.review):
            path = output_root / rp
            if path.exists(): path.unlink()

    def candidates(self, con: sqlite3.Connection, dh: int, ph: int, game: str) -> list[tuple[str, int, int]]:
        clauses, params = [], []
        for kind, value in (("d", dh), ("p", ph)):
            for slot, chunk in hash_chunks(value):
                clauses.append("(c.kind=? AND c.slot=? AND c.value=?)"); params.extend((kind, slot, chunk))
        scope = "" if self.cfg.get("global_dedup", True) else " AND f.game=?"
        if scope: params.append(game)
        query = "SELECT DISTINCT f.path,f.dhash64,f.phash64 FROM fingerprint_chunks c JOIN fingerprints f ON f.path=c.path WHERE (" + " OR ".join(clauses) + ")" + scope
        return [(str(r[0]), int(r[1], 16), int(r[2], 16)) for r in con.execute(query, params)]

    def add_fingerprint(self, con: sqlite3.Connection, rel: str, game: str, dh: int, ph: int, emb: np.ndarray, quality: float) -> None:
        con.execute("DELETE FROM fingerprint_chunks WHERE path=?", (rel,))
        con.execute("INSERT OR REPLACE INTO fingerprints(path,game,dhash64,phash64,embedding,quality_score,created_at) VALUES(?,?,?,?,?,?,?)",
                    (rel, game, f"{dh:016x}", f"{ph:016x}", emb.astype("<f4").tobytes(), quality, now_iso()))
        con.executemany("INSERT INTO fingerprint_chunks(path,kind,slot,value) VALUES(?,?,?,?)",
                        [(rel, kind, slot, value) for kind, h in (("d", dh), ("p", ph)) for slot, value in hash_chunks(h)])

    def load_memory(self, con: sqlite3.Connection) -> None:
        limit = int(self.cfg.get("novelty_memory_size", 8192))
        vectors = [np.frombuffer(r[0], dtype="<f4") for r in con.execute("SELECT embedding FROM fingerprints ORDER BY rowid DESC LIMIT ?", (limit,)) if r[0]]
        self.memory = np.stack(vectors) if vectors else np.empty((0, 30), dtype=np.float32)

    def novelty(self, emb: np.ndarray) -> tuple[float, int]:
        if self.memory.size == 0: return 1.0, 0
        sims = self.memory @ emb
        best = float(np.clip(sims.max(), 0, 1))
        density = int((sims >= float(self.cfg.get("dense_similarity", 0.985))).sum())
        return float(np.clip((1.0 - best) / 0.12, 0, 1)), density

    def link_copy(self, src: Path, dst: Path) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): dst.unlink()
        size = src.stat().st_size
        try:
            os.link(src, dst)
            self.storage_stats["hardlinks_created"] += 1
            self.storage_stats["hardlink_bytes_saved"] += size
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            self.storage_stats["copies_created"] += 1
            self.storage_stats["copied_bytes"] += size
            return "copy"

    def save(self, con: sqlite3.Connection, rel: str, row: dict, src: Path | None, status: str, reason: str, metrics: dict | None = None, **extra) -> None:
        st, m = (src.stat() if src and src.exists() else None), (metrics or {})
        data = {"path": rel, "game": row.get("game", "UnknownGame"), "session": row.get("session", "UnknownSession"),
                "timestamp": row.get("timestamp", ""), "size": st.st_size if st else 0, "mtime_ns": st.st_mtime_ns if st else 0,
                "status": status, "reason": reason, "mean": m.get("mean"), "std": m.get("std"),
                "dhash": f"{extra['dh']:016x}" if extra.get("dh") is not None else None,
                "dhash_distance": extra.get("dhash_distance"), "clean_path": extra.get("output_path"),
                "processed_at": now_iso(), "pipeline_version": PIPELINE_VERSION, "config_id": self.config_id,
                "entropy": m.get("entropy"), "sharpness": m.get("sharpness"), "edge_density": m.get("edge_density"),
                "clipped_fraction": m.get("clipped_fraction"), "border_dark_fraction": m.get("border_dark_fraction"),
                "color_cast": m.get("color_cast"), "quality_score": m.get("quality_score"),
                "novelty_score": extra.get("novelty"), "density": extra.get("density"),
                "flags": ";".join(m.get("flags", [])), "phash": f"{extra['ph']:016x}" if extra.get("ph") is not None else None,
                "duplicate_of": extra.get("duplicate_of")}
        names = list(data); updates = ",".join(f"{x}=excluded.{x}" for x in names if x != "path")
        con.execute(f"INSERT INTO files({','.join(names)}) VALUES({','.join(':'+x for x in names)}) ON CONFLICT(path) DO UPDATE SET {updates}", data)

    def migrate_legacy_keeps(self, con: sqlite3.Connection) -> int:
        legacy = con.execute("SELECT path,game FROM files WHERE status='KEEP' AND path NOT IN (SELECT path FROM fingerprints)").fetchall()
        count = 0
        progress = Progress("MIGRATE", len(legacy), self.cfg.get("progress_update_seconds", 0.5))
        for rel, game in legacy:
            src = self.raw / str(rel).replace("\\", os.sep)
            if not src.exists():
                progress.advance("SKIP")
                continue
            try:
                with Image.open(src) as im:
                    im.load(); metrics = image_metrics(im); dh, ph = dhash64(im), phash64(im)
                self.add_fingerprint(con, str(rel), str(game or "UnknownGame"), dh, ph, metrics["embedding"], metrics["quality_score"])
                con.execute("UPDATE files SET entropy=?,sharpness=?,edge_density=?,clipped_fraction=?,border_dark_fraction=?,color_cast=?,quality_score=?,flags=?,phash=?,dhash=?,pipeline_version=? WHERE path=?",
                            (metrics["entropy"], metrics["sharpness"], metrics["edge_density"], metrics["clipped_fraction"], metrics["border_dark_fraction"], metrics["color_cast"], metrics["quality_score"], ";".join(metrics["flags"]), f"{ph:016x}", f"{dh:016x}", PIPELINE_VERSION, rel))
                count += 1
                if count % 100 == 0: con.commit()
                progress.advance("KEEP")
            except Exception:
                progress.advance("ERROR")
        con.commit(); return count

    def write_reports(self, con: sqlite3.Connection, batch: list[dict], skipped: int) -> dict:
        columns = [d[1] for d in con.execute("PRAGMA table_info(files)")]
        all_rows = [dict(zip(columns, r)) for r in con.execute("SELECT * FROM files ORDER BY path")]
        def write_csv(path: Path, rows: list[dict]) -> None:
            if not rows: path.write_text("", encoding="utf-8-sig"); return
            fields = sorted({k for r in rows for k in r})
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        write_csv(self.report / "last_run.csv", batch); write_csv(self.report / "dataset_report.csv", all_rows)
        counts = {str(k): int(v) for k, v in con.execute("SELECT status,COUNT(*) FROM files GROUP BY status")}
        reasons = {str(k): int(v) for k, v in con.execute("SELECT reason,COUNT(*) FROM files GROUP BY reason")}
        q = con.execute("SELECT AVG(quality_score),MIN(quality_score),MAX(quality_score),AVG(novelty_score) FROM files WHERE quality_score IS NOT NULL").fetchone()
        summary = {"pipeline_version": PIPELINE_VERSION, "config_id": self.config_id,
                   "processed_this_run": len(batch), "skipped_unchanged": skipped,
                   "status_totals": counts, "reason_totals": reasons,
                   "average_quality": q[0], "min_quality": q[1], "max_quality": q[2], "average_novelty": q[3],
                   "storage_this_run": dict(self.storage_stats),
                   "raw_is_never_deleted": True, "generated_at": now_iso()}
        (self.report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def rebuild_outputs(self) -> None:
        for folder in (self.clean, self.review):
            if folder.exists(): shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    def run(self, rebuild: bool = False) -> dict:
        if rebuild: self.rebuild_outputs()
        con = self.connect(rebuild)
        print(f"GameSR Cleaner V{PIPELINE_VERSION} | incremental | config={self.config_id}")
        print("[1/3] Checking database migration...")
        migrated = 0 if rebuild else self.migrate_legacy_keeps(con)
        self.load_memory(con)
        if migrated: print(f"[MIGRATION] Added V3 metrics/fingerprints to {migrated} historical KEEP images.")
        batch, skipped = [], 0
        dth, pth = int(self.cfg.get("dhash_threshold", 5)), int(self.cfg.get("phash_threshold", 7))
        if dth > 7 or pth > 7: raise ValueError("Hash thresholds must be <= 7 for indexed no-miss guarantee")
        print("[2/3] Scanning RAW file list...")
        rows = self.rows()
        print(f"Found {len(rows):,} RAW images. Processing...")
        progress = Progress("CLEAN", len(rows), self.cfg.get("progress_update_seconds", 0.5))
        if not rows:
            progress.render(force=True)
        for row in rows:
            src, rel, game, session, timestamp = self.info(row)
            row.update({"game": game, "session": session, "timestamp": timestamp})
            if not src.exists():
                self.save(con, rel, row, None, "ERROR", "missing_file"); batch.append({**row, "status": "ERROR", "reason": "missing_file"}); progress.advance("ERROR"); continue
            if not rebuild and self.processed(con, rel, src):
                skipped += 1; progress.advance("SKIP"); continue
            self.invalidate_changed(con, rel)
            try:
                with Image.open(src) as im:
                    im.load(); size = im.size; metrics = image_metrics(im)
                    if size != (int(self.cfg["expected_width"]), int(self.cfg["expected_height"])):
                        self.save(con, rel, row, src, "REJECT", "wrong_resolution", metrics); batch.append({**row, "status": "REJECT", "reason": "wrong_resolution"}); progress.advance("REJECT"); continue
                    if (metrics["mean"] < float(self.cfg.get("min_mean", 6)) and metrics["std"] < 8) or metrics["std"] < float(self.cfg.get("min_std", 3)):
                        self.save(con, rel, row, src, "REJECT", "low_information", metrics); batch.append({**row, "status": "REJECT", "reason": "low_information"}); progress.advance("REJECT"); continue
                    dh, ph = dhash64(im), phash64(im)
            except Exception as exc:
                self.save(con, rel, row, src, "ERROR", "decode_error"); batch.append({**row, "status": "ERROR", "reason": "decode_error", "error": str(exc)}); progress.advance("ERROR"); continue
            duplicate = None; dd = pd = None
            for old, old_d, old_p in self.candidates(con, dh, ph, game):
                td, tp = hamming(dh, old_d), hamming(ph, old_p)
                if td <= dth and tp <= pth:
                    duplicate, dd, pd = old, td, tp; break
            novelty, density = self.novelty(metrics["embedding"])
            if duplicate:
                self.save(con, rel, row, src, "REJECT", "global_near_duplicate", metrics, dh=dh, ph=ph, dhash_distance=dd, duplicate_of=duplicate, novelty=novelty, density=density)
                batch.append({**row, "status": "REJECT", "reason": "global_near_duplicate", "duplicate_of": duplicate, "dhash_distance": dd, "phash_distance": pd}); progress.advance("REJECT"); continue
            quality = metrics["quality_score"]; status, reason = "KEEP", "keep"
            if quality < float(self.cfg.get("review_quality_below", 45)):
                status, reason = "REVIEW", "low_quality_review"
            elif any(x in metrics["flags"] for x in self.cfg.get("review_flags", ["heavy_clipping", "heavy_black_border", "extreme_color_cast"])):
                status, reason = "REVIEW", "anomaly_review"
            elif density >= int(self.cfg.get("dense_neighbor_limit", 8)) and novelty < float(self.cfg.get("low_novelty_threshold", 0.18)) and quality < float(self.cfg.get("quality_override", 86)):
                if stable_fraction(rel) > float(self.cfg.get("dense_keep_probability", 0.20)):
                    status, reason = "REVIEW", "low_novelty_dense_region"
            output_root = self.clean if status == "KEEP" else self.review
            rel_path = Path(rel.replace("\\", os.sep)); storage_method = self.link_copy(src, output_root / rel_path)
            self.save(con, rel, row, src, status, reason, metrics, dh=dh, ph=ph, output_path=str(rel_path), novelty=novelty, density=density)
            if status == "KEEP":
                self.add_fingerprint(con, rel, game, dh, ph, metrics["embedding"], quality)
                self.memory = np.concatenate((self.memory, metrics["embedding"][None, :]), axis=0)[-int(self.cfg.get("novelty_memory_size", 8192)):]
            batch.append({**row, "status": status, "reason": reason, "quality_score": round(quality, 3),
                          "novelty_score": round(novelty, 4), "density": density, "flags": ";".join(metrics["flags"]),
                          "storage_method": storage_method})
            progress.advance(status)
            if len(batch) % 100 == 0: con.commit()
        con.commit()
        print("[3/3] Writing CSV/JSON reports...")
        summary = self.write_reports(con, batch, skipped); con.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        stats = self.storage_stats
        print(
            f"[STORAGE] Hard links: {stats['hardlinks_created']:,} "
            f"(saved about {format_bytes(stats['hardlink_bytes_saved'])}); "
            f"fallback copies: {stats['copies_created']:,} "
            f"({format_bytes(stats['copied_bytes'])} extra disk use)."
        )
        if stats["copies_created"]:
            print("[STORAGE WARN] Some files were copied because hard-link creation failed. Keep RAW/CLEAN/REVIEW on the same NTFS drive for best space savings.")
        print("RAW was not modified. KEEP and REVIEW are separate.")
        return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental GameSR dataset cleaner V3")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--config", type=Path); ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(); Cleaner(args.root, args.config).run(args.rebuild)


if __name__ == "__main__": main()
