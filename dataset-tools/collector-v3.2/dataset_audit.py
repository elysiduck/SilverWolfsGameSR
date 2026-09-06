from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def query_dicts(con: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    cur = con.execute(sql, params)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def contact_sheet(root: Path, raw: Path, rows: list[dict], output: Path) -> int:
    cells = []
    for row in rows:
        path = raw / str(row["path"]).replace("\\", os.sep)
        try:
            with Image.open(path) as im:
                thumb = im.convert("RGB"); thumb.thumbnail((300, 169))
                canvas = Image.new("RGB", (320, 215), "#171a21")
                canvas.paste(thumb, ((320 - thumb.width) // 2, 6))
                draw = ImageDraw.Draw(canvas)
                label = f"{row.get('status')} Q={float(row.get('quality_score') or 0):.1f} N={float(row.get('novelty_score') or 0):.2f}"
                draw.text((8, 180), label, fill="white", font=ImageFont.load_default())
                draw.text((8, 196), str(row.get("game", ""))[:44], fill="#aeb8cc", font=ImageFont.load_default())
                cells.append(canvas)
        except Exception:
            continue
    if not cells: return 0
    cols, rows_n = 4, (len(cells) + 3) // 4
    sheet = Image.new("RGB", (cols * 320, rows_n * 215), "#0d0f14")
    for i, cell in enumerate(cells): sheet.paste(cell, ((i % cols) * 320, (i // cols) * 215))
    sheet.save(output, quality=90)
    return len(cells)


def storage_audit(con: sqlite3.Connection, raw: Path, clean: Path, review: Path) -> dict:
    result = {
        "hardlinks": 0,
        "copies": 0,
        "missing_outputs": 0,
        "hardlink_bytes_saved": 0,
        "copied_bytes": 0,
    }
    rows = con.execute(
        "SELECT path,status,clean_path,size FROM files WHERE status IN ('KEEP','REVIEW')"
    ).fetchall()
    for source_rel, status, output_rel, stored_size in rows:
        rel = str(source_rel).replace("\\", os.sep)
        out_rel = str(output_rel or source_rel).replace("\\", os.sep)
        src = raw / rel
        dst = (clean if status == "KEEP" else review) / out_rel
        if not src.is_file() or not dst.is_file():
            result["missing_outputs"] += 1
            continue
        size = int(stored_size or src.stat().st_size)
        try:
            same = os.path.samefile(src, dst)
        except OSError:
            same = False
        if same:
            result["hardlinks"] += 1
            result["hardlink_bytes_saved"] += size
        else:
            result["copies"] += 1
            result["copied_bytes"] += size
    return result


def audit(root: Path) -> dict:
    cfg = json.loads((root / "cleaner_config.json").read_text(encoding="utf-8"))
    report = root / cfg["report_root"]
    raw = root / cfg["raw_root"]
    clean = root / cfg["clean_root"]
    review = root / cfg.get("review_root", "GameSR_Dataset/review")
    db = report / "cleaner_state.sqlite3"
    if not db.exists(): raise SystemExit("Cleaner database not found. Run run_cleaner.bat first.")
    con = sqlite3.connect(db)
    totals = {k: int(v) for k, v in con.execute("SELECT status,COUNT(*) FROM files GROUP BY status")}
    total = sum(totals.values())
    games = query_dicts(con, "SELECT game,COUNT(*) count,SUM(status='KEEP') kept,AVG(quality_score) avg_quality,AVG(novelty_score) avg_novelty FROM files GROUP BY game ORDER BY count DESC")
    sessions = query_dicts(con, "SELECT game,session,COUNT(*) count,AVG(mean) avg_mean,AVG(std) avg_contrast,AVG(color_cast) avg_color_cast,AVG(quality_score) avg_quality FROM files WHERE quality_score IS NOT NULL GROUP BY game,session")
    reasons = {str(k): int(v) for k, v in con.execute("SELECT reason,COUNT(*) FROM files GROUP BY reason")}
    quality_bands = {str(k): int(v) for k, v in con.execute("SELECT CASE WHEN quality_score>=80 THEN 'excellent_80_100' WHEN quality_score>=60 THEN 'good_60_79' WHEN quality_score>=45 THEN 'usable_45_59' ELSE 'low_0_44' END,COUNT(*) FROM files WHERE quality_score IS NOT NULL GROUP BY 1")}
    flag_counts = Counter()
    for (flags,) in con.execute("SELECT flags FROM files WHERE flags IS NOT NULL AND flags<>''"):
        flag_counts.update(x for x in str(flags).split(";") if x)
    warnings = []
    if games and total and games[0]["count"] / total > 0.55: warnings.append(f"One game dominates the dataset: {games[0]['game']} ({games[0]['count']/total:.1%}).")
    dup_rate = reasons.get("global_near_duplicate", 0) / max(1, total)
    if dup_rate > 0.35: warnings.append(f"High near-duplicate input rate: {dup_rate:.1%}. Consider a longer capture interval.")
    review_rate = totals.get("REVIEW", 0) / max(1, total)
    if review_rate > 0.25: warnings.append(f"Large REVIEW queue: {review_rate:.1%}. Inspect the contact sheet and thresholds.")
    if totals.get("ERROR", 0): warnings.append(f"There are {totals['ERROR']} decode/missing-file errors.")
    storage = storage_audit(con, raw, clean, review)
    if storage["copies"]:
        warnings.append(
            f"{storage['copies']} KEEP/REVIEW files are full copies rather than hard links and use extra disk space."
        )
    if storage["missing_outputs"]:
        warnings.append(f"{storage['missing_outputs']} KEEP/REVIEW output files are missing.")

    session_anomalies = []
    if len(sessions) >= 3:
        for field in ("avg_mean", "avg_contrast", "avg_color_cast", "avg_quality"):
            vals = [float(s[field]) for s in sessions if s[field] is not None]
            if len(vals) < 3: continue
            mean = sum(vals) / len(vals); sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            if sd <= 1e-8: continue
            for s in sessions:
                if s[field] is not None and abs((float(s[field]) - mean) / sd) >= 2.5:
                    session_anomalies.append({"game": s["game"], "session": s["session"], "metric": field, "value": s[field], "z_score": round((float(s[field]) - mean) / sd, 2)})
    if session_anomalies: warnings.append(f"Detected {len(session_anomalies)} session-level statistical anomalies.")

    samples = query_dicts(con, "SELECT path,game,status,reason,quality_score,novelty_score,flags FROM files WHERE quality_score IS NOT NULL ORDER BY CASE status WHEN 'REVIEW' THEN 0 WHEN 'KEEP' THEN 1 ELSE 2 END, quality_score ASC, novelty_score DESC LIMIT 40")
    sheet_count = contact_sheet(root, raw, samples, report / "audit_contact_sheet.jpg")
    keep = totals.get("KEEP", 0)
    score = 100
    score -= min(25, round(dup_rate * 40))
    score -= min(20, round(review_rate * 30))
    score -= min(20, len(warnings) * 5)
    score -= min(15, len(session_anomalies) * 2)
    result = {"audit_version": "3.2.1", "generated_at": datetime.now().isoformat(timespec="seconds"),
              "dataset_score": max(0, score), "processed_total": total, "training_candidates": keep,
              "status_totals": totals, "reason_totals": reasons, "quality_bands": quality_bands,
              "flag_totals": dict(flag_counts), "game_distribution": games,
              "storage": storage,
              "session_anomalies": session_anomalies, "warnings": warnings,
              "contact_sheet_images": sheet_count}
    (report / "dataset_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_html = "".join(f"<tr><td>{html.escape(str(g['game']))}</td><td>{g['count']}</td><td>{g['kept']}</td><td>{float(g['avg_quality'] or 0):.1f}</td><td>{float(g['avg_novelty'] or 0):.2f}</td></tr>" for g in games)
    warns_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings) or "<li>No high-confidence dataset-level warning.</li>"
    page = f"""<!doctype html><meta charset='utf-8'><title>GameSR Dataset Audit</title>
<style>body{{font:15px system-ui;margin:36px;max-width:1100px;background:#0d1117;color:#e6edf3}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.card{{background:#161b22;padding:16px 22px;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}img{{max-width:100%;border-radius:10px}}code{{color:#79c0ff}}</style>
<h1>GameSR Dataset Audit</h1><div class='cards'><div class='card'><b>Score</b><br>{result['dataset_score']}/100</div><div class='card'><b>Processed</b><br>{total}</div><div class='card'><b>KEEP</b><br>{keep}</div><div class='card'><b>REVIEW</b><br>{totals.get('REVIEW',0)}</div><div class='card'><b>Hard links</b><br>{storage['hardlinks']}</div><div class='card'><b>Fallback copies</b><br>{storage['copies']}</div><div class='card'><b>Space saved</b><br>{storage['hardlink_bytes_saved']/1024**3:.2f} GiB</div><div class='card'><b>Extra copy space</b><br>{storage['copied_bytes']/1024**3:.2f} GiB</div></div>
<h2>Warnings</h2><ul>{warns_html}</ul><h2>Game distribution</h2><table><tr><th>Game</th><th>Total</th><th>KEEP</th><th>Avg quality</th><th>Avg novelty</th></tr>{rows_html}</table>
<h2>Visual sample</h2><p>REVIEW and low-score samples are shown first.</p><img src='audit_contact_sheet.jpg'><h2>Files</h2><p><code>dataset_audit.json</code> contains the complete machine-readable audit.</p>"""
    (report / "dataset_audit.html").write_text(page, encoding="utf-8")
    con.close(); print(json.dumps(result, ensure_ascii=False, indent=2)); return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a GameSR cleaner database")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args(); audit(args.root.resolve())


if __name__ == "__main__": main()
