#!/usr/bin/env python3
"""Compact logs/fight_detail.csv so it can't grow unbounded.

fight_detail.csv holds one summarized row per fight (turns, damage, max-hit,
block, won). That's already compact, but it grows one row per fight forever.
This rolls the OLD rows into per-(act, fight_type, won) aggregates appended to
fight_detail_summary.csv, then rewrites fight_detail.csv with only the most
recent rows -- so the per-fight detail needed for analysis stays available while
the file stays bounded.

Safe to run from cron: the workers write fight rows via a fresh open()+close()
per fight (no long-lived fd), so os.replace() only risks losing a fight written
in the few-ms swap window -- negligible.
"""
import csv
import io
import os
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAIL = os.environ.get("FD_DETAIL", os.path.join(_ROOT, "logs", "fight_detail.csv"))
SUMMARY = os.environ.get("FD_SUMMARY", os.path.join(_ROOT, "logs", "fight_detail_summary.csv"))

KEEP_RECENT = int(os.environ.get("FD_KEEP_RECENT", "30000"))  # detail rows to retain
TRIGGER = int(os.environ.get("FD_TRIGGER", "45000"))          # compact once above this

DETAIL_COLS = ["timestamp", "source", "worker", "game", "floor", "act",
               "fight_type", "monsters", "hp_before", "hp_after", "max_hp",
               "won", "turns", "steps", "damage_taken", "max_hit", "block_gained"]
SUMMARY_COLS = ["compacted_at", "ts_first", "ts_last", "act", "fight_type", "won",
                "n", "sum_turns", "sum_damage_taken", "sum_max_hit",
                "sum_block_gained", "sum_hp_before"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    if not os.path.exists(DETAIL):
        return
    raw = open(DETAIL, "rb").read().replace(b"\x00", b"").decode("utf-8", "ignore")
    rows = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("fight_type")]
    if len(rows) <= TRIGGER:
        return

    old, recent = rows[:-KEEP_RECENT], rows[-KEEP_RECENT:]

    agg: dict = {}
    for r in old:
        key = (r.get("act", ""), r.get("fight_type", ""), r.get("won", ""))
        a = agg.get(key)
        if a is None:
            ts = r.get("timestamp", "")
            a = agg[key] = {"n": 0, "sum_turns": 0.0, "sum_damage_taken": 0.0,
                            "sum_max_hit": 0.0, "sum_block_gained": 0.0,
                            "sum_hp_before": 0.0, "ts_first": ts, "ts_last": ts}
        a["n"] += 1
        a["sum_turns"] += _f(r.get("turns"))
        a["sum_damage_taken"] += _f(r.get("damage_taken"))
        a["sum_max_hit"] += _f(r.get("max_hit"))
        a["sum_block_gained"] += _f(r.get("block_gained"))
        a["sum_hp_before"] += _f(r.get("hp_before"))
        ts = r.get("timestamp", "")
        if ts and (not a["ts_first"] or ts < a["ts_first"]):
            a["ts_first"] = ts
        if ts and ts > a["ts_last"]:
            a["ts_last"] = ts

    now = datetime.now().isoformat()
    write_header = not os.path.exists(SUMMARY)
    with open(SUMMARY, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if write_header:
            w.writeheader()
        for (act, ft, won), a in sorted(agg.items()):
            w.writerow({"compacted_at": now, "ts_first": a["ts_first"],
                        "ts_last": a["ts_last"], "act": act, "fight_type": ft,
                        "won": won, "n": a["n"],
                        "sum_turns": round(a["sum_turns"], 1),
                        "sum_damage_taken": round(a["sum_damage_taken"], 1),
                        "sum_max_hit": round(a["sum_max_hit"], 1),
                        "sum_block_gained": round(a["sum_block_gained"], 1),
                        "sum_hp_before": round(a["sum_hp_before"], 1)})

    tmp = DETAIL + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS, extrasaction="ignore")
        w.writeheader()
        for r in recent:
            w.writerow(r)
    os.replace(tmp, DETAIL)
    print(f"{now} compacted {len(old)} old fights -> {len(agg)} summary groups; "
          f"kept {len(recent)} recent detail rows")


if __name__ == "__main__":
    main()
