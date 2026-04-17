"""
analyze_trace.py — Summarize a game_logger JSONL trace.

Groups states by (screen_type, room_phase, available-flag fingerprint) and
prints each unique combination along with a few example choice_list values
and the number of times it was observed. This gives a compact picture of
what CommunicationMod exposes for every situation encountered during
the manual logging run.

Usage:
    python scripts/analyze_trace.py                       # newest trace in logs/
    python scripts/analyze_trace.py logs/game_trace_X.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any


def _newest_trace() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root, "logs")
    if not os.path.isdir(log_dir):
        raise SystemExit(f"no logs directory at {log_dir}")
    candidates = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith("game_trace_") and f.endswith(".jsonl")
    ]
    if not candidates:
        raise SystemExit(f"no game_trace_*.jsonl files in {log_dir}")
    return max(candidates, key=os.path.getmtime)


def _fingerprint(rec: dict) -> tuple:
    flags = (
        rec.get("in_combat"),
        rec.get("play_available"),
        rec.get("end_available"),
        rec.get("potion_available"),
        rec.get("proceed_available"),
        rec.get("cancel_available"),
        rec.get("choice_available"),
    )
    return (
        rec.get("screen_type"),
        rec.get("room_phase"),
        flags,
        len(rec.get("choice_list") or []),
    )


def _fmt_flags(flags: tuple) -> str:
    names = [
        "combat",
        "play",
        "end",
        "potion",
        "proceed",
        "cancel",
        "choice",
    ]
    return ",".join(n for n, v in zip(names, flags) if v) or "-"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else _newest_trace()
    print(f"Analyzing: {path}")

    buckets: dict[tuple, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "choice_samples": set(), "floors": set()}
    )
    total = 0
    errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "error" in rec and rec.get("error"):
                errors += 1
                continue
            if not rec.get("in_game"):
                continue
            total += 1
            key = _fingerprint(rec)
            b = buckets[key]
            b["count"] += 1
            choices = rec.get("choice_list") or []
            if choices and len(b["choice_samples"]) < 8:
                b["choice_samples"].add(" | ".join(str(c) for c in choices))
            floor = rec.get("floor")
            if floor is not None and len(b["floors"]) < 12:
                b["floors"].add(floor)

    print(f"Total in-game states: {total}  unique shapes: {len(buckets)}  errors: {errors}")
    print()

    for key, b in sorted(buckets.items(), key=lambda kv: -kv[1]["count"]):
        screen_type, room_phase, flags, n_choices = key
        print(f"[{b['count']:5d}x] {screen_type}  phase={room_phase}")
        print(f"         flags: {_fmt_flags(flags)}   choices: {n_choices}")
        if b["floors"]:
            floors = sorted(b["floors"])
            print(f"         floors seen: {floors}")
        for sample in sorted(b["choice_samples"]):
            print(f"         choice_list: {sample}")
        print()


if __name__ == "__main__":
    main()
