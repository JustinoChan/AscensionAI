"""Shared per-game behavior-cloning stats logging."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Callable


_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
BC_STATS_CSV = os.path.join(_ROOT, "logs", "bc_stats.csv")

BC_STATS_COLUMNS = [
    "timestamp", "run_id", "source", "game", "target_games",
    "steps", "samples", "skipped_samples",
    "final_hp", "final_max_hp", "final_floor", "final_act",
    "victory", "terminated",
    "elites_fought", "elites_won", "bosses_fought", "bosses_won",
    "checkpoint_path", "model_path",
]


def append_bc_stats(row: dict, log: Callable[[str], None] | None = None) -> None:
    try:
        os.makedirs(os.path.dirname(BC_STATS_CSV), exist_ok=True)
        exists = os.path.exists(BC_STATS_CSV)
        with open(BC_STATS_CSV, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BC_STATS_COLUMNS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            out = {c: row.get(c, "") for c in BC_STATS_COLUMNS}
            out["timestamp"] = out["timestamp"] or datetime.now().isoformat()
            writer.writerow(out)
    except Exception as e:
        if log is not None:
            log(f"bc stats append failed: {e}")
