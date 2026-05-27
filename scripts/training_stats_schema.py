"""Shared training_stats.csv schema and writer helpers.

Keeping this in one module prevents train_ppo.py, train_bc_ppo.py,
rollout_worker.py, and train_offline.py from silently drifting into different
CSV headers. If an older stats file is missing new columns, it is archived and
a fresh file is started so plotting code can trust the header.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Callable, Iterable


TRAINING_STATS_COLUMNS = [
    "timestamp", "game", "total_updates", "steps", "transitions",
    "total_reward", "final_hp", "final_max_hp", "final_floor", "final_act",
    "victory", "terminated", "pg_loss", "vf_loss",
    "entropy", "normalized_entropy",
    "worker",
    "elites_fought", "elites_won", "bosses_fought", "bosses_won",
    "elites_fought_act1", "elites_won_act1", "bosses_fought_act1", "bosses_won_act1",
    "elites_fought_act2", "elites_won_act2", "bosses_fought_act2", "bosses_won_act2",
    "elites_fought_act3", "elites_won_act3", "bosses_fought_act3", "bosses_won_act3",
    "approx_kl", "clip_fraction", "explained_variance",
    "mean_advantage", "std_advantage", "invalid_action_count",
    "mean_chosen_action_prob",
    "bc_loss", "bc_coef",
    "lr", "ent_coef", "auto_tune_action",
    "early_stop",
    "stale_rollouts", "legacy_rollouts", "skipped_rollouts",
    "batch_model_updates", "batch_checkpoint_ids",
]


def _log(log_fn: Callable[[str], None] | None, msg: str) -> None:
    if log_fn is None:
        return
    try:
        log_fn(msg)
    except Exception:
        pass


def _read_header(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            first = f.readline().strip()
        return next(csv.reader([first])) if first else []
    except Exception:
        return []


def _write_header(path: str, columns: Iterable[str] = TRAINING_STATS_COLUMNS) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(list(columns))


def ensure_training_stats_csv(path: str, log_fn: Callable[[str], None] | None = None) -> None:
    """Create or migrate training_stats.csv to the shared schema.

    If the only change is new columns (additive), rewrite the file in-place
    with the updated header and blank values for old rows so history is
    preserved.  If columns were removed or renamed, archive and start fresh.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.exists(path):
        _write_header(path)
        return

    header = _read_header(path)
    missing = [c for c in TRAINING_STATS_COLUMNS if c not in header]
    extra = [c for c in header if c and c not in TRAINING_STATS_COLUMNS]
    if not missing and not extra:
        return

    if missing and not extra:
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                old_rows = list(reader)
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRAINING_STATS_COLUMNS)
                writer.writeheader()
                for row in old_rows:
                    out = {c: row.get(c, "") for c in TRAINING_STATS_COLUMNS}
                    writer.writerow(out)
            _log(
                log_fn,
                f"training_stats.csv: added {len(missing)} new columns in-place "
                f"({len(old_rows)} rows preserved). New columns: {missing}",
            )
            return
        except Exception as e:
            _log(log_fn, f"In-place column migration failed, falling back to archive: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = path.replace(".csv", f"_pre_schema_{ts}.csv")
    try:
        os.replace(path, archive)
        _log(
            log_fn,
            "training_stats.csv schema changed; archived old file to "
            f"{archive} and started a fresh stats file. Missing={missing} Extra={extra}",
        )
    except Exception as e:
        _log(log_fn, f"training_stats.csv schema migration failed: {e}")
        return

    _write_header(path)


def append_training_stats_csv(
    path: str,
    row: dict,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Append a row using the shared schema and csv module escaping."""
    ensure_training_stats_csv(path, log_fn=log_fn)
    try:
        out = {c: row.get(c, "") for c in TRAINING_STATS_COLUMNS}
        if not out.get("timestamp"):
            out["timestamp"] = datetime.now().isoformat()
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=TRAINING_STATS_COLUMNS,
                extrasaction="ignore",
            )
            writer.writerow(out)
    except Exception as e:
        _log(log_fn, f"stats csv append failed: {e}")
