"""
analyze_training_rewards.py - Check whether shaped reward matches progress.

Usage:
  python scripts/analyze_training_rewards.py
  python scripts/analyze_training_rewards.py --csv logs/training_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Iterable


def _num(value, default=float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--csv", default=os.path.join(root, "logs", "training_stats.csv"))
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r.get("final_floor") not in (None, "")
        ]

    if not rows:
        raise SystemExit(f"No episode rows found in {args.csv}")

    rewards = [_num(r.get("total_reward")) for r in rows]
    floors = [_num(r.get("final_floor")) for r in rows]
    acts = [_num(r.get("final_act")) for r in rows]
    wins = [_num(r.get("victory"), 0.0) for r in rows]
    elite_wins = [_num(r.get("elites_won"), 0.0) for r in rows]
    elite_fights = [_num(r.get("elites_fought"), 0.0) for r in rows]
    boss_wins = [_num(r.get("bosses_won"), 0.0) for r in rows]
    boss_fights = [_num(r.get("bosses_fought"), 0.0) for r in rows]

    print(f"Stats file: {args.csv}")
    print(f"Episodes: {len(rows)}")
    print(f"Reward vs final_floor: { _pearson(rewards, floors): .3f}")
    print(f"Reward vs final_act:   { _pearson(rewards, acts): .3f}")
    print(f"Reward vs victory:     { _pearson(rewards, wins): .3f}")
    print(f"Reward vs elite_wins:  { _pearson(rewards, elite_wins): .3f}")
    print(f"Reward vs elite_fights:{ _pearson(rewards, elite_fights): .3f}")
    print(f"Reward vs boss_wins:   { _pearson(rewards, boss_wins): .3f}")
    print(f"Reward vs boss_fights: { _pearson(rewards, boss_fights): .3f}")


if __name__ == "__main__":
    main()
