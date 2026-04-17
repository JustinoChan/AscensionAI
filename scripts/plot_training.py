"""
plot_training.py — Visualize the learning curve from training_stats.csv.

Reads logs/training_stats.csv (written by train_ppo.py after every game)
and draws rolling-average curves for:
    - Final floor reached
    - Total episode reward
    - Win rate
    - Entropy (exploration)
    - Policy / value loss

Usage:
    python scripts/plot_training.py                 # show interactive window
    python scripts/plot_training.py --save out.png   # write PNG instead
    python scripts/plot_training.py --window 25      # rolling window size
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List


def _load_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"no stats file at {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def _num(v, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _rolling(values: List[float], window: int) -> List[float]:
    out = []
    running = 0.0
    count = 0
    from collections import deque

    buf: deque = deque()
    for v in values:
        if v != v:  # NaN — skip
            out.append(float("nan"))
            continue
        buf.append(v)
        running += v
        count += 1
        if count > window:
            dropped = buf.popleft()
            running -= dropped
            count -= 1
        out.append(running / count)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Save to PNG instead of showing")
    parser.add_argument("--window", type=int, default=25, help="Rolling average window")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv or os.path.join(root, "logs", "training_stats.csv")

    rows = _load_rows(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} has no rows yet")

    games = [int(_num(r.get("game"), 0)) for r in rows]
    floors = [_num(r.get("final_floor")) for r in rows]
    rewards = [_num(r.get("total_reward")) for r in rows]
    victories = [_num(r.get("victory"), 0.0) for r in rows]
    pg = [_num(r.get("pg_loss")) for r in rows]
    vf = [_num(r.get("vf_loss")) for r in rows]
    ent = [_num(r.get("entropy")) for r in rows]

    w = args.window
    floor_avg = _rolling(floors, w)
    reward_avg = _rolling(rewards, w)
    win_avg = _rolling(victories, w)
    ent_avg = _rolling(ent, w)
    pg_avg = _rolling(pg, w)
    vf_avg = _rolling(vf, w)

    n_total = len(rows)
    n_wins = int(sum(v for v in victories if v == v))
    best_floor = max((f for f in floors if f == f), default=float("nan"))
    last_floor_avg = floor_avg[-1] if floor_avg else float("nan")

    print(f"Stats file: {csv_path}")
    print(f"Games logged: {n_total}  wins: {n_wins}  overall win rate: {n_wins / n_total:.1%}")
    print(f"Best floor reached: {best_floor}")
    print(f"Rolling-{w} avg floor (last): {last_floor_avg:.2f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; install with: pip install matplotlib", file=sys.stderr)
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(f"AscensionAI training curves (rolling window = {w})")

    ax = axes[0][0]
    ax.plot(games, floors, color="#bbb", lw=0.6, label="per game")
    ax.plot(games, floor_avg, color="#1f77b4", lw=2, label=f"avg{w}")
    ax.set_ylabel("Final floor")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(games, rewards, color="#bbb", lw=0.6, label="per game")
    ax.plot(games, reward_avg, color="#2ca02c", lw=2, label=f"avg{w}")
    ax.set_ylabel("Episode reward")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(games, win_avg, color="#d62728", lw=2, label=f"win rate avg{w}")
    ax.set_ylabel("Win rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(games, ent_avg, color="#9467bd", lw=2, label=f"entropy avg{w}")
    ax.set_ylabel("Policy entropy")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2][0]
    ax.plot(games, pg_avg, color="#ff7f0e", lw=2, label=f"pg_loss avg{w}")
    ax.set_ylabel("Policy loss")
    ax.set_xlabel("Game #")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2][1]
    ax.plot(games, vf_avg, color="#17becf", lw=2, label=f"vf_loss avg{w}")
    ax.set_ylabel("Value loss")
    ax.set_xlabel("Game #")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if args.save:
        out = args.save if os.path.isabs(args.save) else os.path.join(root, args.save)
        plt.savefig(out, dpi=120)
        print(f"Saved plot to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
