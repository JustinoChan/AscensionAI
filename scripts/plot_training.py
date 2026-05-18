"""
plot_training.py — Visualize the learning curve from training_stats.csv.

Reads logs/training_stats.csv (written by train_ppo.py after every game)
and draws rolling-average curves for:
    - Final floor reached
    - Total episode reward
    - Win rate
    - Boss win rate / Elite win rate
    - Normalized entropy (exploration scaled by legal actions)
    - PPO KL / clip fraction
    - Policy / value loss

Usage:
    python scripts/plot_training.py                 # show interactive window
    python scripts/plot_training.py --save out.png   # write PNG instead
    python scripts/plot_training.py --window 100     # primary rolling window size
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


def _expanding(values: List[float]) -> List[float]:
    out = []
    running = 0.0
    count = 0
    for v in values:
        if v != v:  # NaN
            out.append(float("nan"))
            continue
        running += v
        count += 1
        out.append(running / count)
    return out


def _plot_trend(ax, x, values, short_avg, main_avg, lifetime_avg,
                ylabel: str, color: str, short_color: str,
                short_window: int, main_window: int, per_label: str = "per game") -> None:
    ax.plot(x, values, color="#bbb", lw=0.55, alpha=0.7, label=per_label)
    if short_window > 0 and short_window != main_window:
        ax.plot(x, short_avg, color=short_color, lw=1.1, alpha=0.65,
                label=f"avg{short_window}")
    ax.plot(x, main_avg, color=color, lw=2.3, label=f"avg{main_window}")
    ax.plot(x, lifetime_avg, color="#222", lw=1.1, ls="--", alpha=0.75,
            label="lifetime")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Save to PNG instead of showing")
    parser.add_argument("--window", type=int, default=100, help="Primary rolling average window")
    parser.add_argument("--short-window", type=int, default=25,
                        help="Secondary short-term rolling average window")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv or os.path.join(root, "logs", "training_stats.csv")

    rows = _load_rows(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} has no rows yet")

    episode_rows = [r for r in rows if r.get("final_floor") not in (None, "")]
    update_rows = [
        r for r in rows
        if any(
            r.get(k) not in (None, "")
            for k in ("pg_loss", "vf_loss", "entropy", "normalized_entropy", "approx_kl", "clip_fraction")
        )
    ]
    if not episode_rows:
        raise SystemExit(f"{csv_path} has no episode rows yet")

    episodes = list(range(1, len(episode_rows) + 1))
    updates = [
        int(_num(r.get("total_updates"), i + 1))
        for i, r in enumerate(update_rows)
    ]

    floors = [_num(r.get("final_floor")) for r in episode_rows]
    rewards = [_num(r.get("total_reward")) for r in episode_rows]
    victories = []
    for r in episode_rows:
        v = _num(r.get("victory"), 0.0)
        act = _num(r.get("final_act"), 0.0)
        floor = _num(r.get("final_floor"), 0.0)
        victories.append(1.0 if v and (act >= 3 or floor >= 50) else 0.0)
    boss_wins = []
    elite_wins = []
    for r in episode_rows:
        bf = _num(r.get("bosses_fought"), 0.0)
        bw = _num(r.get("bosses_won"), 0.0)
        boss_wins.append(bw / bf if bf > 0 else float("nan"))
        ef = _num(r.get("elites_fought"), 0.0)
        ew = _num(r.get("elites_won"), 0.0)
        elite_wins.append(ew / ef if ef > 0 else float("nan"))

    pg = [_num(r.get("pg_loss")) for r in update_rows]
    vf = [_num(r.get("vf_loss")) for r in update_rows]
    raw_ent = [_num(r.get("entropy")) for r in update_rows]
    has_norm_entropy = any(r.get("normalized_entropy") not in (None, "") for r in update_rows)
    ent = [
        _num(r.get("normalized_entropy" if has_norm_entropy else "entropy"))
        for r in update_rows
    ]
    ent_label = "Normalized policy entropy" if has_norm_entropy else "Raw policy entropy"
    kl = [_num(r.get("approx_kl")) for r in update_rows]
    clip = [_num(r.get("clip_fraction")) for r in update_rows]

    w = max(1, args.window)
    sw = max(0, args.short_window)
    floor_avg = _rolling(floors, w)
    floor_short = _rolling(floors, sw) if sw else []
    floor_lifetime = _expanding(floors)
    reward_avg = _rolling(rewards, w)
    reward_short = _rolling(rewards, sw) if sw else []
    reward_lifetime = _expanding(rewards)
    win_avg = _rolling(victories, w)
    win_short = _rolling(victories, sw) if sw else []
    win_lifetime = _expanding(victories)
    boss_avg = _rolling(boss_wins, w)
    boss_short = _rolling(boss_wins, sw) if sw else []
    boss_lifetime = _expanding(boss_wins)
    elite_avg = _rolling(elite_wins, w)
    elite_short = _rolling(elite_wins, sw) if sw else []
    elite_lifetime = _expanding(elite_wins)
    ent_avg = _rolling(ent, w)
    ent_short = _rolling(ent, sw) if sw else []
    raw_ent_avg = _rolling(raw_ent, w)
    kl_avg = _rolling(kl, w)
    kl_short = _rolling(kl, sw) if sw else []
    clip_avg = _rolling(clip, w)
    clip_short = _rolling(clip, sw) if sw else []
    pg_avg = _rolling(pg, w)
    pg_short = _rolling(pg, sw) if sw else []
    vf_avg = _rolling(vf, w)
    vf_short = _rolling(vf, sw) if sw else []

    n_total = len(episode_rows)
    n_wins = int(sum(v for v in victories if v == v))
    best_floor = max((f for f in floors if f == f), default=float("nan"))
    last_floor_avg = floor_avg[-1] if floor_avg else float("nan")
    last_floor_short = floor_short[-1] if floor_short else float("nan")
    last_floor_lifetime = floor_lifetime[-1] if floor_lifetime else float("nan")

    print(f"Stats file: {csv_path}")
    print(f"Rows logged: {len(rows)}  episodes: {n_total}  update rows: {len(update_rows)}")
    if updates:
        print(f"PPO updates: {updates[0]}..{updates[-1]}")
    print(f"Games logged: {n_total}  wins: {n_wins}  overall win rate: {n_wins / n_total:.1%}")
    print(f"Best floor reached: {best_floor}")
    if sw:
        print(f"Rolling-{sw} avg floor (last): {last_floor_short:.2f}")
    print(f"Rolling-{w} avg floor (last): {last_floor_avg:.2f}")
    print(f"Lifetime avg floor (last): {last_floor_lifetime:.2f}")

    try:
        if args.save:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; install with: pip install matplotlib", file=sys.stderr)
        raise SystemExit(1)

    fig, axes = plt.subplots(5, 2, figsize=(13, 15))
    subtitle = f"primary rolling window = {w}"
    if sw and sw != w:
        subtitle += f", short window = {sw}"
    fig.suptitle(f"AscensionAI training curves ({subtitle})")

    ax = axes[0][0]
    _plot_trend(ax, episodes, floors, floor_short, floor_avg, floor_lifetime,
                "Final floor", "#1f77b4", "#74a9cf", sw, w)

    ax = axes[0][1]
    _plot_trend(ax, episodes, rewards, reward_short, reward_avg, reward_lifetime,
                "Episode reward", "#2ca02c", "#98df8a", sw, w)

    ax = axes[1][0]
    _plot_trend(ax, episodes, victories, win_short, win_avg, win_lifetime,
                "Win rate", "#d62728", "#ff9896", sw, w)
    ax.set_ylabel("Win rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Game #")

    ax = axes[1][1]
    _plot_trend(ax, episodes, boss_wins, boss_short, boss_avg, boss_lifetime,
                "Boss win rate", "#8c564b", "#c49c94", sw, w, per_label="per game")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Game #")

    ax = axes[2][0]
    _plot_trend(ax, episodes, elite_wins, elite_short, elite_avg, elite_lifetime,
                "Elite win rate", "#e377c2", "#f7b6d2", sw, w, per_label="per game")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Game #")

    ax = axes[2][1]
    if has_norm_entropy:
        ax.axhspan(0.25, 0.50, color="#9467bd", alpha=0.08, label="target 0.25–0.50")
        ax.plot(updates, raw_ent_avg, color="#bbb", lw=0.8, alpha=0.45, label=f"raw entropy avg{w}")
    if sw and sw != w:
        ax.plot(updates, ent_short, color="#c5b0d5", lw=1.1, alpha=0.65,
                label=f"entropy avg{sw}")
    ax.plot(updates, ent_avg, color="#9467bd", lw=2.3, label=f"entropy avg{w}")
    ax.set_ylabel(ent_label)
    ax.set_xlabel("PPO update #")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3][0]
    if sw and sw != w:
        ax.plot(updates, kl_short, color="#ffbb78", lw=1.1, alpha=0.65,
                label=f"approx_kl avg{sw}")
    ax.plot(updates, kl_avg, color="#ff7f0e", lw=2.3, label=f"approx_kl avg{w}")
    ax.axhline(0.03, color="#555", lw=1.0, ls="--", alpha=0.65, label="target_kl 0.03")
    ax.set_ylabel("Approx KL")
    ax.set_xlabel("PPO update #")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3][1]
    if sw and sw != w:
        ax.plot(updates, clip_short, color="#98df8a", lw=1.1, alpha=0.65,
                label=f"clip_fraction avg{sw}")
    ax.plot(updates, clip_avg, color="#2ca02c", lw=2.3, label=f"clip_fraction avg{w}")
    ax.axhspan(0.05, 0.25, color="#2ca02c", alpha=0.08, label="healthy-ish 0.05–0.25")
    ax.set_ylabel("Clip fraction")
    ax.set_xlabel("PPO update #")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[4][0]
    if sw and sw != w:
        ax.plot(updates, pg_short, color="#ffbb78", lw=1.1, alpha=0.65,
                label=f"pg_loss avg{sw}")
    ax.plot(updates, pg_avg, color="#ff7f0e", lw=2.3, label=f"pg_loss avg{w}")
    ax.set_ylabel("Policy loss")
    ax.set_xlabel("PPO update #")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[4][1]
    if sw and sw != w:
        ax.plot(updates, vf_short, color="#9edae5", lw=1.1, alpha=0.65,
                label=f"vf_loss avg{sw}")
    ax.plot(updates, vf_avg, color="#17becf", lw=2.3, label=f"vf_loss avg{w}")
    ax.set_ylabel("Value loss")
    ax.set_xlabel("PPO update #")
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
