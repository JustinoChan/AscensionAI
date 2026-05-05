"""
train_offline.py — Central trainer for parallel rollout collection.

Watches a shared directory for .npz transition files written by rollout
workers. Loads batches of game data, runs PPO updates, and saves updated
checkpoints that workers reload.

Usage:
  python train_offline.py --model models/ppo_sts.pt --data rollouts_shared
"""

from __future__ import annotations

import os
import sys
import time
import glob
import argparse
import traceback
from datetime import datetime
from typing import List

_scripts = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_scripts)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from obs_encoder import OBS_SIZE
from sts_gym_env import NUM_ACTIONS

os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "train_offline_debug.log")

def log(msg: str):
    ts = datetime.now().isoformat()
    line = f"{ts}  {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


from ppo_model import PPOTrainer, GameBuffer

# Shared training stats CSV (same file the GUI reads for progress display)
_STATS_CSV = os.path.join(_root, "logs", "training_stats.csv")
_STATS_COLUMNS = [
    "timestamp", "game", "total_updates", "steps", "transitions",
    "total_reward", "final_hp", "final_max_hp", "final_floor", "final_act",
    "victory", "terminated", "pg_loss", "vf_loss", "entropy", "worker",
    "elites_fought", "elites_won", "bosses_fought", "bosses_won",
]

def _init_stats_csv():
    try:
        os.makedirs(os.path.dirname(_STATS_CSV), exist_ok=True)
        if not os.path.exists(_STATS_CSV):
            with open(_STATS_CSV, "w", encoding="utf-8") as f:
                f.write(",".join(_STATS_COLUMNS) + "\n")
    except Exception:
        pass

def _append_training_stats(row: dict):
    try:
        _init_stats_csv()
        with open(_STATS_CSV, "a", encoding="utf-8") as f:
            f.write(",".join(str(row.get(c, "")) for c in _STATS_COLUMNS) + "\n")
    except Exception as e:
        log(f"stats csv append failed: {e}")


# ---------------------------------------------------------------------------
# Transition loading
# ---------------------------------------------------------------------------
def load_npz_files(data_dir: str, consumed: set) -> List[str]:
    """Find new .npz files not yet consumed."""
    pattern = os.path.join(data_dir, "*.npz")
    all_files = sorted(glob.glob(pattern))
    ready: List[str] = []
    now = time.time()
    for f in all_files:
        if f in consumed or f.endswith(".tmp.npz"):
            continue
        try:
            if now - os.path.getmtime(f) < 2.0:
                continue
        except OSError:
            continue
        ready.append(f)
    return ready


def load_transitions(paths: List[str]) -> tuple:
    """Merge multiple .npz files into one GameBuffer.

    Returns (buf, loaded_paths, failed_paths).
    """
    buf = GameBuffer()
    loaded: List[str] = []
    failed: List[str] = []
    for p in paths:
        try:
            with np.load(p, allow_pickle=False) as data:
                obs = data["observations"]
                acts = data["actions"]
                rews = data["rewards"]
                dones = data["dones"]
                masks = data["action_masks"]
                lps = data["log_probs"]
                vals = data["values"]
                n = len(obs)
                lengths = [len(acts), len(rews), len(dones), len(masks), len(lps), len(vals)]
                if any(x != n for x in lengths):
                    raise ValueError(f"array length mismatch: obs={n}, others={lengths}")
                for i in range(n):
                    buf.add(obs[i], int(acts[i]), float(rews[i]), bool(dones[i]),
                            masks[i], float(lps[i]), float(vals[i]))
            loaded.append(p)
        except Exception as e:
            log(f"Failed to load {p}: {e}")
            failed.append(p)
    return buf, loaded, failed


def retire_file(path: str, suffix: str) -> None:
    """Move a rollout out of the trainer glob so it does not block future batches."""
    try:
        target = path + suffix
        if os.path.exists(target):
            target = f"{path}.{int(time.time())}{suffix}"
        os.replace(path, target)
        log(f"Retired {path} -> {target}")
    except OSError as e:
        log(f"Failed to retire {path}: {e}")


def delete_files(paths: List[str]) -> None:
    for f in paths:
        try:
            os.remove(f)
        except OSError as e:
            log(f"Failed to delete {f}: {e}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_sts.pt",
                        help="Model checkpoint path (load + save)")
    parser.add_argument("--data", type=str, default="rollouts_shared",
                        help="Directory where workers write .npz files")
    parser.add_argument("--batch-games", type=int, default=5,
                        help="Minimum game files before triggering a PPO update")
    parser.add_argument("--poll-interval", type=float, default=10.0,
                        help="Seconds between checking for new data")
    parser.add_argument("--delete-consumed", action="store_true",
                        help="Delete .npz files after training on them")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.15,
                        help="Entropy bonus coefficient (higher = more exploration)")
    args = parser.parse_args()

    model_path = os.path.join(_root, args.model)
    data_dir = os.path.join(_root, args.data)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    log("=== OFFLINE TRAINER STARTING ===")
    log(f"Model: {model_path}")
    log(f"Data dir: {data_dir}")
    log(f"Hyperparams: lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, ent_coef={args.ent_coef}")

    trainer = PPOTrainer(
        obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
        lr=args.lr, n_epochs=args.epochs, batch_size=args.batch_size,
        ent_coef=args.ent_coef, net_arch=(256, 256),
    )

    if os.path.isfile(model_path):
        trainer.load(model_path)
        log("Loaded existing model")
    else:
        log("Starting with fresh model")

    consumed: set = set()
    total_transitions = 0
    total_updates = int(getattr(trainer, "total_updates", 0) or 0)

    log("Entering training loop (Ctrl+C to stop)...")

    try:
        while True:
            new_files = load_npz_files(data_dir, consumed)

            if len(new_files) < args.batch_games:
                time.sleep(args.poll_interval)
                continue

            batch_files = new_files[:args.batch_games]
            log(f"Loading {len(batch_files)} new game files...")

            buf, loaded, failed = load_transitions(batch_files)
            n = len(buf)

            for f in failed:
                consumed.add(f)
                retire_file(f, ".bad")

            for f in loaded:
                consumed.add(f)

            if n < 10:
                log(f"Too few transitions ({n}), skipping update")
                if args.delete_consumed:
                    delete_files(loaded)
                else:
                    for f in loaded:
                        retire_file(f, ".short")
                continue

            log(f"Running PPO update on {n} transitions...")
            stats = trainer.update(buf)
            total_transitions += n
            total_updates = int(getattr(trainer, "total_updates", total_updates + 1) or 0)

            if stats:
                log(f"Update #{total_updates}: pg={stats['pg_loss']:.4f} "
                    f"vf={stats['vf_loss']:.4f} ent={stats['entropy']:.4f} "
                    f"transitions={n} total={total_transitions}")
                _append_training_stats({
                    "timestamp": datetime.now().isoformat(),
                    "total_updates": total_updates,
                    "transitions": n,
                    "pg_loss": round(stats["pg_loss"], 6),
                    "vf_loss": round(stats["vf_loss"], 6),
                    "entropy": round(stats["entropy"], 6),
                    "worker": "trainer",
                })

            trainer.save(model_path)

            if args.delete_consumed:
                delete_files(loaded)

    except KeyboardInterrupt:
        log("Interrupted. Saving final model...")
        trainer.save(model_path)
        log(f"Done. {total_updates} updates, {total_transitions} total transitions.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
