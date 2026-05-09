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

import numpy as np
import torch

from obs_encoder import OBS_SIZE
from sts_gym_env import NUM_ACTIONS

os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "train_offline_debug.log")
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

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
    "approx_kl", "clip_fraction", "explained_variance",
    "mean_advantage", "std_advantage", "invalid_action_count",
    "mean_chosen_action_prob", "bc_loss", "bc_coef", "early_stop",
    "stale_rollouts", "legacy_rollouts", "skipped_rollouts",
    "batch_model_updates", "batch_checkpoint_ids",
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


def _torch_backend_summary() -> str:
    return (
        f"torch={getattr(torch, '__version__', '?')} "
        f"cuda_built={getattr(torch.version, 'cuda', None)} "
        f"hip_built={getattr(torch.version, 'hip', None)} "
        f"cuda_available={torch.cuda.is_available()}"
    )


def _resolve_device(requested: str) -> str:
    """Return a usable torch device.

    ROCm-backed AMD PyTorch still reports devices through torch.cuda, so
    "cuda" is the expected device name for both NVIDIA CUDA and AMD ROCm.
    """
    try:
        gpu_available = bool(torch.cuda.is_available())
    except Exception as e:
        log(f"GPU availability check failed: {e}")
        gpu_available = False

    if requested == "cpu":
        return "cpu"
    if gpu_available:
        name = ""
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        log(f"Using GPU device cuda{f' ({name})' if name else ''}; {_torch_backend_summary()}")
        return "cuda"

    if requested == "gpu":
        log(
            "WARNING: GPU training was requested, but this PyTorch install has "
            f"no usable GPU backend; falling back to CPU. {_torch_backend_summary()}"
        )
        log(
            "For AMD GPUs, install a ROCm-enabled PyTorch build; the ROCm build "
            "still uses torch's 'cuda' device name."
        )
    return "cpu"


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


def _scalar(data, key: str, default=None):
    if key not in data.files:
        return default
    val = data[key]
    try:
        return val.item()
    except Exception:
        return val


def read_rollout_meta(path: str) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {
                "model_update_number": _scalar(data, "model_update_number"),
                "checkpoint_id": _scalar(data, "checkpoint_id"),
                "worker_id": _scalar(data, "worker_id"),
                "episode_number": _scalar(data, "episode_number"),
                "entropy_coeff": _scalar(data, "entropy_coeff"),
                "learning_rate": _scalar(data, "learning_rate"),
                "created_at": _scalar(data, "created_at"),
            }
    except Exception as e:
        log(f"Failed to read rollout metadata {path}: {e}")
        return {}


def load_bc_demo(path: str) -> tuple:
    with np.load(path, allow_pickle=False) as data:
        obs = data["observations"]
        actions = data["actions"]
        masks = data["action_masks"]
    n = len(obs)
    if len(actions) != n or len(masks) != n:
        raise ValueError(f"BC demo length mismatch: obs={n}, actions={len(actions)}, masks={len(masks)}")
    return obs, actions, masks


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
            if VERBOSE:
                log(f"Loaded rollout {os.path.basename(p)} transitions={n}")
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
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_sts.pt",
                        help="Model checkpoint path (load + save)")
    parser.add_argument("--data", type=str, default="rollouts_shared",
                        help="Directory where workers write .npz files")
    parser.add_argument("--batch-games", type=int, default=60,
                        help="Minimum game files before triggering a PPO update")
    parser.add_argument("--poll-interval", type=float, default=10.0,
                        help="Seconds between checking for new data")
    parser.add_argument("--delete-consumed", action="store_true",
                        help="Delete .npz files after training on them")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="Entropy bonus coefficient (higher = more exploration)")
    parser.add_argument("--clip", type=float, default=0.15,
                        help="PPO clip range (default: 0.15)")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Stop PPO epochs early when approx KL exceeds this value")
    parser.add_argument("--max-rollout-lag", type=int, default=2,
                        help="Reject rollouts more than N model updates behind")
    parser.add_argument("--allow-legacy-rollouts", action="store_true",
                        help="Allow rollout files without checkpoint metadata")
    parser.add_argument("--bc-demo", type=str, default=None,
                        help="Optional BC demo npz for imitation anchor loss")
    parser.add_argument("--bc-coef", type=float, default=1.0,
                        help="BC anchor coefficient if --bc-demo exists")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "gpu"],
                        help="Device for training: auto (GPU if available), cpu, or gpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Write detailed polling and rollout loading logs")
    args = parser.parse_args()
    VERBOSE = VERBOSE or args.verbose

    device = _resolve_device(args.device)

    model_path = os.path.join(_root, args.model)
    data_dir = os.path.join(_root, args.data)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    log("=== OFFLINE TRAINER STARTING ===")
    log(f"Model: {model_path}")
    log(f"Data dir: {data_dir}")
    log(f"Hyperparams: lr={args.lr}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, ent_coef={args.ent_coef}, "
        f"clip={args.clip}, target_kl={args.target_kl}, "
        f"batch_games={args.batch_games}, max_rollout_lag={args.max_rollout_lag}, "
        f"allow_legacy={args.allow_legacy_rollouts}, device={device}, verbose={VERBOSE}")

    trainer = PPOTrainer(
        obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device=device,
        lr=args.lr, n_epochs=args.epochs, batch_size=args.batch_size,
        ent_coef=args.ent_coef, clip_range=args.clip, net_arch=(256, 256),
        target_kl=args.target_kl,
    )

    if os.path.isfile(model_path):
        trainer.load(model_path)
        trainer.set_lr(args.lr)
        log(f"Loaded existing model; optimizer lr reset to {args.lr}")
    else:
        log("Starting with fresh model")

    bc_demo_path = args.bc_demo
    if bc_demo_path is None:
        bc_demo_path = model_path.replace(".pt", "_bc_demos.npz")
    elif not os.path.isabs(bc_demo_path):
        bc_demo_path = os.path.join(_root, bc_demo_path)
    if args.bc_coef > 0.0 and os.path.isfile(bc_demo_path):
        try:
            bc_obs, bc_actions, bc_masks = load_bc_demo(bc_demo_path)
            trainer.set_bc_reference(
                bc_obs, bc_actions, bc_masks,
                coef=args.bc_coef,
                batch_size=args.batch_size,
            )
            log(f"Loaded BC anchor demos: {len(bc_actions)} samples "
                f"coef={args.bc_coef} path={bc_demo_path}")
        except Exception as e:
            log(f"Failed to load BC anchor demos {bc_demo_path}: {e}")
    elif args.bc_coef > 0.0:
        log(f"No BC anchor demos found at {bc_demo_path}; PPO runs without BC anchor")

    consumed: set = set()
    total_transitions = 0
    total_updates = int(getattr(trainer, "total_updates", 0) or 0)
    stale_rollouts_total = 0
    legacy_rollouts_total = 0
    skipped_rollouts_total = 0

    log("Entering training loop (Ctrl+C to stop)...")

    try:
        while True:
            new_files = load_npz_files(data_dir, consumed)
            eligible_files: List[str] = []
            stale_files: List[str] = []
            legacy_files: List[str] = []
            batch_updates: list[int] = []
            batch_checkpoint_ids: list[str] = []
            current_update = int(getattr(trainer, "total_updates", 0) or 0)
            for f in new_files:
                meta = read_rollout_meta(f)
                model_update = meta.get("model_update_number")
                checkpoint_id = meta.get("checkpoint_id")
                try:
                    model_update_int = int(model_update)
                except Exception:
                    model_update_int = None
                if model_update_int is None:
                    if not args.allow_legacy_rollouts:
                        legacy_files.append(f)
                        continue
                else:
                    lag = current_update - model_update_int
                    if lag > args.max_rollout_lag:
                        stale_files.append(f)
                        continue
                    batch_updates.append(model_update_int)
                if checkpoint_id not in (None, ""):
                    batch_checkpoint_ids.append(str(checkpoint_id))
                eligible_files.append(f)

            if legacy_files:
                legacy_rollouts_total += len(legacy_files)
                for f in legacy_files:
                    consumed.add(f)
                log(f"Rejecting {len(legacy_files)} legacy rollout(s) without metadata; "
                    f"use --allow-legacy-rollouts to train on old files")
                if args.delete_consumed:
                    delete_files(legacy_files)
                else:
                    for f in legacy_files:
                        retire_file(f, ".legacy")

            if stale_files:
                stale_rollouts_total += len(stale_files)
                for f in stale_files:
                    consumed.add(f)
                log(f"Rejecting {len(stale_files)} stale rollout(s): "
                    f"current_update={current_update} max_lag={args.max_rollout_lag}")
                if args.delete_consumed:
                    delete_files(stale_files)
                else:
                    for f in stale_files:
                        retire_file(f, ".stale")

            if len(eligible_files) < args.batch_games:
                if VERBOSE:
                    log(f"Waiting for rollouts: ready={len(eligible_files)} "
                        f"need={args.batch_games} consumed={len(consumed)} "
                        f"stale_total={stale_rollouts_total} "
                        f"legacy_total={legacy_rollouts_total}")
                time.sleep(args.poll_interval)
                continue

            batch_files = eligible_files[:args.batch_games]
            log(f"Loading {len(batch_files)} new game files...")
            if VERBOSE:
                log("Batch files: " + ", ".join(os.path.basename(f) for f in batch_files))

            buf, loaded, failed = load_transitions(batch_files)
            n = len(buf)

            for f in failed:
                consumed.add(f)
                retire_file(f, ".bad")

            for f in loaded:
                consumed.add(f)
            skipped_rollouts_total += len(failed)

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
                    f"kl={stats.get('approx_kl', 0.0):.5f} "
                    f"clip={stats.get('clip_fraction', 0.0):.3f} "
                    f"ev={stats.get('explained_variance', 0.0):.3f} "
                    f"early_stop={stats.get('early_stop', 0)} "
                    f"transitions={n} total={total_transitions}")
                _append_training_stats({
                    "timestamp": datetime.now().isoformat(),
                    "total_updates": total_updates,
                    "transitions": n,
                    "pg_loss": round(stats["pg_loss"], 6),
                    "vf_loss": round(stats["vf_loss"], 6),
                    "entropy": round(stats["entropy"], 6),
                    "worker": "trainer",
                    "approx_kl": round(stats.get("approx_kl", 0.0), 8),
                    "clip_fraction": round(stats.get("clip_fraction", 0.0), 6),
                    "explained_variance": round(stats.get("explained_variance", 0.0), 6),
                    "mean_advantage": round(stats.get("mean_advantage", 0.0), 6),
                    "std_advantage": round(stats.get("std_advantage", 0.0), 6),
                    "invalid_action_count": stats.get("invalid_action_count", 0),
                    "mean_chosen_action_prob": round(stats.get("mean_chosen_action_prob", 0.0), 6),
                    "bc_loss": round(stats.get("bc_loss", 0.0), 6),
                    "bc_coef": round(stats.get("bc_coef", 0.0), 6),
                    "early_stop": stats.get("early_stop", 0),
                    "stale_rollouts": stale_rollouts_total,
                    "legacy_rollouts": legacy_rollouts_total,
                    "skipped_rollouts": skipped_rollouts_total,
                    "batch_model_updates": ";".join(str(x) for x in sorted(set(batch_updates))),
                    "batch_checkpoint_ids": ";".join(sorted(set(batch_checkpoint_ids))),
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
