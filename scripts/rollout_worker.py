"""
rollout_worker.py — Collect rollouts using the current model checkpoint.

Designed for parallel operation: multiple STS instances each run a copy of
this script, writing transitions to a shared directory. A separate training
process (train_offline.py) reads these files and updates the model.

Each worker:
  1. Loads the latest model checkpoint on startup and every N games
  2. Plays games, using the RL policy for combat + heuristic for non-combat
  3. Writes per-game transitions as .npz files to a shared directory
  4. Reloads the model periodically to incorporate training progress

Usage (in CommunicationMod config.properties):
  command=python rollout_worker.py --model models/ppo_sts.pt --out rollouts_shared --id 1
"""

from __future__ import annotations

import sys
import os

_real_stdout = sys.stdout
sys.stdout = sys.stderr

import spirecomm.communication.coordinator as _coord_module

def _patched_write_stdout(output_queue):
    while True:
        output = output_queue.get()
        _real_stdout.write(output + '\n')
        _real_stdout.flush()

_coord_module.write_stdout = _patched_write_stdout

_scripts = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_scripts)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import traceback
import time
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import torch

_worker_id = "?"
DEBUG_LOG = None

def _init_log(worker_id: str):
    global DEBUG_LOG, _worker_id
    _worker_id = worker_id
    DEBUG_LOG = os.path.join(_root, f"worker_{worker_id}_debug.log")

def log(msg: str):
    if DEBUG_LOG is None:
        return
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  [W{_worker_id}] {msg}\n")
            f.flush()
    except Exception:
        pass

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import (
    Action, ChooseAction, PlayCardAction, PotionAction, StartGameAction,
)
from spirecomm.spire.character import PlayerClass

from obs_encoder import OBS_SIZE, encode_game_state, living_monsters
from sts_gym_env import (
    NUM_ACTIONS, compute_action_mask, flat_action_to_spire_action,
    RewardTracker, _NOOP, _CHOOSE_START, _PROCEED, _LEAVE,
)
from ppo_model import PPOTrainer


# ---------------------------------------------------------------------------
# Per-game transition buffer (mirrors GameBuffer from train_ppo.py)
# ---------------------------------------------------------------------------
class TransitionBuffer:
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.action_masks: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(self, obs, action, reward, done, mask, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(mask)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self):
        return len(self.observations)

    def clear(self):
        self.__init__()

    def save_npz(self, path: str):
        np.savez_compressed(
            path,
            observations=np.array(self.observations, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.int64),
            rewards=np.array(self.rewards, dtype=np.float32),
            dones=np.array(self.dones, dtype=np.bool_),
            action_masks=np.array(self.action_masks, dtype=np.bool_),
            log_probs=np.array(self.log_probs, dtype=np.float32),
            values=np.array(self.values, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Worker agent (same hybrid approach: heuristic non-combat, RL combat)
# ---------------------------------------------------------------------------
class WorkerAgent:
    def __init__(self, trainer: PPOTrainer, model_path: str, out_dir: str,
                 worker_id: str, reload_every: int = 5):
        self.trainer = trainer
        self.model_path = model_path
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.reload_every = reload_every

        self.buffer = TransitionBuffer()
        self.reward_tracker = RewardTracker()
        self.total_games = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.initialized = False

        self.prev_obs = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.prev_mask = None
        self.pending_reward = 0.0

        self._stuck_floor = -1
        self._stuck_count = 0
        self._model_mtime = 0.0

    def _maybe_reload_model(self):
        """Reload model if a newer checkpoint exists."""
        try:
            if os.path.isfile(self.model_path):
                mtime = os.path.getmtime(self.model_path)
                if mtime > self._model_mtime:
                    self.trainer.load(self.model_path)
                    self._model_mtime = mtime
                    log(f"Reloaded model (mtime={mtime:.0f})")
        except Exception as e:
            log(f"Model reload failed: {e}")

    def on_state_change(self, gs) -> Action:
        try:
            return self._handle_state(gs)
        except Exception as e:
            log(f"ERROR: {e}")
            log(traceback.format_exc())
            return Action("state")

    def _handle_state(self, gs) -> Action:
        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)

        screen_type = getattr(gs, "screen_type", None)
        screen_name = str(getattr(screen_type, "name", screen_type) or "NONE")
        terminal = screen_name in {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}

        victory = False
        if terminal:
            scr_obj = getattr(gs, "screen", None)
            victory = bool(getattr(scr_obj, "victory", False)) or screen_name in {"COMPLETE", "VICTORY"}

        if not self.initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.initialized = True
            self.prev_obs = obs
            self.prev_mask = mask
            log(f"Game #{self.total_games + 1} started, floor={getattr(gs, 'floor', '?')}")

        reward = self.reward_tracker.compute(gs, terminal, victory)
        self.episode_reward += reward

        if self.prev_action is not None:
            total_reward = self.pending_reward + reward
            self.buffer.add(
                obs=self.prev_obs, action=self.prev_action,
                reward=total_reward, done=terminal,
                mask=self.prev_mask, log_prob=self.prev_log_prob,
                value=self.prev_value,
            )
            self.prev_action = None
            self.pending_reward = 0.0
        else:
            self.pending_reward += reward

        if terminal:
            self._end_game()
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            return Action("proceed") if proceed_avail else Action("state")

        auto = self._auto_handle_screen(gs, screen_name)
        if auto is not None:
            self._stuck_count = 0
            self.total_steps += 1
            return auto

        cur_floor = int(getattr(gs, "floor", -1) or -1)
        if cur_floor == self._stuck_floor:
            self._stuck_count += 1
        else:
            self._stuck_floor = cur_floor
            self._stuck_count = 0

        if self._stuck_count >= 30:
            self._stuck_count = 0
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            choice_list = list(getattr(gs, "choice_list", []) or [])
            if proceed_avail:
                return Action("proceed")
            if choice_list:
                return ChooseAction(choice_index=0)
            return Action("proceed")

        action, log_prob, value = self.trainer.predict(obs, mask)
        spire_action = flat_action_to_spire_action(action, gs)

        self.prev_obs = obs
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.prev_mask = mask
        self.total_steps += 1

        return spire_action

    def _auto_handle_screen(self, gs, screen_name: str) -> Optional[Action]:
        """Heuristic handles ALL non-combat screens — same as train_ppo.py."""
        in_combat = bool(getattr(gs, "in_combat", False))
        choice_list = list(getattr(gs, "choice_list", []) or [])
        proceed_avail = bool(getattr(gs, "proceed_available", False))
        cancel_avail = bool(getattr(gs, "cancel_available", False))
        scr = getattr(gs, "screen", None)

        if in_combat and screen_name == "NONE":
            return None

        if screen_name == "HAND_SELECT":
            can_pick_zero = scr and getattr(scr, "can_pick_zero", False)
            if can_pick_zero and proceed_avail:
                return Action("proceed")
            return ChooseAction(choice_index=0)

        if screen_name == "GRID":
            if scr and getattr(scr, "confirm_up", False):
                return Action("proceed")
            if proceed_avail:
                return Action("proceed")
            return ChooseAction(choice_index=0)

        if screen_name == "CHEST":
            if scr and getattr(scr, "chest_open", False):
                return Action("proceed") if proceed_avail else Action("state")
            return ChooseAction(name="open")

        if screen_name in ("SHOP_ROOM", "SHOP_SCREEN"):
            if screen_name == "SHOP_SCREEN":
                return Action("leave")
            return Action("proceed") if proceed_avail else Action("leave") if cancel_avail else Action("proceed")

        if screen_name == "MAP":
            boss_avail = scr and getattr(scr, "boss_available", False)
            if boss_avail and not choice_list:
                return ChooseAction(name="boss")
            if choice_list:
                hp = int(getattr(gs, "current_hp", 0) or 0)
                mhp = max(1, int(getattr(gs, "max_hp", 1) or 1))
                n = len(choice_list)
                idx = 0 if hp / mhp < 0.45 else min(n // 2, n - 1)
                return ChooseAction(choice_index=idx)
            if boss_avail:
                return ChooseAction(name="boss")
            return Action("proceed") if proceed_avail else Action("state")

        if screen_name == "BOSS_REWARD":
            return ChooseAction(choice_index=0) if choice_list else Action("proceed")

        if screen_name == "CARD_REWARD":
            if choice_list:
                good = {"inflame", "shrug it off", "anger", "uppercut",
                        "offering", "battle trance", "headbutt"}
                ok = {"cleave", "thunderclap", "iron wave", "body slam"}
                lower = [c.lower() for c in choice_list]
                for i, c in enumerate(lower):
                    if c in good:
                        return ChooseAction(choice_index=i)
                for i, c in enumerate(lower):
                    if c in ok:
                        return ChooseAction(choice_index=i)
                if "skip" in lower:
                    return ChooseAction(choice_index=lower.index("skip"))
                return ChooseAction(choice_index=0)
            return Action("proceed") if proceed_avail else Action("state")

        if screen_name == "COMBAT_REWARD":
            if choice_list:
                for p in ["relic", "gold", "potion", "card"]:
                    for i, c in enumerate(choice_list):
                        if p in str(c).lower():
                            return ChooseAction(choice_index=i)
                return ChooseAction(choice_index=0)
            return Action("proceed") if proceed_avail else Action("state")

        if screen_name == "REST":
            if choice_list:
                lower = [str(c).lower() for c in choice_list]
                hp_pct = int(getattr(gs, "current_hp", 0) or 0) / max(1, int(getattr(gs, "max_hp", 1) or 1))
                if hp_pct < 0.6 and "rest" in lower:
                    return ChooseAction(choice_index=lower.index("rest"))
                if "smith" in lower:
                    return ChooseAction(choice_index=lower.index("smith"))
                return ChooseAction(choice_index=0)
            return Action("proceed") if proceed_avail else Action("state")

        if screen_name == "EVENT":
            if choice_list:
                return ChooseAction(choice_index=0)
            return Action("proceed") if proceed_avail else Action("state")

        if not in_combat:
            if choice_list:
                return ChooseAction(choice_index=0)
            if proceed_avail:
                return Action("proceed")
            if cancel_avail:
                return Action("leave")
            return Action("state")

        if choice_list:
            return ChooseAction(choice_index=0)
        if proceed_avail:
            return Action("proceed")

        return None

    def _end_game(self):
        self.total_games += 1
        n = len(self.buffer)
        log(f"Game #{self.total_games} ended: {n} transitions, reward={self.episode_reward:.2f}")

        if n >= 5:
            fname = f"w{self.worker_id}_g{self.total_games}_{int(time.time())}.npz"
            path = os.path.join(self.out_dir, fname)
            self.buffer.save_npz(path)
            log(f"  Saved {n} transitions to {fname}")

        if self.total_games % self.reload_every == 0:
            self._maybe_reload_model()

        self.buffer.clear()
        self.episode_reward = 0.0
        self.pending_reward = 0.0
        self.prev_obs = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.prev_mask = None
        self._stuck_floor = -1
        self._stuck_count = 0
        self.initialized = False

    def on_out_of_game(self) -> Action:
        log(f"OUT OF GAME (games: {self.total_games})")
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        return Action("state")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_sts.pt",
                        help="Path to model checkpoint to load")
    parser.add_argument("--out", type=str, default="rollouts_shared",
                        help="Directory for shared transition files")
    parser.add_argument("--id", type=str, default="1",
                        help="Worker ID (for logging and filenames)")
    parser.add_argument("--reload-every", type=int, default=5,
                        help="Reload model every N games")
    args = parser.parse_args()

    _init_log(args.id)
    log("=== ROLLOUT WORKER STARTING ===")

    model_path = os.path.join(_root, args.model)
    out_dir = os.path.join(_root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    trainer = PPOTrainer(
        obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
        net_arch=(256, 256),
    )

    if os.path.isfile(model_path):
        trainer.load(model_path)
        log(f"Loaded model from {model_path}")
    else:
        log(f"No model at {model_path}, starting with random weights")

    agent = WorkerAgent(
        trainer=trainer, model_path=model_path,
        out_dir=out_dir, worker_id=args.id,
        reload_every=args.reload_every,
    )

    coord = Coordinator()
    coord.register_state_change_callback(agent.on_state_change)
    coord.register_out_of_game_callback(agent.on_out_of_game)
    coord.register_command_error_callback(agent.on_error)

    log("Signaling ready...")
    coord.signal_ready()
    coord.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
