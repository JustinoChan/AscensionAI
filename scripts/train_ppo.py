"""
train_ppo.py — Train a PPO policy on Slay the Spire.

Uses the Coordinator callback pattern instead of a Gym env.  The neural
network chooses actions inside the state callback, transitions are
collected per-game, and PPO updates happen between games.

Communication Mod uses stdin/stdout — all logging goes to file/stderr.
"""

from __future__ import annotations

import sys
import os

# ---- stdout belongs to Communication Mod ----
_real_stdout = sys.stdout
sys.stdout = sys.stderr

import spirecomm.communication.coordinator as _coord_module

def _patched_write_stdout(output_queue):
    while True:
        output = output_queue.get()
        _real_stdout.write(output + '\n')
        _real_stdout.flush()

_coord_module.write_stdout = _patched_write_stdout
# ---- End stdout fix ----

_scripts = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_scripts)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

import argparse
import traceback
import time
from datetime import datetime
from typing import Any, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

DEBUG_LOG = os.path.join(_root, "train_debug.log")
STATS_CSV = os.path.join(_root, "logs", "training_stats.csv")
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

_STATS_COLUMNS = [
    "timestamp", "game", "total_updates", "steps", "transitions",
    "total_reward", "final_hp", "final_max_hp", "final_floor", "final_act",
    "victory", "terminated", "pg_loss", "vf_loss", "entropy",
]


def log(msg: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
            f.flush()
    except Exception:
        pass


def _init_stats_csv() -> None:
    try:
        os.makedirs(os.path.dirname(STATS_CSV), exist_ok=True)
        if not os.path.exists(STATS_CSV):
            with open(STATS_CSV, "w", encoding="utf-8") as f:
                f.write(",".join(_STATS_COLUMNS) + "\n")
    except Exception as e:
        log(f"stats csv init failed: {e}")


def _append_stats_csv(row: dict) -> None:
    try:
        with open(STATS_CSV, "a", encoding="utf-8") as f:
            f.write(",".join(str(row.get(c, "")) for c in _STATS_COLUMNS) + "\n")
    except Exception as e:
        log(f"stats csv append failed: {e}")


log("=== SCRIPT STARTING ===")
_init_stats_csv()

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, ChooseAction, StartGameAction
from spirecomm.spire.character import PlayerClass

from obs_encoder import OBS_SIZE, encode_game_state
from sts_gym_env import (
    NUM_ACTIONS, compute_action_mask, flat_action_to_spire_action,
    RewardTracker, _NOOP,
)
from ppo_model import PPOTrainer, GameBuffer

log("Imports done")


# ---------------------------------------------------------------------------
# Game-playing agent with PPO brain
# ---------------------------------------------------------------------------
class PPOAgent:
    """Plays STS using PPO policy, collects transitions, trains between games."""

    def __init__(self, trainer: PPOTrainer, save_path: str, save_every: int = 5,
                 games_per_update: int = 4):
        self.trainer = trainer
        self.save_path = save_path
        self.save_every = save_every
        self.games_per_update = games_per_update

        self.buffer = GameBuffer()
        self.reward_tracker = RewardTracker()

        self.total_games = 0
        self.total_steps = 0
        self.games_since_update = 0
        self._game_start_buf_len = 0
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.prev_log_prob: Optional[float] = None
        self.prev_value: Optional[float] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.episode_reward = 0.0
        self.pending_reward = 0.0  # accumulates reward during auto-handled steps
        self.initialized = False

        self._stuck_screen: Optional[str] = None
        self._stuck_floor: int = -1
        self._stuck_count: int = 0


    def on_state_change(self, game_state) -> Action:
        try:
            return self._handle_state(game_state)
        except Exception as e:
            log(f"ERROR in on_state_change: {e}")
            log(traceback.format_exc())
            return Action("state")

    def _handle_state(self, gs) -> Action:
        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)

        screen_type = getattr(gs, "screen_type", None)
        screen_name = str(getattr(screen_type, "name", screen_type) or "NONE")
        terminal_screens = {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}
        terminated = screen_name in terminal_screens

        if VERBOSE:
            choice_list_v = list(getattr(gs, "choice_list", []) or [])
            scr_v = getattr(gs, "screen", None)
            confirm_up_v = getattr(scr_v, "confirm_up", None)
            potions_v = list(getattr(gs, "potions", []) or [])
            pot_ids_v = [getattr(p, "potion_id", "?") for p in potions_v]
            pot_full_v = bool(getattr(gs, "are_potions_full", lambda: False)())
            log(f"VERBOSE screen={screen_name} floor={getattr(gs, 'floor', '?')} "
                f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')} "
                f"choices={choice_list_v} proceed={getattr(gs, 'proceed_available', False)} "
                f"cancel={getattr(gs, 'cancel_available', False)} "
                f"confirm_up={confirm_up_v} potions={pot_ids_v} pot_full={pot_full_v} "
                f"mask_sum={int(mask.sum())}")

        victory = False
        if terminated:
            scr_obj = getattr(gs, "screen", None)
            victory = bool(getattr(scr_obj, "victory", False)) or screen_name in {"COMPLETE", "VICTORY"}

        # First state of the game
        if not self.initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.initialized = True
            self.prev_obs = obs
            self.prev_mask = mask
            self._game_start_buf_len = len(self.buffer)
            log(f"Game #{self.total_games + 1} started, floor={getattr(gs, 'floor', '?')}")

        # Compute reward for the previous action
        reward = self.reward_tracker.compute(gs, terminated, victory)
        self.episode_reward += reward

        # Store transition for the PREVIOUS RL action, crediting any
        # accumulated reward from auto-handled steps plus this step's reward
        if self.prev_action is not None:
            total_reward = self.pending_reward + reward
            self.buffer.add(
                obs=self.prev_obs,
                action=self.prev_action,
                reward=total_reward,
                done=terminated,
                mask=self.prev_mask,
                log_prob=self.prev_log_prob,
                value=self.prev_value,
            )
            self.prev_action = None  # stored — don't double-count
            self.pending_reward = 0.0
        else:
            # No RL action to credit — bank the reward for the next one
            self.pending_reward += reward

        # If game is over, train and dismiss the screen
        if terminated:
            self._end_game(gs, victory)
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            cancel_avail = bool(getattr(gs, "cancel_available", False))
            if proceed_avail:
                return Action("proceed")
            elif cancel_avail:
                return Action("leave")
            return Action("state")

        # ---- Auto-handle "mechanical" screens that don't need RL ----
        auto = self._auto_handle_screen(gs, screen_name)
        if auto is not None:
            self._stuck_count = 0
            self.total_steps += 1
            if self.total_steps % 5 == 1:
                log(f"  step={self.total_steps} floor={getattr(gs, 'floor', '?')} "
                    f"screen={screen_name} AUTO→{auto.command if hasattr(auto, 'command') else type(auto).__name__}")
            return auto  # reward will accumulate in pending_reward

        # ---- Stuck detection: same floor too many steps → force action ----
        cur_floor = int(getattr(gs, "floor", -1) or -1)
        if cur_floor == self._stuck_floor:
            self._stuck_count += 1
        else:
            self._stuck_floor = cur_floor
            self._stuck_count = 0
        self._stuck_screen = screen_name

        if self._stuck_count >= 30:
            self._stuck_count = 0
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            cancel_avail = bool(getattr(gs, "cancel_available", False))
            choice_list = list(getattr(gs, "choice_list", []) or [])
            log(f"  STUCK on {screen_name} floor={cur_floor}, forcing action "
                f"(proceed={proceed_avail} cancel={cancel_avail} choices={len(choice_list)})")
            if proceed_avail:
                return Action("proceed")
            if cancel_avail:
                return Action("leave")
            if choice_list:
                return ChooseAction(choice_index=0)
            return Action("proceed")

        # Pick next action via RL
        action, log_prob, value = self.trainer.predict(obs, mask)
        spire_action = flat_action_to_spire_action(action, gs)

        self.prev_obs = obs
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.prev_mask = mask
        self.total_steps += 1

        n_legal = int(mask.sum())
        floor = getattr(gs, "floor", "?")
        hp = getattr(gs, "current_hp", "?")
        in_combat = getattr(gs, "in_combat", False)
        action_str = str(spire_action.command) if hasattr(spire_action, "command") else str(type(spire_action).__name__)
        if self.total_steps % 5 == 1 or self.total_steps <= 3:
            log(f"  step={self.total_steps} floor={floor} hp={hp} screen={screen_name} "
                f"combat={in_combat} legal={n_legal} action={action}→{action_str} r={reward:.3f}")

        return spire_action

    def _auto_handle_screen(self, gs, screen_name: str) -> Optional[Action]:
        """Handle only mechanical screens; decision screens fall through to RL.

        Mechanical = no meaningful choice (the optimal action is always the same).
        Decision = multiple options where the choice affects game outcome.
        Returns an Action for mechanical screens, or None so RL decides.
        """
        in_combat = bool(getattr(gs, "in_combat", False))
        choice_list = list(getattr(gs, "choice_list", []) or [])
        proceed_avail = bool(getattr(gs, "proceed_available", False))
        cancel_avail = bool(getattr(gs, "cancel_available", False))
        scr = getattr(gs, "screen", None)

        # In combat with normal action state → RL handles it
        if in_combat and screen_name == "NONE":
            return None

        # --- Mechanical: CHEST — always open, then proceed ---
        if screen_name == "CHEST":
            if scr and getattr(scr, "chest_open", False):
                return Action("proceed") if proceed_avail else Action("state")
            return ChooseAction(name="open")

        # --- Mechanical: HAND_SELECT with can_pick_zero — just skip ---
        if screen_name == "HAND_SELECT":
            if scr and getattr(scr, "can_pick_zero", False) and proceed_avail:
                return Action("proceed")
            return None  # forced selection (discard etc.) → RL decides

        # --- Mechanical: GRID confirmation — just confirm ---
        if screen_name == "GRID":
            if scr and getattr(scr, "confirm_up", False):
                return Action("proceed")
            if proceed_avail and not choice_list:
                return Action("proceed")
            return None  # card selection (upgrade/transform/purge) → RL decides

        # --- Mechanical: MAP boss-only (no path choices) ---
        if screen_name == "MAP":
            boss_avail = scr and getattr(scr, "boss_available", False)
            if boss_avail and not choice_list:
                return ChooseAction(name="boss")
            return None  # path selection → RL decides

        # --- Decision screens → RL handles ---
        # CARD_REWARD, COMBAT_REWARD, REST, EVENT, SHOP_ROOM,
        # SHOP_SCREEN, BOSS_REWARD, and any others with choices

        # Trivial: no choices and only one possible navigation action
        if not choice_list and not cancel_avail:
            if proceed_avail:
                return Action("proceed")
            return Action("state")  # nothing available, poll
        if not choice_list and cancel_avail and not proceed_avail:
            return Action("leave")

        return None  # choices available → RL decides

    def _end_game(self, final_gs=None, victory: bool = False):
        self.total_games += 1
        self.games_since_update += 1
        n_this_game = len(self.buffer) - self._game_start_buf_len
        n_buffered = len(self.buffer)
        log(f"Game #{self.total_games} ended: {n_this_game} transitions this game "
            f"({n_buffered} buffered, {self.games_since_update}/{self.games_per_update}), "
            f"reward={self.episode_reward:.2f}")

        stats: dict = {}
        do_update = (self.games_since_update >= self.games_per_update
                     and n_buffered >= 10)
        if do_update:
            stats = self.trainer.update(self.buffer) or {}
            if stats:
                log(f"  PPO update #{self.trainer.total_updates}: "
                    f"pg={stats['pg_loss']:.4f} vf={stats['vf_loss']:.4f} "
                    f"ent={stats['entropy']:.4f} transitions={n_buffered}")
            self.buffer.clear()
            self.games_since_update = 0

        row = {
            "timestamp": datetime.now().isoformat(),
            "game": self.total_games,
            "total_updates": self.trainer.total_updates,
            "steps": self.total_steps,
            "transitions": n_this_game,
            "total_reward": round(self.episode_reward, 4),
            "final_hp": int(getattr(final_gs, "current_hp", 0) or 0) if final_gs is not None else "",
            "final_max_hp": int(getattr(final_gs, "max_hp", 0) or 0) if final_gs is not None else "",
            "final_floor": int(getattr(final_gs, "floor", 0) or 0) if final_gs is not None else "",
            "final_act": int(getattr(final_gs, "act", 0) or 0) if final_gs is not None else "",
            "victory": int(bool(victory)),
            "terminated": 1,
            "pg_loss": round(stats.get("pg_loss", 0.0), 6) if stats else "",
            "vf_loss": round(stats.get("vf_loss", 0.0), 6) if stats else "",
            "entropy": round(stats.get("entropy", 0.0), 6) if stats else "",
        }
        _append_stats_csv(row)

        if self.total_games % self.save_every == 0:
            self.trainer.save(self.save_path)

        # Reset for next game (buffer persists across games until the update
        # threshold is hit above, then cleared there)
        self.episode_reward = 0.0
        self.pending_reward = 0.0
        self.prev_obs = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self._stuck_screen = None
        self._stuck_floor = -1
        self._stuck_count = 0
        self.prev_mask = None
        self.initialized = False

    def on_out_of_game(self) -> Action:
        log(f"OUT OF GAME (games so far: {self.total_games})")
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        return Action("state")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="models/ppo_sts.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save model every N games")
    parser.add_argument("--games-per-update", type=int, default=4,
                        help="Accumulate N games of transitions per PPO update (default: 4)")
    args = parser.parse_args()

    save_path = os.path.join(_root, args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    log("Creating PPO trainer...")
    trainer = PPOTrainer(
        obs_size=OBS_SIZE,
        n_actions=NUM_ACTIONS,
        device="cpu",
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        n_epochs=4,
        batch_size=64,
        net_arch=(256, 256),
    )

    if args.resume:
        resume_path = os.path.join(_root, args.resume)
        trainer.load(resume_path)

    log("Creating agent...")
    agent = PPOAgent(trainer, save_path, save_every=args.save_every,
                     games_per_update=args.games_per_update)

    log("Setting up Coordinator...")
    coord = Coordinator()
    coord.register_state_change_callback(agent.on_state_change)
    coord.register_out_of_game_callback(agent.on_out_of_game)
    coord.register_command_error_callback(agent.on_error)

    log("Signaling ready and starting game loop")
    coord.signal_ready()
    coord.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
