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
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

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

os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "train_debug.log")
BUG_DEBUG_LOG = os.path.join(_root, "logs", "bug_debug.log")
STATS_CSV = os.path.join(_root, "logs", "training_stats.csv")
from training_stats_schema import (
    TRAINING_STATS_COLUMNS as _STATS_COLUMNS,
    append_training_stats_csv,
    ensure_training_stats_csv,
)

def log(msg: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
            f.flush()
    except Exception:
        pass

def _init_stats_csv() -> None:
    ensure_training_stats_csv(STATS_CSV, log_fn=log)

def _append_stats_csv(row: dict) -> None:
    append_training_stats_csv(STATS_CSV, row, log_fn=log)


def _dump_stuck_state(gs, screen_name: str, stuck_count: int,
                      recent_actions: list):
    """Write a detailed game state snapshot to bug_debug.log."""
    try:
        scr = getattr(gs, "screen", None)
        rewards = list(getattr(scr, "rewards", []) or []) if scr else []
        hand = list(getattr(gs, "hand", []) or [])
        monsters = list(getattr(gs, "monsters", []) or [])
        potions = list(getattr(gs, "potions", []) or [])
        choice_list = list(getattr(gs, "choice_list", []) or [])

        lines = [
            f"\n{'='*70}",
            f"STUCK DETECTED — {datetime.now().isoformat()}",
            f"  Worker: single-instance",
            f"  Screen: {screen_name}  Floor: {getattr(gs, 'floor', '?')}  "
            f"Act: {getattr(gs, 'act', '?')}",
            f"  HP: {getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')}  "
            f"Gold: {getattr(gs, 'gold', '?')}  "
            f"Energy: {getattr(gs, 'player', None) and getattr(gs.player, 'energy', '?')}",
            f"  In Combat: {getattr(gs, 'in_combat', False)}  "
            f"Stuck Count: {stuck_count}",
            f"  proceed_available: {getattr(gs, 'proceed_available', False)}  "
            f"cancel_available: {getattr(gs, 'cancel_available', False)}",
            f"  choice_list ({len(choice_list)}): "
            f"{[str(c) for c in choice_list[:10]]}",
        ]
        if rewards:
            reward_strs = [getattr(r, "reward_type", r) for r in rewards]
            lines.append(f"  screen.rewards ({len(rewards)}): {reward_strs}")
        if scr:
            lines.append(f"  screen attrs: confirm_up={getattr(scr, 'confirm_up', None)} "
                         f"can_pick_zero={getattr(scr, 'can_pick_zero', None)}")
            if screen_name == "EVENT":
                options = list(getattr(scr, "options", []) or [])
                opt_info = [
                    (getattr(o, "label", None), getattr(o, "text", None),
                     getattr(o, "disabled", None), getattr(o, "choice_index", None))
                    for o in options[:10]
                ]
                lines.append(f"  event: name={getattr(scr, 'event_name', None)} "
                             f"id={getattr(scr, 'event_id', None)} "
                             f"body={getattr(scr, 'body_text', None)!r}")
                lines.append(f"  event options: {opt_info}")
        if hand:
            card_info = [(getattr(c, "name", "?"), getattr(c, "is_playable", "?"),
                          getattr(c, "cost", "?")) for c in hand[:10]]
            lines.append(f"  Hand ({len(hand)}): {card_info}")
        if monsters:
            m_info = [(getattr(m, "name", "?"), getattr(m, "current_hp", "?"),
                        getattr(m, "is_gone", False)) for m in monsters]
            lines.append(f"  Monsters: {m_info}")
        pot_info = [(getattr(p, "potion_id", "?")) for p in potions]
        lines.append(f"  Potions: {pot_info}  "
                     f"Full: {bool(getattr(gs, 'are_potions_full', lambda: False)())}")
        if recent_actions:
            lines.append(f"  Recent actions: {recent_actions[-10:]}")
        lines.append(f"{'='*70}\n")

        with open(BUG_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        log(f"bug_debug dump failed: {e}")


log("=== SCRIPT STARTING ===")
_init_stats_csv()

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, ChooseAction, StartGameAction
from spirecomm.spire.character import PlayerClass

from obs_encoder import OBS_SIZE, encode_game_state
from sts_gym_env import (
    NUM_ACTIONS, compute_action_mask, flat_action_to_spire_action,
    RewardTracker, _NOOP, is_terminal_state, is_victory_state,
)
from ppo_model import PPOTrainer, GameBuffer
from screen_handler import (
    auto_handle_screen,
    event_choice_targets,
    pick_event_slot_and_choice,
    recover_from_command_error,
)
from fight_tracker import FightTracker

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
        self._game_start_steps = 0
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.prev_log_prob: Optional[float] = None
        self.prev_value: Optional[float] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.episode_reward = 0.0
        self.pending_reward = 0.0  # accumulates reward during auto-handled steps
        self.initialized = False

        self._stuck_key: str = ""
        self._stuck_count: int = 0
        self._recent_actions: list[str] = []
        self.fight_tracker = FightTracker(source="train", worker="single", log=log)


    def _track_stuck(self, screen_name: str, action_cmd: str, in_combat: bool):
        key = f"{screen_name}:{action_cmd}"
        if key == self._stuck_key:
            self._stuck_count += 1
        else:
            self._stuck_key = key
            self._stuck_count = 1
        if in_combat:
            self._stuck_count = 0

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
        terminated = is_terminal_state(gs)
        victory = is_victory_state(gs)
        self.fight_tracker.observe(
            gs, game=self.total_games + 1,
            terminal=terminated, victory=victory,
        )

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

        # First state of the game
        if not self.initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.initialized = True
            self.prev_obs = obs
            self.prev_mask = mask
            self._game_start_buf_len = len(self.buffer)
            self._game_start_steps = self.total_steps
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
            self._track_stuck(screen_name, auto.command, in_combat=False)
            self.total_steps += 1
            if self._stuck_count >= 10:
                _dump_stuck_state(gs, screen_name, self._stuck_count,
                                  self._recent_actions)
                log(f"  STUCK (heuristic) on {screen_name} — dumped to bug_debug.log")
                self._stuck_count = 0
                proceed_avail = bool(getattr(gs, "proceed_available", False))
                cancel_avail = bool(getattr(gs, "cancel_available", False))
                if screen_name == "EVENT":
                    choice_list = list(getattr(gs, "choice_list", []) or [])
                    if event_choice_targets(gs):
                        _slot, choice_idx = pick_event_slot_and_choice(choice_list, gs)
                        return ChooseAction(choice_index=choice_idx)
                if proceed_avail:
                    return Action("proceed")
                if cancel_avail:
                    return Action("leave")
            if self.total_steps % 5 == 1:
                log(f"  step={self.total_steps} floor={getattr(gs, 'floor', '?')} "
                    f"screen={screen_name} AUTO→{auto.command if hasattr(auto, 'command') else type(auto).__name__}")
            return auto  # reward will accumulate in pending_reward

        # Pick next action via RL
        action, log_prob, value = self.trainer.predict(obs, mask)
        spire_action = flat_action_to_spire_action(action, gs)

        self._track_stuck(screen_name, spire_action.command,
                          in_combat=bool(getattr(gs, "in_combat", False)))
        self._recent_actions.append(f"{screen_name}:{spire_action.command}")
        if len(self._recent_actions) > 20:
            self._recent_actions = self._recent_actions[-20:]

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
        return auto_handle_screen(gs, screen_name, heuristic_all=False)

    def _end_game(self, final_gs=None, victory: bool = False):
        self.total_games += 1
        self.games_since_update += 1
        fight_stats = self.fight_tracker.finish_game(
            final_gs, game=self.total_games, victory=victory
        )
        n_this_game = len(self.buffer) - self._game_start_buf_len
        steps_this_game = max(0, self.total_steps - self._game_start_steps)
        n_buffered = len(self.buffer)
        log(f"Game #{self.total_games} ended: {n_this_game} transitions this game "
            f"{steps_this_game} steps this game "
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
                    f"ent={stats['entropy']:.4f} "
                    f"norm_ent={stats.get('normalized_entropy', 0.0):.3f} "
                    f"kl={stats.get('approx_kl', 0.0):.5f} "
                    f"clip={stats.get('clip_fraction', 0.0):.3f} "
                    f"ev={stats.get('explained_variance', 0.0):.3f} "
                    f"early_stop={stats.get('early_stop', 0)} "
                    f"transitions={n_buffered}")
            self.buffer.clear()
            self.games_since_update = 0

        row = {
            "timestamp": datetime.now().isoformat(),
            "game": self.total_games,
            "total_updates": self.trainer.total_updates,
            "steps": steps_this_game,
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
            "normalized_entropy": round(stats.get("normalized_entropy", 0.0), 6) if stats else "",
            "lr": f"{self.trainer.get_lr():.8g}",
            "ent_coef": f"{float(getattr(self.trainer, 'ent_coef', 0.0) or 0.0):.8g}",
            "elites_fought": fight_stats["elites_fought"],
            "elites_won": fight_stats["elites_won"],
            "bosses_fought": fight_stats["bosses_fought"],
            "bosses_won": fight_stats["bosses_won"],
            "approx_kl": round(stats.get("approx_kl", 0.0), 8) if stats else "",
            "clip_fraction": round(stats.get("clip_fraction", 0.0), 6) if stats else "",
            "explained_variance": round(stats.get("explained_variance", 0.0), 6) if stats else "",
            "mean_advantage": round(stats.get("mean_advantage", 0.0), 6) if stats else "",
            "std_advantage": round(stats.get("std_advantage", 0.0), 6) if stats else "",
            "invalid_action_count": stats.get("invalid_action_count", "") if stats else "",
            "mean_chosen_action_prob": round(stats.get("mean_chosen_action_prob", 0.0), 6) if stats else "",
            "bc_loss": round(stats.get("bc_loss", 0.0), 6) if stats else "",
            "bc_coef": round(stats.get("bc_coef", 0.0), 6) if stats else "",
            "early_stop": stats.get("early_stop", "") if stats else "",
        }
        _append_stats_csv(row)
        if fight_stats["elites_fought"] or fight_stats["bosses_fought"]:
            log(f"  Elites: {fight_stats['elites_won']}/"
                f"{fight_stats['elites_fought']}  "
                f"Bosses: {fight_stats['bosses_won']}/"
                f"{fight_stats['bosses_fought']}")

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
        self._stuck_key = ""
        self._stuck_count = 0
        self._recent_actions.clear()
        self.fight_tracker.reset_game()
        self.prev_mask = None
        self.initialized = False

    def on_out_of_game(self) -> Action:
        log(f"OUT OF GAME (games so far: {self.total_games})")
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        return recover_from_command_error(err)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="models/ppo_sts.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save model every N games")
    parser.add_argument("--games-per-update", type=int, default=4,
                        help="Accumulate N games of transitions per PPO update (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="PPO learning rate (default: 1e-4)")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="Entropy bonus coefficient (higher = more exploration)")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Stop PPO epochs early when approx KL exceeds this value")
    parser.add_argument("--verbose", action="store_true",
                        help="Write detailed per-state/per-action debug logs")
    args = parser.parse_args()
    VERBOSE = VERBOSE or args.verbose

    save_path = os.path.join(_root, args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    log("Creating PPO trainer...")
    log(f"Config: save={args.save} resume={args.resume} "
        f"games_per_update={args.games_per_update} lr={args.lr} ent_coef={args.ent_coef} "
        f"target_kl={args.target_kl} verbose={VERBOSE}")
    trainer = PPOTrainer(
        obs_size=OBS_SIZE,
        n_actions=NUM_ACTIONS,
        device="cpu",
        lr=args.lr,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        n_epochs=4,
        batch_size=64,
        net_arch=(256, 256),
        target_kl=args.target_kl,
    )

    if args.resume:
        resume_path = os.path.join(_root, args.resume)
        trainer.load(resume_path)
        trainer.set_lr(args.lr)
        log(f"Loaded checkpoint {resume_path}; optimizer lr reset to {args.lr}")

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
