"""
eval_model.py — Greedy evaluation harness for a trained PPO model.

Runs as a CommunicationMod client: plays games back-to-back, picks the
highest-probability legal action on every combat step (no exploration,
no gradient updates), and records per-game stats to logs/eval_stats.csv.
Non-combat decision screens use the trained model; only mechanical
screens (chest, grid confirm, etc.) are auto-handled.

Usage (point CommunicationMod's command= at this script):

    command=.../python.exe .../scripts/eval_model.py --model models/ppo_sts.pt --games 30

When --games is reached, the process exits cleanly after the current run
finishes. Summary line is written to eval_debug.log and printed as the
final stdout line.
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
        _real_stdout.write(output + "\n")
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
from datetime import datetime
from typing import Any, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, ChooseAction, StartGameAction
from spirecomm.spire.character import PlayerClass

from obs_encoder import OBS_SIZE, encode_game_state
from sts_gym_env import (
    NUM_ACTIONS, compute_action_mask, flat_action_to_spire_action,
    RewardTracker,
)
from ppo_model import PPOTrainer
from screen_handler import auto_handle_screen


os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "eval_debug.log")
EVAL_CSV = os.path.join(_root, "logs", "eval_stats.csv")

_EVAL_COLUMNS = [
    "timestamp", "run", "game", "steps", "total_reward",
    "final_hp", "final_max_hp", "final_floor", "final_act", "victory",
]


def log(msg: str) -> None:
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
    except Exception:
        pass


def _init_csv() -> None:
    os.makedirs(os.path.dirname(EVAL_CSV), exist_ok=True)
    if not os.path.exists(EVAL_CSV):
        with open(EVAL_CSV, "w", encoding="utf-8") as f:
            f.write(",".join(_EVAL_COLUMNS) + "\n")


def _append_csv(row: dict) -> None:
    try:
        with open(EVAL_CSV, "a", encoding="utf-8") as f:
            f.write(",".join(str(row.get(c, "")) for c in _EVAL_COLUMNS) + "\n")
    except Exception as e:
        log(f"csv append failed: {e}")


log("=== EVAL STARTING ===")
_init_csv()


def auto_handle(gs: Any, screen_name: str) -> Optional[Action]:
    return auto_handle_screen(gs, screen_name, heuristic_all=False)


# ---------------------------------------------------------------------------
# Evaluator agent
# ---------------------------------------------------------------------------
class EvalAgent:
    def __init__(self, trainer: PPOTrainer, target_games: int, run_tag: str):
        self.trainer = trainer
        self.target_games = target_games
        self.run_tag = run_tag
        self.reward_tracker = RewardTracker()
        self.games_played = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.initialized = False

        self.wins = 0
        self.sum_floor = 0
        self.sum_reward = 0.0

    def on_state_change(self, gs) -> Action:
        try:
            return self._handle(gs)
        except Exception as e:
            log(f"ERROR on_state_change: {e}")
            log(traceback.format_exc())
            return Action("state")

    def _handle(self, gs) -> Action:
        st = getattr(gs, "screen_type", None)
        screen_name = str(getattr(st, "name", st) or "NONE")
        terminal = screen_name in {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}

        victory = False
        if terminal:
            scr = getattr(gs, "screen", None)
            victory = bool(getattr(scr, "victory", False)) or screen_name in {"COMPLETE", "VICTORY"}

        if not self.initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.initialized = True
            log(f"Eval game #{self.games_played + 1} starting, floor={getattr(gs, 'floor', '?')}")

        reward = self.reward_tracker.compute(gs, terminal, victory)
        self.episode_reward += reward

        if terminal:
            self._end_game(gs, victory)
            if self.games_played >= self.target_games:
                log("target games reached — requesting exit")
                summary = (
                    f"EVAL COMPLETE: {self.games_played} games, "
                    f"wins={self.wins} ({self.wins / max(1, self.games_played):.1%}), "
                    f"avg_floor={self.sum_floor / max(1, self.games_played):.2f}, "
                    f"avg_reward={self.sum_reward / max(1, self.games_played):.2f}"
                )
                log(summary)
                print(summary, file=sys.stderr)
                os._exit(0)
            if bool(getattr(gs, "proceed_available", False)):
                return Action("proceed")
            if bool(getattr(gs, "cancel_available", False)):
                return Action("leave")
            return Action("state")

        auto = auto_handle(gs, screen_name)
        if auto is not None:
            self.total_steps += 1
            return auto

        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)
        action, _lp, _v = self.trainer.predict(obs, mask, deterministic=True)
        self.total_steps += 1
        return flat_action_to_spire_action(action, gs)

    def _end_game(self, final_gs, victory: bool) -> None:
        self.games_played += 1
        floor = int(getattr(final_gs, "floor", 0) or 0)
        self.sum_floor += floor
        self.sum_reward += self.episode_reward
        if victory:
            self.wins += 1

        _append_csv({
            "timestamp": datetime.now().isoformat(),
            "run": self.run_tag,
            "game": self.games_played,
            "steps": self.total_steps,
            "total_reward": round(self.episode_reward, 4),
            "final_hp": int(getattr(final_gs, "current_hp", 0) or 0),
            "final_max_hp": int(getattr(final_gs, "max_hp", 0) or 0),
            "final_floor": floor,
            "final_act": int(getattr(final_gs, "act", 0) or 0),
            "victory": int(bool(victory)),
        })
        log(
            f"Game #{self.games_played}: floor={floor} "
            f"hp={getattr(final_gs, 'current_hp', '?')} victory={victory} "
            f"reward={self.episode_reward:.2f}"
        )

        self.episode_reward = 0.0
        self.initialized = False
        self.total_steps = 0

    def on_out_of_game(self) -> Action:
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        if "proceed" in err and "choose" in err:
            return ChooseAction(choice_index=0)
        return Action("state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_sts.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--run-tag", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    model_path = os.path.join(_root, args.model) if not os.path.isabs(args.model) else args.model
    log(f"Loading model from {model_path} for {args.games} greedy games")

    trainer = PPOTrainer(
        obs_size=OBS_SIZE,
        n_actions=NUM_ACTIONS,
        device="cpu",
        net_arch=(256, 256),
    )
    if os.path.exists(model_path):
        trainer.load(model_path)
        log(f"Loaded checkpoint (total_updates={trainer.total_updates})")
    else:
        log(f"WARNING: no checkpoint at {model_path} — evaluating randomly-initialized policy")

    agent = EvalAgent(trainer, target_games=args.games, run_tag=args.run_tag)

    coord = Coordinator()
    coord.register_state_change_callback(agent.on_state_change)
    coord.register_out_of_game_callback(agent.on_out_of_game)
    coord.register_command_error_callback(agent.on_error)
    coord.signal_ready()
    coord.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        raise
