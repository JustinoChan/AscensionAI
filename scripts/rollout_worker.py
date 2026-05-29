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
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

import spirecomm.communication.coordinator as _coord_module

def _patched_write_stdout(output_queue):
    while True:
        output = output_queue.get()
        _real_stdout.write(output + '\n')
        _real_stdout.flush()

_coord_module.write_stdout = _patched_write_stdout

def _patched_read_stdin(input_queue):
    """Detect stdin EOF (STS died) and exit instead of spinning forever."""
    while True:
        stdin_input = ""
        while True:
            ch = sys.stdin.read(1)
            if ch == '':
                try:
                    _real_stdout.write("stdin EOF — STS process died, exiting\n")
                    _real_stdout.flush()
                except Exception:
                    pass
                os._exit(1)
            if ch == '\n':
                break
            stdin_input += ch
        input_queue.put(stdin_input)
_coord_module.read_stdin = _patched_read_stdin

_scripts = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_scripts)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import threading
import traceback
import time
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import torch

_worker_id = "?"
DEBUG_LOG = None
HEARTBEAT_FILE = None
_last_heartbeat = 0.0
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

def _init_log(worker_id: str):
    global DEBUG_LOG, HEARTBEAT_FILE, _worker_id
    _worker_id = worker_id
    os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
    DEBUG_LOG = os.path.join(_root, "logs", f"worker_{worker_id}_debug.log")
    HEARTBEAT_FILE = os.path.join(_root, "logs", f"worker_{worker_id}_heartbeat.txt")

def log(msg: str):
    if DEBUG_LOG is None:
        return
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  [W{_worker_id}] {msg}\n")
            f.flush()
    except Exception:
        pass


def heartbeat(gs) -> None:
    """Write a lightweight liveness marker for the GUI health monitor."""
    global _last_heartbeat
    if HEARTBEAT_FILE is None:
        return
    now = time.time()
    if now - _last_heartbeat < 15.0:
        return
    _last_heartbeat = now
    try:
        st = getattr(gs, "screen_type", None)
        screen_name = str(getattr(st, "name", st) or "NONE")
        line = (
            f"{now:.3f}\t{datetime.now().isoformat()}\t"
            f"worker={_worker_id}\tscreen={screen_name}\t"
            f"floor={getattr(gs, 'floor', '?')}\t"
            f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')}\n"
        )
        tmp = HEARTBEAT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(line)
        os.replace(tmp, HEARTBEAT_FILE)
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
    is_terminal_state, is_victory_state,
)
from ppo_model import PPOTrainer
from screen_handler import (
    auto_handle_screen,
    event_choice_targets,
    pick_event_slot_and_choice,
    recover_from_command_error,
)
from fight_tracker import FightTracker

BUG_DEBUG_LOG = os.path.join(_root, "logs", "bug_debug.log")

def _dump_stuck_state(gs, screen_name: str, worker_id: str, stuck_count: int,
                      recent_actions: list):
    """Write a detailed game state snapshot to bug_debug.log for debugging."""
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
            f"  Worker: {worker_id}",
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

# Shared training stats CSV (same format the GUI reads for progress display)
_STATS_CSV = os.path.join(_root, "logs", "training_stats.csv")
from training_stats_schema import (
    TRAINING_STATS_COLUMNS as _STATS_COLUMNS,
    append_training_stats_csv,
    ensure_training_stats_csv,
)

def _init_stats_csv():
    ensure_training_stats_csv(_STATS_CSV, log_fn=log)

def _append_training_stats(row: dict):
    append_training_stats_csv(_STATS_CSV, row, log_fn=log)


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

    def save_npz(self, path: str, metadata: dict | None = None):
        tmp = path.replace(".npz", ".tmp.npz")
        payload = {
            "observations": np.array(self.observations, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.bool_),
            "action_masks": np.array(self.action_masks, dtype=np.bool_),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
        }
        for key, value in (metadata or {}).items():
            payload[key] = np.array(value)
        np.savez_compressed(
            tmp,
            **payload,
        )
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Worker agent (same hybrid approach: heuristic non-combat, RL combat)
# ---------------------------------------------------------------------------
class WorkerAgent:
    def __init__(self, trainer: PPOTrainer, model_path: str, out_dir: str,
                 worker_id: str, reload_every: int = 5, max_games: int = 0,
                 restart_every: int = 0):
        self.trainer = trainer
        self.model_path = model_path
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.reload_every = reload_every
        self.max_games = max(0, int(max_games))
        self.restart_every = max(0, int(restart_every))

        self.buffer = TransitionBuffer()
        self.reward_tracker = RewardTracker()
        self.total_games = 0
        self.total_steps = 0
        self._game_start_steps = 0
        self.episode_reward = 0.0
        self.initialized = False
        self._stop_requested = False
        self._exit_scheduled = False

        self.prev_obs = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.prev_mask = None
        self.pending_reward = 0.0

        self._stuck_key = ""
        self._stuck_count = 0
        self._stuck_dumped_key = ""
        self._progress_key = ""
        self._no_progress_count = 0
        self._progress_dumped_key = ""
        self._recent_actions: list[str] = []
        self._model_mtime = 0.0
        if os.path.isfile(self.model_path):
            try:
                self._model_mtime = os.path.getmtime(self.model_path)
            except OSError:
                self._model_mtime = 0.0

        self.fight_tracker = FightTracker(
            source="worker", worker=worker_id, log=log
        )

    def _done_marker_path(self) -> str:
        return os.path.join(_root, "logs", f"worker_{self.worker_id}_done.txt")

    def _write_done_marker(self) -> None:
        path = self._done_marker_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().isoformat()}\tworker={self.worker_id}\t"
                    f"games={self.total_games}\ttarget={self.max_games}\n"
                )
            os.replace(tmp, path)
            log(f"Wrote done marker: {path}")
        except Exception as e:
            log(f"Failed to write done marker: {e}")

    def _schedule_exit(self, delay: float = 2.0) -> None:
        if self._exit_scheduled:
            return
        self._exit_scheduled = True
        log(f"Scheduling worker exit in {delay:.1f}s after game target")
        threading.Timer(delay, lambda: os._exit(0)).start()

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

    _STUCK_HARD_LIMIT = 50
    _NO_PROGRESS_HARD_LIMIT = 35

    def _track_stuck(self, screen_name: str, action_cmd: str, in_combat: bool):
        key = f"{screen_name}:{action_cmd}"
        if key == self._stuck_key:
            self._stuck_count += 1
        else:
            self._stuck_key = key
            self._stuck_count = 1
        if in_combat:
            self._stuck_count = 0

    def _state_progress_key(self, gs, screen_name: str) -> str:
        """Fingerprint non-combat state to catch loops where actions vary but nothing changes."""
        scr = getattr(gs, "screen", None)
        choice_list = tuple(str(c) for c in (getattr(gs, "choice_list", []) or []))
        selected = tuple(
            str(getattr(c, "name", c))
            for c in (getattr(scr, "selected_cards", []) or [])
        ) if scr is not None else ()
        cards = tuple(
            str(getattr(c, "name", c))
            for c in (getattr(scr, "cards", []) or [])[:8]
        ) if scr is not None else ()
        commands = tuple(sorted(str(c) for c in (getattr(gs, "available_commands", []) or [])))
        event_sig = ()
        if screen_name == "EVENT" and scr is not None:
            options = tuple(
                (
                    str(getattr(opt, "label", getattr(opt, "text", opt))),
                    bool(getattr(opt, "disabled", False)),
                    str(getattr(opt, "choice_index", "")),
                )
                for opt in (getattr(scr, "options", []) or [])[:8]
            )
            event_sig = (
                str(getattr(scr, "event_id", "")),
                str(getattr(scr, "event_name", "")),
                str(getattr(scr, "body_text", ""))[:200],
                options,
            )
        return repr((
            screen_name,
            int(getattr(gs, "act", 0) or 0),
            int(getattr(gs, "floor", 0) or 0),
            int(getattr(gs, "current_hp", 0) or 0),
            int(getattr(gs, "max_hp", 0) or 0),
            bool(getattr(gs, "proceed_available", False)),
            bool(getattr(gs, "cancel_available", False)),
            commands,
            choice_list,
            event_sig,
            bool(getattr(scr, "confirm_up", False)) if scr is not None else False,
            int(getattr(scr, "num_cards", 0) or 0) if scr is not None else 0,
            selected,
            cards,
        ))

    def _track_state_progress(self, gs, screen_name: str, in_combat: bool) -> int:
        if in_combat:
            self._progress_key = ""
            self._no_progress_count = 0
            return 0
        key = self._state_progress_key(gs, screen_name)
        if key == self._progress_key:
            self._no_progress_count += 1
        else:
            self._progress_key = key
            self._no_progress_count = 1
        return self._no_progress_count

    def _force_progress_action(self, gs, screen_name: str, why: str) -> Action:
        commands = set(getattr(gs, "available_commands", []) or [])
        choice_list = list(getattr(gs, "choice_list", []) or [])
        log(f"{why} on {screen_name}; forcing progress "
            f"commands={sorted(commands)} choices={choice_list[:5]}")

        event_targets = event_choice_targets(gs) if screen_name == "EVENT" else {}
        if event_targets:
            _slot, choice_idx = pick_event_slot_and_choice(choice_list, gs)
            return ChooseAction(choice_index=choice_idx)

        # GRID/forge confirmation frequently exposes only the `confirm` command.
        if "confirm" in commands:
            return Action("confirm")
        if bool(getattr(gs, "proceed_available", False)) or "proceed" in commands:
            return Action("proceed")

        if "choose" in commands or choice_list:
            return ChooseAction(choice_index=0)

        for cmd in ("cancel", "leave", "return", "skip"):
            if cmd in commands:
                return Action(cmd)
        if bool(getattr(gs, "cancel_available", False)):
            return Action("leave")
        return Action("state")

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
        terminal = is_terminal_state(gs)
        heartbeat(gs)

        if VERBOSE:
            choice_list = list(getattr(gs, "choice_list", []) or [])
            scr = getattr(gs, "screen", None)
            confirm_up = getattr(scr, "confirm_up", None)
            potions = list(getattr(gs, "potions", []) or [])
            pot_ids = [getattr(p, "potion_id", "?") for p in potions]
            pot_full = bool(getattr(gs, "are_potions_full", lambda: False)())
            log(f"STEP screen={screen_name} floor={getattr(gs, 'floor', '?')} "
                f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')} "
                f"choices={choice_list} proceed={getattr(gs, 'proceed_available', False)} "
                f"cancel={getattr(gs, 'cancel_available', False)} "
                f"confirm_up={confirm_up} potions={pot_ids} pot_full={pot_full} "
                f"mask_sum={int(mask.sum())}")

        in_combat = bool(getattr(gs, "in_combat", False))
        victory = is_victory_state(gs)
        self.fight_tracker.observe(
            gs, game=self.total_games + 1,
            terminal=terminal, victory=victory,
        )

        if not self.initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.initialized = True
            self.prev_obs = obs
            self.prev_mask = mask
            self._game_start_steps = self.total_steps
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
            self._end_game(final_gs=gs, victory=victory)
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            action = Action("proceed") if proceed_avail else Action("state")
            if self._stop_requested:
                self._write_done_marker()
                self._schedule_exit()
            return action

        no_progress = self._track_state_progress(gs, screen_name, in_combat)
        if no_progress >= 12 and self._progress_key != self._progress_dumped_key:
            _dump_stuck_state(gs, screen_name, self.worker_id,
                              no_progress, self._recent_actions)
            log(f"NO PROGRESS on {screen_name} — dumped to bug_debug.log")
            self._progress_dumped_key = self._progress_key
        if no_progress >= self._NO_PROGRESS_HARD_LIMIT:
            self._no_progress_count = 0
            return self._force_progress_action(gs, screen_name, "HARD NO-PROGRESS")

        auto = self._auto_handle_screen(gs, screen_name)
        if auto is not None:
            if VERBOSE:
                log(f"  -> HEURISTIC: {auto.command}")
            self._track_stuck(screen_name, auto.command, in_combat=False)
            self.total_steps += 1
            if self._stuck_count >= 10:
                if self._stuck_key != self._stuck_dumped_key:
                    _dump_stuck_state(gs, screen_name, self.worker_id,
                                      self._stuck_count, self._recent_actions)
                    log(f"STUCK (heuristic) on {screen_name} — dumped to bug_debug.log")
                    self._stuck_dumped_key = self._stuck_key
                proceed_avail = bool(getattr(gs, "proceed_available", False))
                cancel_avail = bool(getattr(gs, "cancel_available", False))
                choice_list = list(getattr(gs, "choice_list", []) or [])
                if screen_name == "EVENT" and event_choice_targets(gs):
                    _slot, choice_idx = pick_event_slot_and_choice(choice_list, gs)
                    return ChooseAction(choice_index=choice_idx)
                if proceed_avail:
                    return Action("proceed")
                if cancel_avail:
                    return Action("leave")
                if self._stuck_count >= self._STUCK_HARD_LIMIT:
                    self._stuck_count = 0
                    return self._force_progress_action(gs, screen_name, "HARD STUCK heuristic")
            return auto

        action, log_prob, value = self.trainer.predict(obs, mask)
        spire_action = flat_action_to_spire_action(action, gs)

        self._track_stuck(screen_name, spire_action.command,
                          in_combat=bool(getattr(gs, "in_combat", False)))
        self._recent_actions.append(f"{screen_name}:{spire_action.command}")
        if len(self._recent_actions) > 20:
            self._recent_actions = self._recent_actions[-20:]

        if self._stuck_count >= self._STUCK_HARD_LIMIT:
            if self._stuck_key != self._stuck_dumped_key:
                _dump_stuck_state(gs, screen_name, self.worker_id,
                                  self._stuck_count, self._recent_actions)
                self._stuck_dumped_key = self._stuck_key
            self._stuck_count = 0
            return self._force_progress_action(gs, screen_name, "HARD STUCK RL")

        if VERBOSE:
            log(f"  -> RL: action={action} ({spire_action.command}) "
                f"lp={log_prob:.3f} v={value:.3f}")

        self.prev_obs = obs
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.prev_mask = mask
        self.total_steps += 1

        return spire_action

    def _auto_handle_screen(self, gs, screen_name: str) -> Optional[Action]:
        return auto_handle_screen(gs, screen_name, heuristic_all=False)

    def _end_game(self, final_gs=None, victory: bool = False):
        self.total_games += 1
        n = len(self.buffer)
        steps_this_game = max(0, self.total_steps - self._game_start_steps)
        fight_stats = self.fight_tracker.finish_game(
            final_gs, game=self.total_games, victory=victory
        )
        log(f"Game #{self.total_games} ended: {n} transitions, "
            f"{steps_this_game} steps, reward={self.episode_reward:.2f}")

        try:
            if n >= 5:
                fname = f"w{self.worker_id}_g{self.total_games}_{int(time.time())}.npz"
                path = os.path.join(self.out_dir, fname)
                try:
                    lr = 0.0
                    try:
                        lr = float(self.trainer.optimizer.param_groups[0].get("lr", 0.0))
                    except Exception:
                        pass
                    model_update = int(getattr(self.trainer, "total_updates", 0) or 0)
                    checkpoint_mtime = self._model_mtime
                    if checkpoint_mtime <= 0 and os.path.isfile(self.model_path):
                        checkpoint_mtime = os.path.getmtime(self.model_path)
                    metadata = {
                        "model_update_number": model_update,
                        "checkpoint_id": f"u{model_update}_m{int(checkpoint_mtime)}",
                        "worker_id": self.worker_id,
                        "episode_number": self.total_games,
                        "entropy_coeff": float(getattr(self.trainer, "ent_coef", 0.0)),
                        "learning_rate": lr,
                        "created_at": datetime.now().isoformat(),
                    }
                    self.buffer.save_npz(path, metadata=metadata)
                    log(f"  Saved {n} transitions to {fname} "
                        f"(model_update={model_update}, checkpoint={metadata['checkpoint_id']})")
                except Exception as e:
                    log(f"save_npz failed (non-fatal): {e}")

            _append_training_stats({
                "timestamp": datetime.now().isoformat(),
                "game": self.total_games,
                "worker": self.worker_id,
                "steps": steps_this_game,
                "transitions": n,
                "total_reward": round(self.episode_reward, 4),
                "final_hp": int(getattr(final_gs, "current_hp", 0) or 0) if final_gs else "",
                "final_max_hp": int(getattr(final_gs, "max_hp", 0) or 0) if final_gs else "",
                "final_floor": int(getattr(final_gs, "floor", 0) or 0) if final_gs else "",
                "final_act": int(getattr(final_gs, "act", 0) or 0) if final_gs else "",
                "victory": int(bool(victory)),
                "terminated": 1,
                **fight_stats,
            })

            if fight_stats["elites_fought"] or fight_stats["bosses_fought"]:
                log(f"  Elites: {fight_stats['elites_won']}/"
                    f"{fight_stats['elites_fought']}  "
                    f"Bosses: {fight_stats['bosses_won']}/"
                    f"{fight_stats['bosses_fought']}")

            if self.total_games % self.reload_every == 0:
                self._maybe_reload_model()
        except Exception as e:
            log(f"_end_game error (non-fatal): {e}")
        finally:
            self.buffer.clear()
            self.episode_reward = 0.0
            self.pending_reward = 0.0
            self.prev_obs = None
            self.prev_action = None
            self.prev_log_prob = None
            self.prev_value = None
            self.prev_mask = None
            self._stuck_key = ""
            self._stuck_count = 0
            self._stuck_dumped_key = ""
            self._progress_key = ""
            self._no_progress_count = 0
            self._progress_dumped_key = ""
            self.fight_tracker.reset_game()
            self._recent_actions.clear()
            self.initialized = False
            if self.max_games > 0 and self.total_games >= self.max_games:
                self._stop_requested = True
                log(f"Game target reached: {self.total_games}/{self.max_games}")
            elif self.restart_every > 0 and self.total_games >= self.restart_every:
                log(f"Restart-every threshold reached: {self.total_games}/{self.restart_every} — exiting for RAM cleanup")
                self._schedule_exit()

    def on_out_of_game(self) -> Action:
        log(f"OUT OF GAME (games: {self.total_games})")
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
    parser.add_argument("--model", type=str, default="models/ppo_sts.pt",
                        help="Path to model checkpoint to load")
    parser.add_argument("--out", type=str, default="rollouts_shared",
                        help="Directory for shared transition files")
    parser.add_argument("--id", type=str, default="1",
                        help="Worker ID (for logging and filenames)")
    parser.add_argument("--reload-every", type=int, default=5,
                        help="Reload model every N games")
    parser.add_argument("--games", type=int, default=0,
                        help="Stop after this many completed games (0 = unlimited)")
    parser.add_argument("--restart-every", type=int, default=0,
                        help="Exit after this many games to free RAM; GUI relaunches (0 = disabled)")
    parser.add_argument("--net-arch", type=str, default="512,256,256",
                        help="Comma-separated hidden layer sizes (default: 512,256,256)")
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["tanh", "gelu", "relu"],
                        help="Activation function for shared layers (default: gelu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Write detailed per-state/per-action debug logs")
    args = parser.parse_args()
    VERBOSE = VERBOSE or args.verbose

    _init_log(args.id)
    net_arch = tuple(int(x) for x in args.net_arch.split(","))
    log("=== ROLLOUT WORKER STARTING ===")
    log(f"Config: model={args.model} out={args.out} id={args.id} "
        f"reload_every={args.reload_every} max_games={args.games} "
        f"restart_every={args.restart_every} net_arch={net_arch} "
        f"activation={args.activation} verbose={VERBOSE}")

    model_path = os.path.join(_root, args.model)
    out_dir = os.path.join(_root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    trainer = PPOTrainer(
        obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
        net_arch=net_arch, activation=args.activation,
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
        max_games=args.games,
        restart_every=args.restart_every,
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
