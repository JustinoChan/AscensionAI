"""
train_bc_ppo.py — End-to-end BC warm-start then PPO fine-tuning.

Two-phase training in a single Communication Mod session:
  Phase 1 (BC):  Heuristic plays --bc-games games; records demonstrations
                 and trains supervised cross-entropy to imitate.
  Phase 2 (PPO): Fine-tune with PPO on the full RL decision surface,
                 with entropy annealing and a conservative learning rate
                 to preserve BC knowledge.

Designed to be the final integration step: the agent starts with a
reasonable BC policy and improves it through online RL experience.

Usage (in CommunicationMod config.properties):
  command=python scripts/train_bc_ppo.py --bc-games 150 --ppo-games 200

To resume PPO training from an existing checkpoint (skip BC):
  command=python scripts/train_bc_ppo.py --resume models/ppo_sts.pt
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
from datetime import datetime
from typing import Any, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "train_bc_ppo_debug.log")
STATS_CSV = os.path.join(_root, "logs", "training_stats.csv")
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

# Shared training stats CSV (same file the GUI reads for progress display)
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


def _action_desc(action: Action) -> str:
    command = str(getattr(action, "command", type(action).__name__))
    parts = [command]
    for attr in ("choice_index", "name"):
        if hasattr(action, attr):
            parts.append(f"{attr}={getattr(action, attr)}")
    return " ".join(parts)


def _state_desc(gs, screen: str | None = None) -> str:
    screen = screen or _screen_name(gs)
    choice_list = list(getattr(gs, "choice_list", []) or [])
    scr = getattr(gs, "screen", None)
    selected = list(getattr(scr, "selected_cards", []) or []) if scr else []
    selected_names = [str(getattr(c, "name", c)) for c in selected[:6]]
    flags = []
    for flag in ("proceed_available", "cancel_available", "play_available", "end_available"):
        if bool(getattr(gs, flag, False)):
            flags.append(flag.replace("_available", ""))
    details = [
        f"screen={screen}",
        f"floor={getattr(gs, 'floor', '?')}",
        f"act={getattr(gs, 'act', '?')}",
        f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')}",
        f"gold={getattr(gs, 'gold', '?')}",
        f"combat={bool(getattr(gs, 'in_combat', False))}",
        f"legal_flags={'+'.join(flags) if flags else 'none'}",
        f"choices={len(choice_list)}",
    ]
    current_action = str(getattr(gs, "current_action", "") or "")
    if current_action:
        details.append(f"current_action={current_action!r}")
    if choice_list:
        details.append(f"choice_list={[str(c) for c in choice_list[:8]]}")
    if selected_names:
        details.append(f"selected={selected_names}")
    if scr is not None and screen in {"GRID", "HAND_SELECT"}:
        details.append(
            "grid_flags="
            f"num={getattr(scr, 'num_cards', None)} "
            f"any={getattr(scr, 'any_number', None)} "
            f"confirm={getattr(scr, 'confirm_up', None)} "
            f"upgrade={getattr(scr, 'for_upgrade', None)} "
            f"transform={getattr(scr, 'for_transform', None)} "
            f"purge={getattr(scr, 'for_purge', None)}"
        )
    if scr is not None and screen == "EVENT":
        opts = []
        for opt in list(getattr(scr, "options", []) or [])[:8]:
            opts.append({
                "label": getattr(opt, "label", None),
                "disabled": getattr(opt, "disabled", None),
                "choice_index": getattr(opt, "choice_index", None),
            })
        details.append(f"event={getattr(scr, 'event_id', None)} options={opts}")
    return " ".join(details)


def _screen_name(gs) -> str:
    st = getattr(gs, "screen_type", None)
    name = getattr(st, "name", st) if st is not None else "NONE"
    return str(name) if name else "NONE"


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


def _tmp_npz_path(path: str) -> str:
    return path.replace(".npz", ".tmp.npz") if path.endswith(".npz") else path + ".tmp.npz"


def _remove_if_exists(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
            log(f"Removed BC progress checkpoint: {path}")
    except OSError as e:
        log(f"Failed to remove BC progress checkpoint {path}: {e}")


log("=== BC->PPO PIPELINE STARTING ===")
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
from screen_handler import auto_handle_screen, recover_from_command_error
from behavior_clone import heuristic_action
from bc_stats import append_bc_stats
from fight_tracker import FightTracker

# Re-patch after behavior_clone import (its module-level code overwrites
# _coord_module.write_stdout with a version that uses stderr)
_coord_module.write_stdout = _patched_write_stdout

log(f"Imports done (verbose={'on' if VERBOSE else 'off'})")


# ---------------------------------------------------------------------------
# Two-phase agent: BC collection -> supervised training -> PPO fine-tuning
# ---------------------------------------------------------------------------
class BCPPOAgent:
    """Plays STS in two phases: BC demonstration collection, then PPO training.

    Phase 1 (BC):
        The heuristic agent plays games while we record (obs, action, mask)
        tuples. After enough games, we train supervised cross-entropy on the
        demos to warm-start the network.

    Phase 2 (PPO):
        The warm-started policy plays games, collecting transitions per-game.
        PPO updates happen between games with:
        - Lower LR than default (preserves BC knowledge)
        - Tighter clip range (prevents catastrophic forgetting)
        - Entropy annealing (starts exploratory, converges to exploitation)
    """

    PHASE_BC = "bc"
    PHASE_PPO = "ppo"

    def __init__(
        self,
        bc_games: int,
        ppo_games: int,
        save_path: str,
        save_every: int = 5,
        bc_epochs: int = 30,
        bc_lr: float = 1e-3,
        ppo_lr: float = 1e-4,
        ent_start: float = 0.001,
        ent_end: float = 0.001,
        clip_range: float = 0.15,
        target_kl: float = 0.03,
        resume_path: Optional[str] = None,
        games_per_update: int = 4,
        bc_anchor_coef: float = 0.02,
        demo_save_path: Optional[str] = None,
        bc_checkpoint_path: Optional[str] = None,
        resume_bc: bool = True,
    ):
        self.bc_games = bc_games
        self.ppo_games = ppo_games
        self.save_path = save_path
        self.save_every = save_every
        self.bc_epochs = bc_epochs
        self.bc_lr = bc_lr
        self.ppo_lr = ppo_lr
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.clip_range = clip_range
        self.target_kl = target_kl
        self.games_per_update = games_per_update
        self.bc_anchor_coef = bc_anchor_coef
        self.demo_save_path = demo_save_path or save_path.replace(".pt", "_bc_demos.npz")
        self.bc_checkpoint_path = bc_checkpoint_path or save_path.replace(".pt", "_bc_progress.npz")
        self.resume_bc = resume_bc
        self.games_since_update = 0
        self._game_start_buf_len = 0

        # --- Phase control ---
        self.phase = self.PHASE_PPO if resume_path else self.PHASE_BC

        # --- BC state ---
        self.demo_obs: List[np.ndarray] = []
        self.demo_actions: List[int] = []
        self.demo_masks: List[np.ndarray] = []
        self.bc_games_done = 0
        self.bc_steps = 0
        self._bc_game_start_steps = 0
        self._bc_game_start_demo_len = 0
        self._bc_game_start_skipped = 0
        self.bc_skipped_samples = 0
        self.bc_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bc_initialized = False
        self.total_games = 0

        # --- PPO trainer (created after BC, or loaded from checkpoint) ---
        if resume_path:
            self.trainer = PPOTrainer(
                obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
                lr=ppo_lr, gamma=0.995, gae_lambda=0.95,
                clip_range=clip_range, ent_coef=ent_start, vf_coef=0.5,
                n_epochs=4, batch_size=64, net_arch=(256, 256),
                target_kl=target_kl,
            )
            self.trainer.load(resume_path)
            self.trainer.set_lr(ppo_lr)
            log(f"Resumed from {resume_path} (updates={self.trainer.total_updates}, "
                f"optimizer lr reset to {ppo_lr})")
            self._try_load_bc_anchor()
        else:
            self.trainer: Optional[PPOTrainer] = None

        # --- PPO per-game state ---
        self.buffer = GameBuffer()
        self.reward_tracker = RewardTracker()
        self.ppo_games_done = 0
        self.ppo_steps = 0
        self._ppo_game_start_steps = 0
        self.episode_reward = 0.0
        self.pending_reward = 0.0
        self.ppo_initialized = False
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.prev_log_prob: Optional[float] = None
        self.prev_value: Optional[float] = None
        self.prev_mask: Optional[np.ndarray] = None
        self._stuck_floor: int = -1
        self._stuck_count: int = 0
        self._last_issue_key: str = ""
        tracker_source = "bc_ppo" if self.phase == self.PHASE_PPO else "bc"
        self.fight_tracker = FightTracker(
            source=tracker_source, worker="bc_ppo", log=log
        )
        if self.phase == self.PHASE_BC and self.resume_bc:
            self._try_load_bc_progress()

    # ------------------------------------------------------------------
    # Coordinator callbacks
    # ------------------------------------------------------------------
    def on_state_change(self, gs) -> Action:
        try:
            if self.phase == self.PHASE_BC:
                return self._bc_step(gs)
            return self._ppo_step(gs)
        except Exception as e:
            log(f"ERROR in on_state_change: {e}")
            log(traceback.format_exc())
            return Action("state")

    def on_out_of_game(self) -> Action:
        if self.phase == self.PHASE_BC and self.bc_games_done >= self.bc_games:
            log("BC progress already complete; transitioning to PPO before next game")
            self._transition_to_ppo()

        if (self.phase == self.PHASE_PPO
                and self.ppo_games > 0
                and self.ppo_games_done >= self.ppo_games):
            log(f"PPO training complete ({self.ppo_games_done} games). Final save.")
            if self.trainer:
                self.trainer.save(self.save_path)
            raise StopIteration()

        log(f"OUT OF GAME (phase={self.phase}, "
            f"bc={self.bc_games_done}/{self.bc_games}, "
            f"ppo={self.ppo_games_done})")
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        log(f"COMMAND ERROR CONTEXT: phase={self.phase} "
            f"bc_steps={self.bc_steps} ppo_steps={self.ppo_steps}")
        return recover_from_command_error(err)

    def _save_bc_progress(self) -> None:
        if not self.bc_checkpoint_path:
            return
        try:
            os.makedirs(os.path.dirname(self.bc_checkpoint_path) or ".", exist_ok=True)
            tmp = _tmp_npz_path(self.bc_checkpoint_path)
            np.savez_compressed(
                tmp,
                observations=np.array(self.demo_obs, dtype=np.float32),
                actions=np.array(self.demo_actions, dtype=np.int64),
                action_masks=np.array(self.demo_masks, dtype=np.bool_),
                games_done=np.array(self.bc_games_done, dtype=np.int64),
                total_games=np.array(self.total_games, dtype=np.int64),
                bc_steps=np.array(self.bc_steps, dtype=np.int64),
                skipped_samples=np.array(self.bc_skipped_samples, dtype=np.int64),
                target_games=np.array(self.bc_games, dtype=np.int64),
                run_id=np.array(self.bc_run_id),
                saved_at=np.array(datetime.now().isoformat()),
            )
            os.replace(tmp, self.bc_checkpoint_path)
            log(f"BC progress checkpoint saved: games={self.bc_games_done}/{self.bc_games} "
                f"samples={len(self.demo_actions)} path={self.bc_checkpoint_path}")
        except Exception as e:
            log(f"Failed to save BC progress checkpoint {self.bc_checkpoint_path}: {e}")

    def _try_load_bc_progress(self) -> None:
        if not self.bc_checkpoint_path or not os.path.isfile(self.bc_checkpoint_path):
            return
        try:
            with np.load(self.bc_checkpoint_path, allow_pickle=False) as data:
                self.demo_obs = [x for x in data["observations"]]
                self.demo_actions = [int(x) for x in data["actions"]]
                self.demo_masks = [x for x in data["action_masks"]]
                self.bc_games_done = int(data["games_done"].item())
                self.total_games = int(data["total_games"].item()) if "total_games" in data.files else self.bc_games_done
                self.bc_steps = int(data["bc_steps"].item())
                self.bc_skipped_samples = int(data["skipped_samples"].item())
                if "run_id" in data.files:
                    self.bc_run_id = str(data["run_id"].item())
            self.bc_games_done = min(self.bc_games_done, self.bc_games)
            self.total_games = max(self.total_games, self.bc_games_done)
            log(f"Resumed BC progress checkpoint: games={self.bc_games_done}/{self.bc_games} "
                f"samples={len(self.demo_actions)} skipped={self.bc_skipped_samples} "
                f"path={self.bc_checkpoint_path}")
        except Exception as e:
            log(f"Failed to load BC progress checkpoint {self.bc_checkpoint_path}: {e}")

    # ------------------------------------------------------------------
    # Phase 1: BC demonstration collection
    # ------------------------------------------------------------------
    def _bc_step(self, gs) -> Action:
        try:
            return self._bc_step_inner(gs)
        except Exception:
            log(f"CRASH in _bc_step: {traceback.format_exc()}")
            return Action("state")

    def _bc_step_inner(self, gs) -> Action:
        screen = self._screen_name(gs)
        terminal = is_terminal_state(gs)
        victory = is_victory_state(gs)
        self.fight_tracker.observe(
            gs, game=self.total_games + 1,
            terminal=terminal, victory=victory,
        )

        if not self.bc_initialized:
            self.bc_initialized = True
            self._bc_game_start_steps = self.bc_steps
            self._bc_game_start_demo_len = len(self.demo_obs)
            self._bc_game_start_skipped = self.bc_skipped_samples
            log(f"BC game #{self.bc_games_done + 1}/{self.bc_games} started, "
                f"floor={getattr(gs, 'floor', '?')}")

        if terminal:
            self.bc_games_done += 1
            self.total_games += 1
            self.bc_initialized = False

            fight_stats = self.fight_tracker.finish_game(
                gs, game=self.total_games, victory=victory
            )
            steps_this_game = max(0, self.bc_steps - self._bc_game_start_steps)
            demos_this_game = max(0, len(self.demo_obs) - self._bc_game_start_demo_len)
            skipped_this_game = max(0, self.bc_skipped_samples - self._bc_game_start_skipped)

            log(f"BC game #{self.bc_games_done} ended: "
                f"floor={getattr(gs, 'floor', '?')} victory={victory} "
                f"samples={demos_this_game} total_samples={len(self.demo_obs)} "
                f"skipped={self.bc_skipped_samples}")

            append_bc_stats({
                "timestamp": datetime.now().isoformat(),
                "run_id": self.bc_run_id,
                "source": "bc_ppo",
                "game": self.total_games,
                "target_games": self.bc_games,
                "steps": steps_this_game,
                "samples": demos_this_game,
                "skipped_samples": skipped_this_game,
                "final_hp": int(getattr(gs, "current_hp", 0) or 0),
                "final_max_hp": int(getattr(gs, "max_hp", 0) or 0),
                "final_floor": int(getattr(gs, "floor", 0) or 0),
                "final_act": int(getattr(gs, "act", 0) or 0),
                "victory": int(victory),
                "terminated": 1,
                "elites_fought": fight_stats["elites_fought"],
                "elites_won": fight_stats["elites_won"],
                "bosses_fought": fight_stats["bosses_fought"],
                "bosses_won": fight_stats["bosses_won"],
                "checkpoint_path": self.bc_checkpoint_path,
                "model_path": self.save_path.replace(".pt", "_bc.pt"),
            }, log=log)

            self._save_bc_progress()

            # Collected enough demos — train and switch to PPO
            if self.bc_games_done >= self.bc_games:
                self._transition_to_ppo()

            if getattr(gs, "proceed_available", False):
                return Action("proceed")
            return Action("state")

        # --- Heuristic plays, we record (obs, action, mask) ---
        spire_action, action_id = heuristic_action(gs)
        if spire_action is None or action_id is None:
            self.bc_steps += 1
            self.bc_skipped_samples += 1
            log(f"BC ISSUE no heuristic action: step={self.bc_steps} "
                f"skipped={self.bc_skipped_samples} {_state_desc(gs, screen)}")
            return Action("state")

        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)

        # Only record if the heuristic's action is legal under our mask
        in_range = 0 <= int(action_id) < NUM_ACTIONS
        legal = bool(mask[action_id]) if in_range else False
        if legal:
            self.demo_obs.append(obs)
            self.demo_actions.append(action_id)
            self.demo_masks.append(mask)
        else:
            self.bc_skipped_samples += 1
            log(f"BC ISSUE skipped illegal demo sample: step={self.bc_steps + 1} "
                f"action_id={action_id} in_range={in_range} "
                f"legal_count={int(mask.sum())} action={_action_desc(spire_action)} "
                f"{_state_desc(gs, screen)}")

        self.bc_steps += 1
        if VERBOSE:
            log(f"BC STEP {self.bc_steps}: game={self.bc_games_done + 1}/{self.bc_games} "
                f"samples={len(self.demo_obs)} skipped={self.bc_skipped_samples} "
                f"recorded={int(legal)} action_id={action_id} "
                f"legal_count={int(mask.sum())} action={_action_desc(spire_action)} "
                f"{_state_desc(gs, screen)}")
        elif self.bc_steps % 100 == 0:
            log(f"  BC step={self.bc_steps} samples={len(self.demo_obs)} "
                f"skipped={self.bc_skipped_samples} "
                f"games={self.bc_games_done}/{self.bc_games}")

        return spire_action

    # ------------------------------------------------------------------
    # Transition: supervised training on BC demos, then switch to PPO
    # ------------------------------------------------------------------
    def _transition_to_ppo(self):
        n = len(self.demo_obs)
        log(f"=== TRANSITION: BC -> PPO ({n} demos from {self.bc_games_done} games, "
            f"{self.bc_skipped_samples} skipped samples) ===")

        # Create trainer with BC learning rate for supervised phase
        self.trainer = PPOTrainer(
            obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
            lr=self.bc_lr, gamma=0.995, gae_lambda=0.95,
            clip_range=self.clip_range, ent_coef=self.ent_start, vf_coef=0.5,
            n_epochs=4, batch_size=64, net_arch=(256, 256),
            target_kl=self.target_kl,
        )

        if n >= 50:
            self._train_supervised(n)
        else:
            log(f"Only {n} demos — skipping BC training, starting PPO from scratch")

        # Replace optimizer with PPO fine-tuning LR (lower to preserve BC)
        self.trainer.optimizer = torch.optim.Adam(
            list(self.trainer.shared.parameters())
            + list(self.trainer.policy_head.parameters())
            + list(self.trainer.value_head.parameters()),
            lr=self.ppo_lr,
        )
        self.trainer.ent_coef = self.ent_start

        # Save BC checkpoint as a separate file
        bc_path = self.save_path.replace(".pt", "_bc.pt")
        self.trainer.save(bc_path)
        log(f"BC checkpoint saved to {bc_path}")

        if n >= 50:
            self._save_bc_demo_dataset()
            if self.bc_anchor_coef > 0.0:
                self.trainer.set_bc_reference(
                    self.demo_obs, self.demo_actions, self.demo_masks,
                    coef=self.bc_anchor_coef,
                    batch_size=64,
                )
                log(f"BC anchor enabled for PPO: coef={self.bc_anchor_coef} "
                    f"samples={len(self.demo_actions)}")
            _remove_if_exists(self.bc_checkpoint_path)

        # Free demo data
        self.demo_obs.clear()
        self.demo_actions.clear()
        self.demo_masks.clear()

        self.phase = self.PHASE_PPO
        self.fight_tracker.source = "bc_ppo"
        log(f"=== PPO PHASE STARTED (lr={self.ppo_lr}, ent={self.ent_start}, "
            f"clip={self.clip_range}) ===")

    def _save_bc_demo_dataset(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.demo_save_path) or ".", exist_ok=True)
            tmp = _tmp_npz_path(self.demo_save_path)
            np.savez_compressed(
                tmp,
                observations=np.array(self.demo_obs, dtype=np.float32),
                actions=np.array(self.demo_actions, dtype=np.int64),
                action_masks=np.array(self.demo_masks, dtype=np.bool_),
                created_at=np.array(datetime.now().isoformat()),
                samples=np.array(len(self.demo_actions), dtype=np.int64),
            )
            os.replace(tmp, self.demo_save_path)
            log(f"BC demo dataset saved to {self.demo_save_path} "
                f"({len(self.demo_actions)} samples)")
        except Exception as e:
            log(f"Failed to save BC demo dataset: {e}")

    def _try_load_bc_anchor(self) -> None:
        if self.trainer is None or self.bc_anchor_coef <= 0.0:
            return
        if not os.path.isfile(self.demo_save_path):
            log(f"No BC anchor demos found at {self.demo_save_path}")
            return
        try:
            with np.load(self.demo_save_path, allow_pickle=False) as data:
                obs = data["observations"]
                actions = data["actions"]
                masks = data["action_masks"]
            self.trainer.set_bc_reference(
                obs, actions, masks,
                coef=self.bc_anchor_coef,
                batch_size=64,
            )
            log(f"Loaded BC anchor demos from {self.demo_save_path}: "
                f"{len(actions)} samples coef={self.bc_anchor_coef}")
        except Exception as e:
            log(f"Failed to load BC anchor demos {self.demo_save_path}: {e}")

    def _train_supervised(self, n: int):
        """Train cross-entropy loss on BC demonstrations."""
        log(f"Supervised training: {n} samples, {self.bc_epochs} epochs...")

        obs_t = torch.as_tensor(np.array(self.demo_obs, dtype=np.float32))
        act_t = torch.as_tensor(np.array(self.demo_actions, dtype=np.int64))
        mask_t = torch.as_tensor(np.array(self.demo_masks, dtype=np.bool_))

        optimizer = torch.optim.Adam(
            list(self.trainer.shared.parameters())
            + list(self.trainer.policy_head.parameters())
            + list(self.trainer.value_head.parameters()),
            lr=self.bc_lr,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        batch_size = 128
        acc = 0.0

        for epoch in range(self.bc_epochs):
            indices = np.random.permutation(n)
            total_loss = 0.0
            correct = 0
            batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_act = act_t[idx]
                b_mask = mask_t[idx]

                features = self.trainer.shared(b_obs)
                logits = self.trainer.policy_head(features)
                logits = logits.masked_fill(~b_mask, -1e8)

                loss = loss_fn(logits, b_act)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.trainer.shared.parameters())
                    + list(self.trainer.policy_head.parameters())
                    + list(self.trainer.value_head.parameters()),
                    1.0,
                )
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == b_act).sum().item()
                batches += 1

            acc = correct / n * 100
            avg_loss = total_loss / max(1, batches)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log(f"  BC epoch {epoch + 1}/{self.bc_epochs}: "
                    f"loss={avg_loss:.4f} acc={acc:.1f}%")

        log(f"BC training complete: final acc={acc:.1f}%")

    # ------------------------------------------------------------------
    # Phase 2: PPO fine-tuning
    # ------------------------------------------------------------------
    def _ppo_step(self, gs) -> Action:
        try:
            return self._ppo_step_inner(gs)
        except Exception:
            log(f"CRASH in _ppo_step: {traceback.format_exc()}")
            if getattr(gs, "proceed_available", False):
                return Action("proceed")
            return ChooseAction(choice_index=0)

    def _ppo_step_inner(self, gs) -> Action:
        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)

        screen = self._screen_name(gs)
        terminal = is_terminal_state(gs)

        victory = is_victory_state(gs)
        self.fight_tracker.observe(
            gs, game=self.total_games + 1,
            terminal=terminal, victory=victory,
        )

        # First state of the game
        if not self.ppo_initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.ppo_initialized = True
            self.prev_obs = obs
            self.prev_mask = mask
            self._game_start_buf_len = len(self.buffer)
            self._ppo_game_start_steps = self.ppo_steps
            log(f"PPO game #{self.ppo_games_done + 1} started, "
                f"floor={getattr(gs, 'floor', '?')} "
                f"ent_coef={self.trainer.ent_coef:.4f}")

        # Compute reward for the previous action
        reward = self.reward_tracker.compute(gs, terminal, victory)
        self.episode_reward += reward

        # Store transition for the PREVIOUS RL action
        if self.prev_action is not None:
            total_reward = self.pending_reward + reward
            prev_action = self.prev_action
            self.buffer.add(
                obs=self.prev_obs,
                action=prev_action,
                reward=total_reward,
                done=terminal,
                mask=self.prev_mask,
                log_prob=self.prev_log_prob,
                value=self.prev_value,
            )
            if VERBOSE:
                log(f"PPO TRANSITION: game={self.ppo_games_done + 1} "
                    f"prev_action={prev_action} reward={total_reward:.3f} "
                    f"done={terminal} buffer={len(self.buffer)} "
                    f"pending_before={self.pending_reward:.3f} immediate={reward:.3f} "
                    f"{_state_desc(gs, screen)}")
            self.prev_action = None
            self.pending_reward = 0.0
        else:
            self.pending_reward += reward
            if VERBOSE and abs(reward) > 1e-6:
                log(f"PPO PENDING REWARD: game={self.ppo_games_done + 1} "
                    f"pending={self.pending_reward:.3f} immediate={reward:.3f} "
                    f"{_state_desc(gs, screen)}")

        # Game over — train and dismiss
        if terminal:
            self._end_ppo_game(gs, victory)
            if getattr(gs, "proceed_available", False):
                return Action("proceed")
            if getattr(gs, "cancel_available", False):
                return Action("leave")
            return Action("state")

        # Auto-handle mechanical screens
        auto = self._auto_handle_screen(gs, screen)
        if auto is not None:
            self._stuck_count = 0
            self.ppo_steps += 1
            if VERBOSE:
                log(f"PPO AUTO STEP {self.ppo_steps}: game={self.ppo_games_done + 1} "
                    f"pending_reward={self.pending_reward:.3f} "
                    f"action={_action_desc(auto)} {_state_desc(gs, screen)}")
            return auto

        # Stuck detection
        cur_floor = int(getattr(gs, "floor", -1) or -1)
        if cur_floor == self._stuck_floor:
            self._stuck_count += 1
        else:
            self._stuck_floor = cur_floor
            self._stuck_count = 0

        if self._stuck_count >= 30:
            self._stuck_count = 0
            choice_list = list(getattr(gs, "choice_list", []) or [])
            log(f"PPO ISSUE stuck on {screen} floor={cur_floor}, forcing action: "
                f"{_state_desc(gs, screen)}")
            if getattr(gs, "proceed_available", False):
                return Action("proceed")
            if getattr(gs, "cancel_available", False):
                return Action("leave")
            if choice_list:
                return ChooseAction(choice_index=0)
            return Action("proceed")

        # RL policy picks the action
        if int(mask.sum()) <= 1 and bool(mask[_NOOP]):
            issue_key = f"{screen}:{cur_floor}:noop_only"
            if issue_key != self._last_issue_key:
                log(f"PPO ISSUE only no-op legal before policy action: "
                    f"{_state_desc(gs, screen)}")
                self._last_issue_key = issue_key
        action, log_prob, value = self.trainer.predict(obs, mask)
        spire_action = flat_action_to_spire_action(action, gs)

        self.prev_obs = obs
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.prev_mask = mask
        self.ppo_steps += 1

        if VERBOSE or self.ppo_steps % 5 == 1 or self.ppo_steps <= 3:
            n_legal = int(mask.sum())
            floor = getattr(gs, "floor", "?")
            hp = getattr(gs, "current_hp", "?")
            in_combat = getattr(gs, "in_combat", False)
            action_str = (str(spire_action.command)
                          if hasattr(spire_action, "command")
                          else type(spire_action).__name__)
            prefix = "PPO STEP" if VERBOSE else "  step"
            log(f"{prefix} {self.ppo_steps}: game={self.ppo_games_done + 1} "
                f"floor={floor} hp={hp} screen={screen} "
                f"combat={in_combat} legal={n_legal} action={action}->{action_str} "
                f"log_prob={log_prob:.3f} value={value:.3f} "
                f"reward_now={reward:.3f} pending={self.pending_reward:.3f} "
                f"{_state_desc(gs, screen) if VERBOSE else ''}".rstrip())

        return spire_action

    def _auto_handle_screen(self, gs, screen_name: str) -> Optional[Action]:
        return auto_handle_screen(gs, screen_name, heuristic_all=False)

    def _end_ppo_game(self, gs, victory: bool):
        """PPO update (every N games), entropy annealing, stats, model save."""
        self.ppo_games_done += 1
        self.total_games += 1
        self.games_since_update += 1
        fight_stats = self.fight_tracker.finish_game(
            gs, game=self.total_games, victory=victory
        )
        n_this_game = len(self.buffer) - self._game_start_buf_len
        steps_this_game = max(0, self.ppo_steps - self._ppo_game_start_steps)
        n_buffered = len(self.buffer)
        log(f"PPO game #{self.ppo_games_done} ended: {n_this_game} this game "
            f"{steps_this_game} steps this game "
            f"({n_buffered} buffered, {self.games_since_update}/{self.games_per_update}), "
            f"reward={self.episode_reward:.2f}, victory={victory}")

        # PPO update only when we've accumulated enough games
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
                    f"bc={stats.get('bc_loss', 0.0):.4f} "
                    f"transitions={n_buffered}")
            self.buffer.clear()
            self.games_since_update = 0

        # Entropy annealing: linear decay from ent_start to ent_end
        if self.ppo_games > 0:
            progress = min(1.0, self.ppo_games_done / self.ppo_games)
        else:
            # Unlimited mode: anneal over first 200 games
            progress = min(1.0, self.ppo_games_done / 200)
        new_ent = self.ent_start + (self.ent_end - self.ent_start) * progress
        self.trainer.ent_coef = new_ent

        # Stats CSV (same format as train_ppo.py)
        row = {
            "timestamp": datetime.now().isoformat(),
            "game": self.total_games,
            "total_updates": self.trainer.total_updates,
            "steps": steps_this_game,
            "transitions": n_this_game,
            "total_reward": round(self.episode_reward, 4),
            "final_hp": int(getattr(gs, "current_hp", 0) or 0),
            "final_max_hp": int(getattr(gs, "max_hp", 0) or 0),
            "final_floor": int(getattr(gs, "floor", 0) or 0),
            "final_act": int(getattr(gs, "act", 0) or 0),
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

        # Periodic save
        if self.ppo_games_done % self.save_every == 0:
            self.trainer.save(self.save_path)
            log(f"  Model saved to {self.save_path} "
                f"(ent_coef={new_ent:.4f})")

        # Reset per-game state (buffer persists across games until the update
        # threshold is hit above, then cleared there)
        self.episode_reward = 0.0
        self.pending_reward = 0.0
        self.prev_obs = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.prev_mask = None
        self._stuck_floor = -1
        self._stuck_count = 0
        self.fight_tracker.reset_game()
        self.ppo_initialized = False

    @staticmethod
    def _screen_name(gs) -> str:
        st = getattr(gs, "screen_type", None)
        name = getattr(st, "name", st) if st is not None else "NONE"
        return str(name) if name else "NONE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global VERBOSE
    parser = argparse.ArgumentParser(
        description="End-to-end BC warm-start -> PPO fine-tuning for STS")

    # BC phase
    bc = parser.add_argument_group("BC phase")
    bc.add_argument("--bc-games", type=int, default=150,
                    help="Heuristic demonstration games (default: 150)")
    bc.add_argument("--bc-epochs", type=int, default=30,
                    help="Supervised training epochs (default: 30)")
    bc.add_argument("--bc-lr", type=float, default=1e-3,
                    help="BC learning rate (default: 1e-3)")

    # PPO phase
    ppo = parser.add_argument_group("PPO phase")
    ppo.add_argument("--ppo-games", type=int, default=0,
                     help="PPO games; 0 = unlimited (default: 0)")
    ppo.add_argument("--ppo-lr", type=float, default=1e-4,
                     help="PPO learning rate (default: 1e-4)")
    ppo.add_argument("--ent-start", type=float, default=0.001,
                     help="Initial entropy coefficient (default: 0.001)")
    ppo.add_argument("--ent-end", type=float, default=0.001,
                     help="Final entropy coefficient (default: 0.001)")
    ppo.add_argument("--clip", type=float, default=0.15,
                     help="PPO clip range (default: 0.15)")
    ppo.add_argument("--target-kl", type=float, default=0.03,
                     help="Stop PPO epochs early when approx KL exceeds this value")
    ppo.add_argument("--bc-anchor-coef", type=float, default=0.02,
                     help="Small BC imitation loss during PPO (default: 0.02)")

    # General
    gen = parser.add_argument_group("General")
    gen.add_argument("--save", type=str, default="models/ppo_sts.pt",
                     help="Model save path (default: models/ppo_sts.pt)")
    gen.add_argument("--demo-save", type=str, default=None,
                     help="Where to save/load BC demos for PPO anchoring")
    gen.add_argument("--bc-checkpoint", type=str, default=None,
                     help="Per-game BC progress checkpoint path")
    gen.add_argument("--no-resume-bc", dest="resume_bc", action="store_false",
                     help="Ignore any existing BC progress checkpoint")
    gen.add_argument("--save-every", type=int, default=5,
                     help="Save every N PPO games (default: 5)")
    gen.add_argument("--resume", type=str, default=None,
                     help="Resume from checkpoint (skips BC phase)")
    gen.add_argument("--games-per-update", type=int, default=4,
                     help="Accumulate N games of PPO transitions per update (default: 4)")
    gen.add_argument("--verbose", action="store_true",
                     help="Write detailed per-state/per-action debug logs")

    args = parser.parse_args()
    VERBOSE = VERBOSE or args.verbose

    save_path = os.path.join(_root, args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    demo_save = args.demo_save
    if demo_save is not None and not os.path.isabs(demo_save):
        demo_save = os.path.join(_root, demo_save)
    bc_checkpoint = args.bc_checkpoint
    if bc_checkpoint is not None and not os.path.isabs(bc_checkpoint):
        bc_checkpoint = os.path.join(_root, bc_checkpoint)

    resume_path = os.path.join(_root, args.resume) if args.resume else None

    log(f"Config: bc_games={args.bc_games} bc_epochs={args.bc_epochs} "
        f"bc_lr={args.bc_lr} ppo_games={args.ppo_games} ppo_lr={args.ppo_lr} "
        f"ent={args.ent_start}->{args.ent_end} clip={args.clip} "
        f"target_kl={args.target_kl} bc_anchor={args.bc_anchor_coef} resume={args.resume} "
        f"resume_bc={args.resume_bc} verbose={VERBOSE}")

    agent = BCPPOAgent(
        bc_games=args.bc_games,
        ppo_games=args.ppo_games,
        save_path=save_path,
        save_every=args.save_every,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        ppo_lr=args.ppo_lr,
        ent_start=args.ent_start,
        ent_end=args.ent_end,
        clip_range=args.clip,
        target_kl=args.target_kl,
        resume_path=resume_path,
        games_per_update=args.games_per_update,
        bc_anchor_coef=args.bc_anchor_coef,
        demo_save_path=demo_save,
        bc_checkpoint_path=bc_checkpoint,
        resume_bc=args.resume_bc,
    )

    log("Setting up Coordinator...")
    coord = Coordinator()
    coord.register_state_change_callback(agent.on_state_change)
    coord.register_out_of_game_callback(agent.on_out_of_game)
    coord.register_command_error_callback(agent.on_error)

    phase_str = "PPO (resumed)" if resume_path else f"BC ({args.bc_games} games) -> PPO"
    log(f"Starting: {phase_str}")
    coord.signal_ready()

    try:
        coord.run()
    except StopIteration:
        log("Training complete (StopIteration)")

    # Final save
    if agent.trainer:
        agent.trainer.save(save_path)
        log(f"Final model saved to {save_path}")

    log(f"=== BC->PPO PIPELINE COMPLETE ===\n"
        f"    BC games: {agent.bc_games_done}\n"
        f"    PPO games: {agent.ppo_games_done}\n"
        f"    Total games: {agent.total_games}\n"
        f"    PPO updates: {agent.trainer.total_updates if agent.trainer else 0}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
