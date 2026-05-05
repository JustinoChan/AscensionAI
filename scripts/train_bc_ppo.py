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
  command=python scripts/train_bc_ppo.py --bc-games 50 --ppo-games 200

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

# Same columns as train_ppo.py for plot_training.py compatibility
_STATS_COLUMNS = [
    "timestamp", "game", "total_updates", "steps", "transitions",
    "total_reward", "final_hp", "final_max_hp", "final_floor", "final_act",
    "victory", "terminated", "pg_loss", "vf_loss", "entropy", "worker",
    "elites_fought", "elites_won", "bosses_fought", "bosses_won",
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
from screen_handler import auto_handle_screen
from behavior_clone import heuristic_action

# Re-patch after behavior_clone import (its module-level code overwrites
# _coord_module.write_stdout with a version that uses stderr)
_coord_module.write_stdout = _patched_write_stdout

log("Imports done")


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
        ent_start: float = 0.05,
        ent_end: float = 0.01,
        clip_range: float = 0.15,
        resume_path: Optional[str] = None,
        games_per_update: int = 4,
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
        self.games_per_update = games_per_update
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
        self.bc_initialized = False

        # --- PPO trainer (created after BC, or loaded from checkpoint) ---
        if resume_path:
            self.trainer = PPOTrainer(
                obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
                lr=ppo_lr, gamma=0.995, gae_lambda=0.95,
                clip_range=clip_range, ent_coef=ent_start, vf_coef=0.5,
                n_epochs=4, batch_size=64, net_arch=(256, 256),
            )
            self.trainer.load(resume_path)
            log(f"Resumed from {resume_path} (updates={self.trainer.total_updates})")
        else:
            self.trainer: Optional[PPOTrainer] = None

        # --- PPO per-game state ---
        self.buffer = GameBuffer()
        self.reward_tracker = RewardTracker()
        self.ppo_games_done = 0
        self.ppo_steps = 0
        self.total_games = 0
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
        if "Possible commands" in err and "wait" in err:
            return Action("wait")
        if "proceed" in err and "choose" in err:
            return ChooseAction(choice_index=0)
        return Action("state")

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

        if not self.bc_initialized:
            self.bc_initialized = True
            log(f"BC game #{self.bc_games_done + 1}/{self.bc_games} started, "
                f"floor={getattr(gs, 'floor', '?')}")

        if terminal:
            self.bc_games_done += 1
            self.total_games += 1
            self.bc_initialized = False

            victory = is_victory_state(gs)

            log(f"BC game #{self.bc_games_done} ended: "
                f"floor={getattr(gs, 'floor', '?')} victory={victory} "
                f"samples={len(self.demo_obs)}")

            # Log BC game stats (no PPO losses yet)
            _append_stats_csv({
                "timestamp": datetime.now().isoformat(),
                "game": self.total_games,
                "steps": self.bc_steps,
                "transitions": len(self.demo_obs),
                "final_hp": int(getattr(gs, "current_hp", 0) or 0),
                "final_max_hp": int(getattr(gs, "max_hp", 0) or 0),
                "final_floor": int(getattr(gs, "floor", 0) or 0),
                "final_act": int(getattr(gs, "act", 0) or 0),
                "victory": int(victory),
                "terminated": 1,
            })

            # Collected enough demos — train and switch to PPO
            if self.bc_games_done >= self.bc_games:
                self._transition_to_ppo()

            if getattr(gs, "proceed_available", False):
                return Action("proceed")
            return Action("state")

        # --- Heuristic plays, we record (obs, action, mask) ---
        spire_action, action_id = heuristic_action(gs)
        if spire_action is None or action_id is None:
            return Action("state")

        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)

        # Only record if the heuristic's action is legal under our mask
        if action_id < NUM_ACTIONS and mask[action_id]:
            self.demo_obs.append(obs)
            self.demo_actions.append(action_id)
            self.demo_masks.append(mask)

        self.bc_steps += 1
        if self.bc_steps % 100 == 0:
            log(f"  BC step={self.bc_steps} samples={len(self.demo_obs)} "
                f"games={self.bc_games_done}/{self.bc_games}")

        return spire_action

    # ------------------------------------------------------------------
    # Transition: supervised training on BC demos, then switch to PPO
    # ------------------------------------------------------------------
    def _transition_to_ppo(self):
        n = len(self.demo_obs)
        log(f"=== TRANSITION: BC -> PPO ({n} demos from {self.bc_games_done} games) ===")

        # Create trainer with BC learning rate for supervised phase
        self.trainer = PPOTrainer(
            obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
            lr=self.bc_lr, gamma=0.995, gae_lambda=0.95,
            clip_range=self.clip_range, ent_coef=self.ent_start, vf_coef=0.5,
            n_epochs=4, batch_size=64, net_arch=(256, 256),
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

        # Free demo data
        self.demo_obs.clear()
        self.demo_actions.clear()
        self.demo_masks.clear()

        self.phase = self.PHASE_PPO
        log(f"=== PPO PHASE STARTED (lr={self.ppo_lr}, ent={self.ent_start}, "
            f"clip={self.clip_range}) ===")

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

        # First state of the game
        if not self.ppo_initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.ppo_initialized = True
            self.prev_obs = obs
            self.prev_mask = mask
            self._game_start_buf_len = len(self.buffer)
            log(f"PPO game #{self.ppo_games_done + 1} started, "
                f"floor={getattr(gs, 'floor', '?')} "
                f"ent_coef={self.trainer.ent_coef:.4f}")

        # Compute reward for the previous action
        reward = self.reward_tracker.compute(gs, terminal, victory)
        self.episode_reward += reward

        # Store transition for the PREVIOUS RL action
        if self.prev_action is not None:
            total_reward = self.pending_reward + reward
            self.buffer.add(
                obs=self.prev_obs,
                action=self.prev_action,
                reward=total_reward,
                done=terminal,
                mask=self.prev_mask,
                log_prob=self.prev_log_prob,
                value=self.prev_value,
            )
            self.prev_action = None
            self.pending_reward = 0.0
        else:
            self.pending_reward += reward

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
            log(f"  STUCK on {screen} floor={cur_floor}, forcing action")
            if getattr(gs, "proceed_available", False):
                return Action("proceed")
            if getattr(gs, "cancel_available", False):
                return Action("leave")
            if choice_list:
                return ChooseAction(choice_index=0)
            return Action("proceed")

        # RL policy picks the action
        action, log_prob, value = self.trainer.predict(obs, mask)
        spire_action = flat_action_to_spire_action(action, gs)

        self.prev_obs = obs
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.prev_mask = mask
        self.ppo_steps += 1

        if self.ppo_steps % 5 == 1 or self.ppo_steps <= 3:
            n_legal = int(mask.sum())
            floor = getattr(gs, "floor", "?")
            hp = getattr(gs, "current_hp", "?")
            in_combat = getattr(gs, "in_combat", False)
            action_str = (str(spire_action.command)
                          if hasattr(spire_action, "command")
                          else type(spire_action).__name__)
            log(f"  step={self.ppo_steps} floor={floor} hp={hp} screen={screen} "
                f"combat={in_combat} legal={n_legal} "
                f"action={action}->{action_str} r={reward:.3f}")

        return spire_action

    def _auto_handle_screen(self, gs, screen_name: str) -> Optional[Action]:
        return auto_handle_screen(gs, screen_name, heuristic_all=False)

    def _end_ppo_game(self, gs, victory: bool):
        """PPO update (every N games), entropy annealing, stats, model save."""
        self.ppo_games_done += 1
        self.total_games += 1
        self.games_since_update += 1
        n_this_game = len(self.buffer) - self._game_start_buf_len
        n_buffered = len(self.buffer)
        log(f"PPO game #{self.ppo_games_done} ended: {n_this_game} this game "
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
                    f"ent={stats['entropy']:.4f} transitions={n_buffered}")
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
            "steps": self.ppo_steps,
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
        }
        _append_stats_csv(row)

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
    parser = argparse.ArgumentParser(
        description="End-to-end BC warm-start -> PPO fine-tuning for STS")

    # BC phase
    bc = parser.add_argument_group("BC phase")
    bc.add_argument("--bc-games", type=int, default=50,
                    help="Heuristic demonstration games (default: 50)")
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
    ppo.add_argument("--ent-start", type=float, default=0.05,
                     help="Initial entropy coefficient (default: 0.05)")
    ppo.add_argument("--ent-end", type=float, default=0.01,
                     help="Final entropy coefficient (default: 0.01)")
    ppo.add_argument("--clip", type=float, default=0.15,
                     help="PPO clip range (default: 0.15)")

    # General
    gen = parser.add_argument_group("General")
    gen.add_argument("--save", type=str, default="models/ppo_sts.pt",
                     help="Model save path (default: models/ppo_sts.pt)")
    gen.add_argument("--save-every", type=int, default=5,
                     help="Save every N PPO games (default: 5)")
    gen.add_argument("--resume", type=str, default=None,
                     help="Resume from checkpoint (skips BC phase)")
    gen.add_argument("--games-per-update", type=int, default=4,
                     help="Accumulate N games of PPO transitions per update (default: 4)")

    args = parser.parse_args()

    save_path = os.path.join(_root, args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    resume_path = os.path.join(_root, args.resume) if args.resume else None

    log(f"Config: bc_games={args.bc_games} bc_epochs={args.bc_epochs} "
        f"bc_lr={args.bc_lr} ppo_games={args.ppo_games} ppo_lr={args.ppo_lr} "
        f"ent={args.ent_start}->{args.ent_end} clip={args.clip} "
        f"resume={args.resume}")

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
        resume_path=resume_path,
        games_per_update=args.games_per_update,
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
