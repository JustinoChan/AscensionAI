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
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

from spirecomm_patches import apply_all as _apply_spirecomm_patches
_apply_spirecomm_patches(_real_stdout)
# ---- End stdout fix ----

_scripts = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_scripts)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

import argparse
import csv
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
    RewardTracker, is_terminal_state, is_victory_state,
)
from ppo_model import PPOTrainer
from screen_handler import auto_handle_screen, recover_from_command_error
from fight_tracker import FightTracker
from behavior_clone import heuristic_action

_apply_spirecomm_patches(_real_stdout)


os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "Eval", "eval_debug.log")
EVAL_CSV = os.path.join(_root, "logs", "eval_stats.csv")
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

_EVAL_COLUMNS = [
    "timestamp", "run", "policy", "model", "seed", "game", "steps", "total_reward",
    "final_hp", "final_max_hp", "final_floor", "final_act", "victory",
    "elites_fought", "elites_won", "bosses_fought", "bosses_won",
    "elites_fought_act1", "elites_won_act1", "bosses_fought_act1", "bosses_won_act1",
    "elites_fought_act2", "elites_won_act2", "bosses_fought_act2", "bosses_won_act2",
    "elites_fought_act3", "elites_won_act3", "bosses_fought_act3", "bosses_won_act3",
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
        with open(EVAL_CSV, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=_EVAL_COLUMNS).writeheader()
        return

    try:
        with open(EVAL_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and all(c in reader.fieldnames for c in _EVAL_COLUMNS):
                return
            rows = list(reader)
        with open(EVAL_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_EVAL_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        log(f"eval csv migration failed: {e}")


def _csv_game_exists(run_tag: str, game: int) -> bool:
    if game <= 0 or not os.path.exists(EVAL_CSV):
        return False
    try:
        with open(EVAL_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    existing_game = int(row.get("game", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if row.get("run") == run_tag and existing_game == game:
                    return True
    except Exception as e:
        log(f"csv duplicate check failed: {e}")
    return False


def _append_csv(row: dict) -> None:
    try:
        _init_csv()
        run_tag = str(row.get("run", ""))
        try:
            game = int(row.get("game", 0) or 0)
        except (TypeError, ValueError):
            game = 0
        if _csv_game_exists(run_tag, game):
            log(f"Skipping duplicate eval csv row for run={run_tag} game={game}")
            return
        with open(EVAL_CSV, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=_EVAL_COLUMNS,
                           extrasaction="ignore").writerow(row)
    except Exception as e:
        log(f"csv append failed: {e}")


def _row_int(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _row_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _completed_rows_for_run(run_tag: str) -> list[dict]:
    rows_by_game: dict[int, dict] = {}
    if not os.path.exists(EVAL_CSV):
        return []
    try:
        with open(EVAL_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("run") != run_tag:
                    continue
                game = _row_int(row, "game")
                if game > 0 and game not in rows_by_game:
                    rows_by_game[game] = row
    except Exception as e:
        log(f"resume scan failed for run={run_tag}: {e}")
        return []

    completed: list[dict] = []
    game = 1
    while game in rows_by_game:
        completed.append(rows_by_game[game])
        game += 1
    return completed


def _action_desc(action: Optional[Action]) -> str:
    command = str(getattr(action, "command", type(action).__name__))
    parts = [command]
    for attr in ("choice_index", "name", "card_index", "target_index"):
        value = getattr(action, attr, None)
        if value not in (None, -1):
            parts.append(f"{attr}={value}")
    return " ".join(parts)


def auto_handle(gs: Any, screen_name: str) -> Optional[Action]:
    return auto_handle_screen(gs, screen_name, heuristic_all=False)


# ---------------------------------------------------------------------------
# Evaluator agent
# ---------------------------------------------------------------------------
class EvalAgent:
    def __init__(
        self,
        trainer: Optional[PPOTrainer],
        target_games: int,
        run_tag: str,
        policy: str = "model",
        model_label: str = "",
        seeds: Optional[list[str]] = None,
        top_actions: int = 0,
        completed_rows: Optional[list[dict]] = None,
        restart_every: int = 0,
    ):
        self.trainer = trainer
        self.target_games = target_games
        self.run_tag = run_tag
        self.policy = policy
        self.model_label = model_label
        self.seeds = seeds or []
        self.top_actions = top_actions
        self.restart_every = max(0, int(restart_every))
        self.reward_tracker = RewardTracker()
        completed_rows = completed_rows or []
        self.games_played = len(completed_rows)
        self._games_at_start = self.games_played
        self.total_steps = 0
        self.episode_reward = 0.0
        self.initialized = False

        self.wins = 0
        self.sum_floor = 0
        self.sum_reward = 0.0
        self.elites_fought = 0
        self.elites_won = 0
        self.bosses_fought = 0
        self.bosses_won = 0
        self._per_act_totals: dict[str, int] = {}
        self.fight_tracker = FightTracker(source="eval", worker="eval", log=log)
        self._current_seed = ""
        self._preload_completed_summary(completed_rows)

    def _preload_completed_summary(self, rows: list[dict]) -> None:
        if not rows:
            log(f"Starting fresh eval run {self.run_tag}")
            return
        for row in rows:
            self.sum_floor += _row_int(row, "final_floor")
            self.sum_reward += _row_float(row, "total_reward")
            self.wins += 1 if _row_int(row, "victory") else 0
            self.elites_fought += _row_int(row, "elites_fought")
            self.elites_won += _row_int(row, "elites_won")
            self.bosses_fought += _row_int(row, "bosses_fought")
            self.bosses_won += _row_int(row, "bosses_won")
            for a in (1, 2, 3):
                for k in (f"elites_fought_act{a}", f"elites_won_act{a}",
                          f"bosses_fought_act{a}", f"bosses_won_act{a}"):
                    self._per_act_totals[k] = self._per_act_totals.get(k, 0) + _row_int(row, k)
        log(f"Resuming eval run {self.run_tag}: {self.games_played} contiguous games already recorded")

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
        terminal = is_terminal_state(gs)

        victory = is_victory_state(gs)
        self.fight_tracker.observe(
            gs, game=self.games_played + 1,
            terminal=terminal, victory=victory,
        )

        if not self.initialized:
            self.reward_tracker.reset(gs)
            self.reward_tracker._last_act = int(getattr(gs, "act", 0) or 0)
            self.initialized = True
            log(f"Eval game #{self.games_played + 1} starting, "
                f"policy={self.policy} seed={self._current_seed or 'random'} "
                f"floor={getattr(gs, 'floor', '?')}")

        reward = self.reward_tracker.compute(gs, terminal, victory)
        self.episode_reward += reward

        if terminal:
            self._end_game(gs, victory)
            games_this_process = self.games_played - self._games_at_start
            if (self.restart_every > 0
                    and games_this_process >= self.restart_every
                    and self.games_played < self.target_games):
                log(f"Restart-every threshold reached: {games_this_process}/{self.restart_every} "
                    f"— exiting for RAM cleanup ({self.games_played}/{self.target_games} total)")
                os._exit(0)
            if self.games_played >= self.target_games:
                log("target games reached — requesting exit")
                pa = self._per_act_totals
                summary = (
                    f"EVAL COMPLETE: {self.games_played} games, "
                    f"wins={self.wins} ({self.wins / max(1, self.games_played):.1%}), "
                    f"avg_floor={self.sum_floor / max(1, self.games_played):.2f}, "
                    f"avg_reward={self.sum_reward / max(1, self.games_played):.2f}, "
                    f"elites={self.elites_won}/{self.elites_fought} "
                    f"(A1:{pa.get('elites_won_act1',0)}/{pa.get('elites_fought_act1',0)} "
                    f"A2:{pa.get('elites_won_act2',0)}/{pa.get('elites_fought_act2',0)} "
                    f"A3:{pa.get('elites_won_act3',0)}/{pa.get('elites_fought_act3',0)}), "
                    f"bosses={self.bosses_won}/{self.bosses_fought} "
                    f"(A1:{pa.get('bosses_won_act1',0)}/{pa.get('bosses_fought_act1',0)} "
                    f"A2:{pa.get('bosses_won_act2',0)}/{pa.get('bosses_fought_act2',0)} "
                    f"A3:{pa.get('bosses_won_act3',0)}/{pa.get('bosses_fought_act3',0)})"
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
            if VERBOSE:
                choice_list = list(getattr(gs, "choice_list", []) or [])
                log(f"EVAL AUTO STEP {self.total_steps}: game={self.games_played + 1} "
                    f"floor={getattr(gs, 'floor', '?')} screen={screen_name} "
                    f"choices={choice_list[:8]} action={_action_desc(auto)}")
            return auto

        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)
        if self.policy == "heuristic":
            spire_action, action = heuristic_action(gs)
            if spire_action is None or action is None:
                return Action("state")
        else:
            action, _lp, _v = self.trainer.predict(obs, mask, deterministic=True)
            spire_action = flat_action_to_spire_action(action, gs)
            if self.top_actions > 0:
                self._log_top_actions(gs, obs, mask)
        self.total_steps += 1
        if VERBOSE:
            choice_list = list(getattr(gs, "choice_list", []) or [])
            log(f"EVAL RL STEP {self.total_steps}: game={self.games_played + 1} "
                f"floor={getattr(gs, 'floor', '?')} "
                f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')} "
                f"screen={screen_name} choices={choice_list[:8]} "
                f"mask_sum={int(mask.sum())} action_id={action} "
                f"action={_action_desc(spire_action)}")
        return spire_action

    def _log_top_actions(self, gs, obs: np.ndarray, mask: np.ndarray) -> None:
        if self.trainer is None:
            return
        try:
            probs, value = self.trainer.action_probabilities(obs, mask)
            legal = np.where(mask)[0]
            if legal.size == 0:
                return
            top = legal[np.argsort(probs[legal])[::-1][:self.top_actions]]
            parts = []
            for rank, action_id in enumerate(top, start=1):
                act = flat_action_to_spire_action(int(action_id), gs)
                parts.append(f"{rank}. {int(action_id)} {_action_desc(act)}={probs[action_id]:.1%}")
            log(f"EVAL TOP ACTIONS: game={self.games_played + 1} "
                f"floor={getattr(gs, 'floor', '?')} "
                f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')} "
                f"value={value:.3f} {' | '.join(parts)}")
        except Exception as e:
            log(f"top action logging failed: {e}")

    def _end_game(self, final_gs, victory: bool) -> None:
        self.games_played += 1
        floor = int(getattr(final_gs, "floor", 0) or 0)
        fight_stats = self.fight_tracker.finish_game(
            final_gs, game=self.games_played, victory=victory
        )
        self.elites_fought += fight_stats["elites_fought"]
        self.elites_won += fight_stats["elites_won"]
        self.bosses_fought += fight_stats["bosses_fought"]
        self.bosses_won += fight_stats["bosses_won"]
        for k, v in fight_stats.items():
            if k.endswith(("_act1", "_act2", "_act3")):
                self._per_act_totals[k] = self._per_act_totals.get(k, 0) + v
        self.sum_floor += floor
        self.sum_reward += self.episode_reward
        if victory:
            self.wins += 1

        _append_csv({
            "timestamp": datetime.now().isoformat(),
            "run": self.run_tag,
            "policy": self.policy,
            "model": self.model_label,
            "seed": self._current_seed,
            "game": self.games_played,
            "steps": self.total_steps,
            "total_reward": round(self.episode_reward, 4),
            "final_hp": int(getattr(final_gs, "current_hp", 0) or 0),
            "final_max_hp": int(getattr(final_gs, "max_hp", 0) or 0),
            "final_floor": floor,
            "final_act": int(getattr(final_gs, "act", 0) or 0),
            "victory": int(bool(victory)),
            **fight_stats,
        })
        log(
            f"Game #{self.games_played}: floor={floor} "
            f"hp={getattr(final_gs, 'current_hp', '?')} victory={victory} "
            f"reward={self.episode_reward:.2f} "
            f"elites={fight_stats['elites_won']}/{fight_stats['elites_fought']} "
            f"bosses={fight_stats['bosses_won']}/{fight_stats['bosses_fought']}"
        )

        self.episode_reward = 0.0
        self.initialized = False
        self.total_steps = 0

    def on_out_of_game(self) -> Action:
        if self.games_played < len(self.seeds):
            self._current_seed = str(self.seeds[self.games_played])
        else:
            self._current_seed = ""
        return StartGameAction(
            player_class=PlayerClass.IRONCLAD,
            ascension_level=0,
            seed=self._current_seed or None,
        )

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        return recover_from_command_error(err)


def main() -> None:
    global DEBUG_LOG, VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_sts.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--run-tag", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--policy", choices=("model", "heuristic"), default="model",
                        help="Evaluate trained model or heuristic baseline")
    parser.add_argument("--seed", action="append", default=[],
                        help="Fixed seed; can be repeated")
    parser.add_argument("--seed-file", type=str, default=None,
                        help="File of one seed per line for fixed-seed eval")
    parser.add_argument("--top-actions", type=int, default=0,
                        help="Log top N model actions at each RL decision")
    parser.add_argument("--resume-run", action="store_true",
                        help="Resume from completed rows in logs/eval_stats.csv for this run tag")
    parser.add_argument("--restart-every", type=int, default=0,
                        help="Exit after this many games to free RAM; GUI relaunches (0 = disabled)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Debug log path for this eval run")
    parser.add_argument("--net-arch", type=str, default="512,256,256",
                        help="Comma-separated hidden layer sizes (default: 512,256,256)")
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["tanh", "gelu", "relu"],
                        help="Activation function for shared layers (default: gelu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Write detailed per-state/per-action debug logs")
    args = parser.parse_args()
    if args.log_file:
        DEBUG_LOG = args.log_file if os.path.isabs(args.log_file) else os.path.join(_root, args.log_file)
    os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
    VERBOSE = VERBOSE or args.verbose
    log("=== EVAL STARTING ===")
    _init_csv()

    seeds: list[str] = []
    if args.seed_file:
        seed_path = args.seed_file if os.path.isabs(args.seed_file) else os.path.join(_root, args.seed_file)
        try:
            with open(seed_path, "r", encoding="utf-8") as f:
                for line in f:
                    seed = line.strip()
                    if seed and not seed.startswith("#"):
                        seeds.append(seed)
        except Exception as e:
            log(f"WARNING: failed to read seed file {seed_path}: {e}")
    seeds.extend(str(s) for s in args.seed)
    if seeds and len(seeds) < args.games:
        log(f"WARNING: only {len(seeds)} seeds for {args.games} games; remaining games use random seeds")

    completed_rows = _completed_rows_for_run(args.run_tag) if args.resume_run else []
    if len(completed_rows) >= args.games:
        log(f"Resume requested and run {args.run_tag} already has {len(completed_rows)}/{args.games} games")
        return

    trainer: Optional[PPOTrainer] = None
    model_path = os.path.join(_root, args.model) if not os.path.isabs(args.model) else args.model
    log(f"Starting eval for {args.games} greedy games "
        f"run_tag={args.run_tag} policy={args.policy} verbose={VERBOSE} "
        f"seeds={len(seeds)} top_actions={args.top_actions} "
        f"resume={args.resume_run} completed={len(completed_rows)}")
    net_arch = tuple(int(x) for x in args.net_arch.split(","))
    if args.policy == "model":
        log(f"Loading model from {model_path} net_arch={net_arch} activation={args.activation}")
        trainer = PPOTrainer(
            obs_size=OBS_SIZE,
            n_actions=NUM_ACTIONS,
            device="cpu",
            net_arch=net_arch,
            activation=args.activation,
        )
        if os.path.exists(model_path):
            trainer.load(model_path)
            log(f"Loaded checkpoint (total_updates={trainer.total_updates})")
        else:
            log(f"WARNING: no checkpoint at {model_path} - evaluating randomly-initialized policy")

    agent = EvalAgent(
        trainer,
        target_games=args.games,
        run_tag=args.run_tag,
        policy=args.policy,
        model_label=args.model if args.policy == "model" else "heuristic",
        seeds=seeds,
        top_actions=max(0, args.top_actions),
        completed_rows=completed_rows,
        restart_every=args.restart_every,
    )

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
