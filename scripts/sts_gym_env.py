"""
sts_gym_env.py — Action space, masking, and rewards for Slay the Spire RL.

  - 341-dim observation via obs_encoder (cards, monster knowledge, powers, screen)
  - Discrete(114) action space with per-card, per-target control
  - Action masking ensures only legal moves are chosen
  - Reward shaping with dominant terminal bonuses
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import (
    Action,
    ChooseAction,
    PlayCardAction,
    PotionAction,
    StartGameAction,
)
from spirecomm.spire.character import PlayerClass

from obs_encoder import (
    OBS_SIZE,
    MAX_HAND,
    MAX_MONSTERS,
    MAX_POTIONS,
    MAX_CHOICES,
    encode_game_state,
    living_monsters,
)


# ---------------------------------------------------------------------------
# Debug logging (to file, never stdout)
# ---------------------------------------------------------------------------
import os as _os
_ENV_LOG = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "logs", "env_debug.log")

def _log(msg: str):
    try:
        with open(_ENV_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Action space layout — Discrete(134)
#
# Slots are sized against obs_encoder constants: MAX_HAND=10, MAX_MONSTERS=5,
# MAX_POTIONS=5, MAX_CHOICES=40. Update here in lockstep with obs_encoder
# whenever any of those caps change.
# ---------------------------------------------------------------------------
NUM_ACTIONS = 50 + 10 + 1 + 25 + 5 + MAX_CHOICES + 3  # 134 when MAX_CHOICES=40

_PLAY_TARGETED_START = 0                              # 0..49    (10 cards × 5 monsters)
_PLAY_UNTARGETED_START = _PLAY_TARGETED_START + MAX_HAND * MAX_MONSTERS  # 50
_END_TURN = _PLAY_UNTARGETED_START + MAX_HAND                             # 60
_POTION_TARGETED_START = _END_TURN + 1                                    # 61..85
_POTION_UNTARGETED_START = _POTION_TARGETED_START + MAX_POTIONS * MAX_MONSTERS  # 86..90
_CHOOSE_START = _POTION_UNTARGETED_START + MAX_POTIONS                    # 91..(91+MAX_CHOICES-1)
_PROCEED = _CHOOSE_START + MAX_CHOICES
_LEAVE = _PROCEED + 1
_NOOP = _LEAVE + 1

assert NUM_ACTIONS == _NOOP + 1, f"action layout mismatch: {NUM_ACTIONS} vs {_NOOP + 1}"


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------
def compute_action_mask(gs: Any) -> np.ndarray:
    """Build a boolean mask of legal actions for the current game state."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)

    if gs is None:
        mask[_NOOP] = True
        return mask

    in_combat = bool(getattr(gs, "in_combat", False))
    hand = list(getattr(gs, "hand", []) or [])
    monsters = list(getattr(gs, "monsters", []) or [])
    potions = list(getattr(gs, "potions", []) or [])
    choice_list = list(getattr(gs, "choice_list", []) or [])

    play_avail = bool(getattr(gs, "play_available", False))
    end_avail = bool(getattr(gs, "end_available", False))
    potion_avail = bool(getattr(gs, "potion_available", False))
    proceed_avail = bool(getattr(gs, "proceed_available", False))
    cancel_avail = bool(getattr(gs, "cancel_available", False))

    alive = living_monsters(monsters)
    n_alive = len(alive)

    has_playable = False
    if in_combat and play_avail:
        for i, card in enumerate(hand[:MAX_HAND]):
            if not getattr(card, "is_playable", False):
                continue
            has_playable = True
            if getattr(card, "has_target", False):
                for j in range(min(n_alive, MAX_MONSTERS)):
                    mask[_PLAY_TARGETED_START + i * MAX_MONSTERS + j] = True
            else:
                mask[_PLAY_UNTARGETED_START + i] = True

    if in_combat and end_avail and not has_playable:
        mask[_END_TURN] = True

    if in_combat and potion_avail:
        for k, pot in enumerate(potions[:MAX_POTIONS]):
            if not getattr(pot, "can_use", False):
                continue
            if str(getattr(pot, "potion_id", "") or "").lower() == "potion slot":
                continue
            if getattr(pot, "requires_target", False):
                for j in range(min(n_alive, MAX_MONSTERS)):
                    mask[_POTION_TARGETED_START + k * MAX_MONSTERS + j] = True
            else:
                mask[_POTION_UNTARGETED_START + k] = True

    scr = getattr(gs, "screen", None)
    screen_type = getattr(gs, "screen_type", None)
    st_name = getattr(screen_type, "name", "") if screen_type else ""

    # GRID with confirm_up: the card is already selected — only
    # proceed/cancel are valid, choosing another card would error.
    grid_confirmed = (st_name == "GRID"
                      and scr is not None
                      and getattr(scr, "confirm_up", False))

    if not grid_confirmed:
        potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())

        # For COMBAT_REWARD, screen.rewards is authoritative — choice_list
        # may be empty even when rewards are available.
        combat_rewards = (list(getattr(scr, "rewards", []) or [])
                          if st_name == "COMBAT_REWARD" and scr else [])
        n_choices = len(choice_list)
        if st_name == "COMBAT_REWARD" and not n_choices and combat_rewards:
            n_choices = len(combat_rewards)

        for idx in range(min(n_choices, MAX_CHOICES)):
            if st_name == "COMBAT_REWARD" and potions_full:
                is_potion = False
                if idx < len(combat_rewards):
                    rt = getattr(combat_rewards[idx], "reward_type", None)
                    is_potion = rt is not None and rt.name == "POTION"
                elif idx < len(choice_list):
                    is_potion = "potion" in str(choice_list[idx]).lower()
                if is_potion:
                    continue
            mask[_CHOOSE_START + idx] = True

    if proceed_avail:
        mask[_PROCEED] = True

    if cancel_avail:
        mask[_LEAVE] = True

    if not mask.any():
        mask[_NOOP] = True

    return mask


# ---------------------------------------------------------------------------
# Action -> SpireComm translation
# ---------------------------------------------------------------------------
def flat_action_to_spire_action(action_id: int, gs: Any) -> Action:
    """Convert a flat action index to a SpireComm Action."""
    hand = list(getattr(gs, "hand", []) or [])
    monsters = list(getattr(gs, "monsters", []) or [])
    potions = list(getattr(gs, "potions", []) or [])
    alive = living_monsters(monsters)

    if _PLAY_TARGETED_START <= action_id < _PLAY_UNTARGETED_START:
        rel = action_id - _PLAY_TARGETED_START
        card_idx = rel // MAX_MONSTERS
        monster_idx = rel % MAX_MONSTERS
        if card_idx < len(hand) and monster_idx < len(alive):
            return PlayCardAction(card=hand[card_idx], target_monster=alive[monster_idx])
        return Action("state")

    if _PLAY_UNTARGETED_START <= action_id < _END_TURN:
        card_idx = action_id - _PLAY_UNTARGETED_START
        if card_idx < len(hand):
            return PlayCardAction(card=hand[card_idx])
        return Action("state")

    if action_id == _END_TURN:
        return Action("end")

    if _POTION_TARGETED_START <= action_id < _POTION_UNTARGETED_START:
        rel = action_id - _POTION_TARGETED_START
        pot_idx = rel // MAX_MONSTERS
        monster_idx = rel % MAX_MONSTERS
        if pot_idx < len(potions) and monster_idx < len(alive):
            return PotionAction(use=True, potion=potions[pot_idx], target_monster=alive[monster_idx])
        return Action("state")

    if _POTION_UNTARGETED_START <= action_id < _CHOOSE_START:
        pot_idx = action_id - _POTION_UNTARGETED_START
        if pot_idx < len(potions):
            return PotionAction(use=True, potion=potions[pot_idx])
        return Action("state")

    if _CHOOSE_START <= action_id < _PROCEED:
        choice_idx = action_id - _CHOOSE_START
        return ChooseAction(choice_index=choice_idx)

    if action_id == _PROCEED:
        return Action("proceed")

    if action_id == _LEAVE:
        return Action("leave")

    return Action("state")


# ---------------------------------------------------------------------------
# Reward tracker
# ---------------------------------------------------------------------------
SPAWNER_IDS = frozenset({
    "GremlinLeader", "Reptomancer", "BronzeAutomaton",
})


class RewardTracker:
    """Dense reward shaping + dominant terminal bonuses."""

    def __init__(self):
        self.last_gold = 0
        self.last_hp = 0
        self.last_max_hp = 0
        self.last_deck_size = 0
        self.last_relics = 0
        self.last_floor = 0
        self.last_in_combat = False
        self.last_enemy_hp: Optional[int] = None
        self.last_alive = 0
        self._last_act = 0
        self._last_spawner_hp: Dict[str, int] = {}

    def _enemy_stats(self, gs: Any) -> Tuple[Optional[int], int]:
        total_hp = 0
        alive = 0
        for m in (getattr(gs, "monsters", []) or []):
            if getattr(m, "is_gone", False):
                continue
            hp = int(getattr(m, "current_hp", 0) or 0)
            if hp > 0 or getattr(m, "half_dead", False):
                alive += 1
                total_hp += max(0, hp)
        return (total_hp if alive else None), alive

    def _spawner_hp_map(self, gs: Any) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for m in (getattr(gs, "monsters", []) or []):
            mid = str(getattr(m, "monster_id", "") or "")
            if mid not in SPAWNER_IDS:
                continue
            if getattr(m, "is_gone", False):
                result[mid] = 0
                continue
            result[mid] = max(0, int(getattr(m, "current_hp", 0) or 0))
        return result

    def reset(self, gs: Any) -> None:
        self.last_gold = int(getattr(gs, "gold", 0) or 0)
        self.last_hp = int(getattr(gs, "current_hp", 0) or 0)
        self.last_max_hp = int(getattr(gs, "max_hp", 0) or 0)
        self.last_deck_size = len(getattr(gs, "deck", []) or [])
        self.last_relics = len(getattr(gs, "relics", []) or [])
        self.last_floor = int(getattr(gs, "floor", 0) or 0)
        self.last_in_combat = bool(getattr(gs, "in_combat", False))
        self._last_act = int(getattr(gs, "act", 0) or 0)
        if self.last_in_combat:
            self.last_enemy_hp, self.last_alive = self._enemy_stats(gs)
            self._last_spawner_hp = self._spawner_hp_map(gs)
        else:
            self.last_enemy_hp, self.last_alive = None, 0
            self._last_spawner_hp = {}

    def compute(self, gs: Any, terminated: bool, victory: bool) -> float:
        reward = 0.0

        gold = int(getattr(gs, "gold", 0) or 0)
        hp = int(getattr(gs, "current_hp", 0) or 0)
        deck_size = len(getattr(gs, "deck", []) or [])
        relic_count = len(getattr(gs, "relics", []) or [])
        floor_num = int(getattr(gs, "floor", 0) or 0)
        in_combat = bool(getattr(gs, "in_combat", False))

        reward += (gold - self.last_gold) * 0.01
        reward += (relic_count - self.last_relics) * 1.0
        if deck_size < self.last_deck_size:
            reward += (self.last_deck_size - deck_size) * 0.2

        if floor_num > self.last_floor:
            reward += 0.5

        if in_combat:
            reward -= max(0, self.last_hp - hp) * 0.05
            e_hp, alive = self._enemy_stats(gs)
            if e_hp is not None and self.last_enemy_hp is not None:
                reward += max(0, self.last_enemy_hp - e_hp) * 0.02
            reward += max(0, self.last_alive - alive) * 0.5

            spawner_hp = self._spawner_hp_map(gs)
            for mid, prev_hp in self._last_spawner_hp.items():
                cur_hp = spawner_hp.get(mid, 0)
                dmg = max(0, prev_hp - cur_hp)
                if dmg > 0:
                    reward += dmg * 0.03
                if prev_hp > 0 and cur_hp <= 0:
                    reward += 2.0

        if terminated:
            if victory:
                reward += 50.0
            else:
                reward -= 15.0
                reward += floor_num * 0.3

        act = int(getattr(gs, "act", 0) or 0)
        if act > self._last_act and act > 1:
            reward += 10.0
        self._last_act = act

        self.last_gold = gold
        self.last_hp = hp
        self.last_max_hp = int(getattr(gs, "max_hp", 0) or 0)
        self.last_deck_size = deck_size
        self.last_relics = relic_count
        self.last_floor = floor_num
        self.last_in_combat = in_combat
        if in_combat:
            self.last_enemy_hp, self.last_alive = self._enemy_stats(gs)
            self._last_spawner_hp = self._spawner_hp_map(gs)
        else:
            self.last_enemy_hp, self.last_alive = None, 0
            self._last_spawner_hp = {}

        return reward


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
_TERMINAL_SCREENS = {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS", "GAME_OVER_SCREEN"}


class STSEnv(gym.Env):
    """
    Slay the Spire as a Gymnasium env with action masking.

    Uses two synchronization events:
      _state_event  — set on every state callback (for reset)
      _step_event   — set only after the env's queued action gets a response (for step)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        coordinator: Coordinator,
        step_timeout: float = 25.0,
        reset_timeout: float = 45.0,
        max_episode_steps: int = 10_000,
    ):
        super().__init__()
        self.coordinator = coordinator
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._game_state: Any = None
        self._action_queue: deque = deque()
        self._lock = threading.Lock()

        self._state_event = threading.Event()
        self._step_event = threading.Event()
        self._action_pending = False

        self._step_count = 0
        self._reward_tracker = RewardTracker()
        self._current_mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        self._current_mask[_NOOP] = True

        self.coordinator.register_state_change_callback(self._on_state_change)
        self.coordinator.register_command_error_callback(self._on_error)

    def action_masks(self) -> np.ndarray:
        return self._current_mask

    def _on_error(self, error: str) -> Action:
        """Called by Coordinator when CommunicationMod returns an error.

        Without this, the coordinator thread would crash on NoneType callback
        and the env would hang forever.  We just log and re-poll state so the
        agent can try a different action.
        """
        _log(f"COMM ERROR: {error}")
        if self._action_pending:
            self._action_pending = False
            self._step_event.set()
        return Action("state")

    def _on_state_change(self, game_state: Any) -> Action:
        """Called by Coordinator thread on every state update."""
        self._game_state = game_state

        # Always signal state_event (for reset)
        self._state_event.set()

        with self._lock:
            if self._action_queue:
                # Pick up the action — mark pending so we wait for its response
                self._action_pending = True
                return self._action_queue.popleft()

        if self._action_pending:
            # This state is the response to our action — signal step()
            self._action_pending = False
            self._step_event.set()

        return Action("state")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._action_pending = False

        with self._lock:
            self._action_queue.clear()

        self._state_event.clear()
        with self._lock:
            self._action_queue.append(Action("state"))

        if not self._state_event.wait(timeout=self.reset_timeout):
            _log("RESET TIMEOUT")
            raise RuntimeError("Reset timeout: no game state received")
        if self._game_state is None:
            raise RuntimeError("No game state after reset")

        _log(f"RESET OK — floor={getattr(self._game_state, 'floor', '?')}")
        self._reward_tracker.reset(self._game_state)
        obs = encode_game_state(self._game_state)
        self._current_mask = compute_action_mask(self._game_state)

        return obs, {"screen": _screen_name(self._game_state)}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._game_state is None:
            raise RuntimeError("Env not initialized; call reset() first")

        spire_action = flat_action_to_spire_action(action, self._game_state)

        # Clear step event and queue our action
        self._step_event.clear()
        with self._lock:
            self._action_queue.append(spire_action)

        # Wait for the post-action state (not a stale polling response)
        if not self._step_event.wait(timeout=self.step_timeout):
            _log(f"STEP TIMEOUT action={action}")
            obs = encode_game_state(self._game_state)
            self._current_mask = compute_action_mask(self._game_state)
            return obs, 0.0, False, True, {"timeout": True}

        self._step_count += 1
        screen = _screen_name(self._game_state)
        terminated = screen in _TERMINAL_SCREENS

        victory = False
        if terminated:
            scr_obj = getattr(self._game_state, "screen", None)
            victory = bool(getattr(scr_obj, "victory", False)) or screen in {"COMPLETE", "VICTORY"}

        reward = self._reward_tracker.compute(self._game_state, terminated, victory)
        obs = encode_game_state(self._game_state)
        self._current_mask = compute_action_mask(self._game_state)
        truncated = self._step_count >= self.max_episode_steps

        if self._step_count % 50 == 0:
            _log(f"STEP {self._step_count}: screen={screen} hp={getattr(self._game_state, 'current_hp', '?')} r={reward:.3f}")

        return obs, reward, terminated, truncated, {
            "screen": screen,
            "step": self._step_count,
            "victory": victory,
        }

    def close(self) -> None:
        pass


def _screen_name(gs: Any) -> str:
    st = getattr(gs, "screen_type", None)
    if st is None:
        return "NONE"
    name = getattr(st, "name", st)
    return str(name) if name else "NONE"
