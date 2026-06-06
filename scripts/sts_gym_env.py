"""
sts_gym_env.py — Action space, masking, and rewards for Slay the Spire RL.

  - Observation via obs_encoder (cards, monsters, powers, relics, potions, map)
  - Discrete(134) action space with per-card, per-target control
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
from screen_handler import event_choice_targets as get_event_choice_targets
from screen_handler import event_choice_for_slot, note_rest_choice
from game_data import CARD_MECHANICS


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
def _norm_label(value) -> str:
    return str(value or "").lower().strip()


def _choice_is_locked(label: str) -> bool:
    label = _norm_label(label)
    return (
        "locked" in label
        or "disabled" in label
        or "unavailable" in label
        or "requires" in label
    )


def _has_relic(gs: Any, names: set[str]) -> bool:
    for relic in list(getattr(gs, "relics", []) or []):
        candidates = {
            _norm_label(getattr(relic, "name", "")),
            _norm_label(getattr(relic, "relic_id", "")),
            _norm_label(getattr(relic, "id", "")),
        }
        candidates = {c.replace("_", " ").replace("-", " ") for c in candidates if c}
        if any(c in names for c in candidates):
            return True
    return False


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
    available_commands = set(getattr(gs, "available_commands", []) or [])
    if available_commands:
        proceed_avail = bool({"proceed", "confirm"} & available_commands)
        cancel_avail = bool({"cancel", "leave", "return", "skip"} & available_commands)

    alive = living_monsters(monsters)
    n_alive = len(alive)

    if in_combat and play_avail:
        for i, card in enumerate(hand[:MAX_HAND]):
            if not getattr(card, "is_playable", False):
                continue
            if getattr(card, "has_target", False):
                for j in range(min(n_alive, MAX_MONSTERS)):
                    mask[_PLAY_TARGETED_START + i * MAX_MONSTERS + j] = True
            else:
                mask[_PLAY_UNTARGETED_START + i] = True

    if in_combat and end_avail:
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

    n_choices = len(choice_list)
    event_choice_targets: Dict[int, int] = {}
    if not grid_confirmed:
        potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())

        # For COMBAT_REWARD, screen.rewards is authoritative — choice_list
        # may be empty even when rewards are available.
        combat_rewards = (list(getattr(scr, "rewards", []) or [])
                          if st_name == "COMBAT_REWARD" and scr else [])
        if st_name == "COMBAT_REWARD" and not n_choices and combat_rewards:
            n_choices = len(combat_rewards)
        # GRID purge/upgrade is now RL-controlled. If choice_list is empty,
        # fall back to the grid cards so the choices get masked legal.
        if st_name == "GRID" and not n_choices and scr is not None:
            grid_cards = list(getattr(scr, "cards", []) or [])
            if grid_cards:
                n_choices = len(grid_cards)
        if st_name == "EVENT":
            event_choice_targets = get_event_choice_targets(gs, max_choices=MAX_CHOICES)
            if event_choice_targets:
                n_choices = max(event_choice_targets) + 1

        for idx in range(min(n_choices, MAX_CHOICES)):
            if st_name == "EVENT" and idx not in event_choice_targets:
                continue
            if st_name == "REST" and idx < len(choice_list):
                label = _norm_label(choice_list[idx])
                if _choice_is_locked(label):
                    continue
                if "rest" in label and _has_relic(gs, {"coffee dripper"}):
                    continue
                if "smith" in label and _has_relic(gs, {"fusion hammer"}):
                    continue
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

    suppress_proceed = (
        (st_name == "MAP" and n_choices > 0)
        or (st_name == "COMBAT_REWARD" and n_choices > 0)
        or (st_name == "BOSS_REWARD" and n_choices > 0)
        or (st_name == "EVENT" and bool(event_choice_targets))
    )
    suppress_cancel = (
        # Closing the map after combat bounces back to the reward screen,
        # creating a proceed -> map -> return loop. Autonomous agents should
        # choose a node once map choices are present.
        (st_name == "MAP" and n_choices > 0)
        # Boss relics are mandatory. Leaving returns to the boss chest, which
        # reopens the relic screen and can trap the agent forever.
        or (st_name == "BOSS_REWARD" and n_choices > 0)
        # Event "leave" buttons are often event choices, not the generic leave
        # command. Prefer choose <index> while enabled event options exist.
        or (st_name == "EVENT" and bool(event_choice_targets))
    )

    if proceed_avail and not suppress_proceed:
        mask[_PROCEED] = True

    if cancel_avail and not suppress_cancel:
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
        screen_type = getattr(gs, "screen_type", None)
        st_name = getattr(screen_type, "name", "") if screen_type else ""
        scr = getattr(gs, "screen", None)
        choice_list = list(getattr(gs, "choice_list", []) or [])
        if st_name == "REST" and choice_idx < len(choice_list):
            note_rest_choice(gs, choice_list[choice_idx])
        if st_name == "EVENT" and scr is not None:
            event_choice_targets = get_event_choice_targets(gs, max_choices=MAX_CHOICES)
            if choice_idx in event_choice_targets:
                return ChooseAction(choice_index=event_choice_targets[choice_idx])
        return ChooseAction(choice_index=choice_idx)

    if action_id == _PROCEED:
        screen_type = getattr(gs, "screen_type", None)
        st_name = getattr(screen_type, "name", "") if screen_type else ""
        if st_name == "EVENT":
            targets = get_event_choice_targets(gs, max_choices=MAX_CHOICES)
            if targets:
                _slot, choice_idx = event_choice_for_slot(gs, 0)
                return ChooseAction(choice_index=choice_idx)
        commands = set(getattr(gs, "available_commands", []) or [])
        if "confirm" in commands and "proceed" not in commands:
            return Action("confirm")
        return Action("proceed")

    if action_id == _LEAVE:
        screen_type = getattr(gs, "screen_type", None)
        st_name = getattr(screen_type, "name", "") if screen_type else ""
        if st_name == "EVENT":
            targets = get_event_choice_targets(gs, max_choices=MAX_CHOICES)
            if targets:
                _slot, choice_idx = event_choice_for_slot(gs, 0)
                return ChooseAction(choice_index=choice_idx)
        commands = set(getattr(gs, "available_commands", []) or [])
        for cmd in ("cancel", "leave", "return", "skip"):
            if cmd in commands:
                return Action(cmd)
        return Action("leave")

    return Action("state")


# ---------------------------------------------------------------------------
# Reward tracker
# ---------------------------------------------------------------------------
SPAWNER_IDS = frozenset({
    "GremlinLeader", "Reptomancer", "BronzeAutomaton",
})
PRIORITY_MONSTER_WEIGHTS = {
    "GremlinWizard": 0.12,
    "SnakeDagger": 0.12,
    "Dagger": 0.12,
    "SlaverRed": 0.10,
    "SlaverBlue": 0.08,
    "SlaverBoss": 0.06,
    "Exploder": 0.10,
    "GremlinNob": 0.06,
    "BookOfStabbing": 0.05,
    "Byrd": 0.04,
    "Chosen": 0.04,
    "SnakePlant": 0.05,
    "ShelledParasite": 0.04,
    "Mugger": 0.04,
    "Looter": 0.04,
    "GremlinLeader": 0.04,
    "Reptomancer": 0.04,
    "BronzeAutomaton": 0.03,
    "Donu": 0.08,
    "TorchHead": 0.06,
    "BronzeOrb": 0.08,
}
PRIORITY_KILL_BONUS = {
    "GremlinWizard": 4.0,
    "SnakeDagger": 4.0,
    "Dagger": 4.0,
    "SlaverRed": 4.0,
    "SlaverBlue": 2.5,
    "SlaverBoss": 2.5,
    "Exploder": 3.0,
    "GremlinNob": 2.5,
    "BookOfStabbing": 2.5,
    "Byrd": 1.5,
    "Chosen": 1.5,
    "SnakePlant": 2.0,
    "ShelledParasite": 1.5,
    "Mugger": 1.5,
    "Looter": 1.5,
    "GremlinLeader": 2.0,
    "Reptomancer": 2.0,
    "BronzeAutomaton": 1.5,
    "Donu": 5.0,
    "TorchHead": 2.0,
    "BronzeOrb": 3.0,
}

HP_LOSS_PENALTY = 0.08
ENEMY_DAMAGE_REWARD = 0.015
MONSTER_KILL_REWARD = 0.75
FLOOR_ADVANCE_BASE = 0.50
FLOOR_ADVANCE_HP_BONUS = 0.25
# Act 1 elites are high-value (relics scale Ironclad), Act 2 are dangerous, Act 3 should be free
ELITE_WIN_BONUS_ACT1 = 4.0
ELITE_WIN_BONUS_ACT2 = 2.0
ELITE_WIN_BONUS_ACT3 = 1.0
ACT_ADVANCE_REWARD = 12.0
VICTORY_REWARD = 60.0
DEFEAT_PENALTY = 25.0
DEFEAT_FLOOR_OFFSET = 0.25
MAX_HP_GAIN_REWARD = 0.10
REST_HEAL_PER_HP = 0.025

# Deck-quality reward. Phi(deck) = mean card quality with an upgrade boost.
#
# OPTION B EXPERIMENT (2026-06): the original shaping was potential-based,
# reward += 1.0 * (Phi(s') - Phi(s)). A 200-game greedy eval found the 717-d
# deck-vector model statistically identical to the 585-d baseline, and an
# ablation proved the model DOES read the deck (zeroing the deck-vec flips 30%
# of removal/upgrade picks, randomizing it 80%) — it just makes heuristic-quality
# choices. Two reasons it never beat baseline: (1) potential-based shaping is
# policy-invariant *by design* (Ng et al. 1999) and telescopes to Phi_final -
# Phi_init, so it cannot push toward a *better* deck, only a deck-conditional
# one; (2) it was ~2% of the reward. This replaces it with a DIRECT, asymmetric,
# capped reward: gains are amplified and losses penalized less (breaks
# policy-invariance), giving a real incentive to improve the deck, while a
# per-episode cap stops the agent farming deck quality instead of winning.
DECK_QUALITY_GAIN_COEF = 5.0   # amplified reward on per-step Phi increases
DECK_QUALITY_LOSS_COEF = 2.0   # milder penalty on Phi decreases (asymmetric -> non-invariant)
DECK_REWARD_EPISODE_CAP = 8.0  # cap cumulative deck gain per run so it can't dominate winning
UPGRADE_QUALITY_BONUS = 0.3    # u: an upgraded card counts as quality*(1+u)


def _deck_potential(deck) -> float:
    """Phi(deck): mean card quality with an upgrade boost (0.0 for empty deck)."""
    if not deck:
        return 0.0
    total = 0.0
    for c in deck:
        mech = CARD_MECHANICS.get(str(getattr(c, "card_id", "") or ""))
        q = mech[4] if mech is not None else 0.0
        upg = 1 if int(getattr(c, "upgrades", 0) or 0) > 0 else 0
        total += q * (1.0 + UPGRADE_QUALITY_BONUS * upg)
    return total / len(deck)

# ---------------------------------------------------------------------------
# Boss fight reward shaping
# ---------------------------------------------------------------------------
ACT1_BOSS_IDS = frozenset({"Hexaghost", "TheGuardian", "SlimeBoss"})
ACT2_BOSS_IDS = frozenset({"BronzeAutomaton", "TheCollector", "Champ"})
ACT3_BOSS_IDS = frozenset({"AwakenedOne", "Donu", "Deca", "TimeEater"})
ALL_BOSS_IDS = ACT1_BOSS_IDS | ACT2_BOSS_IDS | ACT3_BOSS_IDS
BOSS_KILL_REWARD = 8.0
RETALIATION_PENALTY = 0.25
GUARDIAN_OPEN_DMG_BONUS = 0.03
BIG_HIT_THRESHOLD = 20
BIG_HIT_EXTRA_PENALTY = 0.10

_RETALIATION_POWERS = frozenset({"thorns", "sharphide"})


def _get_boss(gs: Any) -> Any:
    for m in (getattr(gs, "monsters", []) or []):
        mid = str(getattr(m, "monster_id", "") or "")
        if mid in ALL_BOSS_IDS and not getattr(m, "is_gone", False):
            hp = int(getattr(m, "current_hp", 0) or 0)
            if hp > 0 or getattr(m, "half_dead", False):
                return m
    return None


def _monster_has_retaliation(m: Any) -> int:
    for p in (getattr(m, "powers", []) or []):
        pid = str(getattr(p, "power_id", "") or "").lower().replace(" ", "").replace("_", "")
        if pid in _RETALIATION_POWERS:
            return max(0, int(getattr(p, "amount", 0) or 0))
    return 0


def _monster_strength(m: Any) -> int:
    for p in (getattr(m, "powers", []) or []):
        pid = str(getattr(p, "power_id", "") or "").lower().replace(" ", "").replace("_", "")
        if pid == "strength":
            return int(getattr(p, "amount", 0) or 0)
    return 0


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
        self._last_priority_hp: Dict[str, int] = {}
        self._boss_fight_start_hp = 0
        self._boss_id = ""
        self._boss_last_hp = 0
        self._boss_max_hp = 0
        self._in_elite_fight = False
        self.last_deck_potential = 0.0
        self.deck_reward_total = 0.0

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

    def _priority_hp_map(self, gs: Any) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for idx, m in enumerate(getattr(gs, "monsters", []) or []):
            mid = str(getattr(m, "monster_id", "") or "")
            if mid not in PRIORITY_MONSTER_WEIGHTS:
                continue
            key = f"{idx}:{mid}"
            if getattr(m, "is_gone", False):
                result[key] = 0
                continue
            result[key] = max(0, int(getattr(m, "current_hp", 0) or 0))
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
        self._boss_fight_start_hp = 0
        self._boss_id = ""
        self._boss_last_hp = 0
        self._boss_max_hp = 0
        self._in_elite_fight = False
        self.last_deck_potential = _deck_potential(getattr(gs, "deck", []) or [])
        self.deck_reward_total = 0.0
        if self.last_in_combat:
            self.last_enemy_hp, self.last_alive = self._enemy_stats(gs)
            self._last_priority_hp = self._priority_hp_map(gs)
            boss = _get_boss(gs)
            if boss is not None:
                self._boss_fight_start_hp = self.last_hp
                self._boss_id = str(getattr(boss, "monster_id", "") or "")
                self._boss_last_hp = int(getattr(boss, "current_hp", 0) or 0)
                self._boss_max_hp = int(getattr(boss, "max_hp", 0) or 0)
        else:
            self.last_enemy_hp, self.last_alive = None, 0
            self._last_priority_hp = {}

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
        reward += max(0, int(getattr(gs, "max_hp", 0) or 0) - self.last_max_hp) * MAX_HP_GAIN_REWARD
        # Direct, asymmetric, capped deck-quality reward (Option B). Reward Phi
        # rising (good removals/drafts/upgrades) with an amplified gain coef, and
        # penalize it falling at a smaller coef so the net signal is non-invariant
        # and pushes toward a *better* deck. Cap cumulative gain per run so the
        # agent can't farm deck quality in place of winning.
        deck_potential = _deck_potential(getattr(gs, "deck", []) or [])
        d_phi = deck_potential - self.last_deck_potential
        if d_phi >= 0.0:
            gain = min(DECK_QUALITY_GAIN_COEF * d_phi,
                       max(0.0, DECK_REWARD_EPISODE_CAP - self.deck_reward_total))
            self.deck_reward_total += gain
            reward += gain
        else:
            reward += DECK_QUALITY_LOSS_COEF * d_phi
        self.last_deck_potential = deck_potential

        if floor_num > self.last_floor:
            max_hp = max(1, int(getattr(gs, "max_hp", 1) or 1))
            hp_ratio = hp / max_hp
            reward += FLOOR_ADVANCE_BASE + FLOOR_ADVANCE_HP_BONUS * hp_ratio
            if self._in_elite_fight:
                act = int(getattr(gs, "act", 1) or 1)
                if act == 1:
                    reward += ELITE_WIN_BONUS_ACT1
                elif act == 2:
                    reward += ELITE_WIN_BONUS_ACT2
                else:
                    reward += ELITE_WIN_BONUS_ACT3
                self._in_elite_fight = False

        if not in_combat:
            hp_recovered = max(0, hp - self.last_hp)
            if hp_recovered > 0:
                max_hp = max(1, int(getattr(gs, "max_hp", 1) or 1))
                urgency = 1.0 - (self.last_hp / max_hp)
                reward += hp_recovered * REST_HEAL_PER_HP * urgency

        if in_combat and not self.last_in_combat:
            room = str(getattr(gs, "room_type", "") or "")
            if "Elite" in room:
                self._in_elite_fight = True

        if in_combat:
            hp_lost = max(0, self.last_hp - hp)
            reward -= hp_lost * HP_LOSS_PENALTY
            e_hp, alive = self._enemy_stats(gs)
            if e_hp is not None and self.last_enemy_hp is not None:
                reward += max(0, self.last_enemy_hp - e_hp) * ENEMY_DAMAGE_REWARD
            reward += max(0, self.last_alive - alive) * MONSTER_KILL_REWARD

            priority_hp = self._priority_hp_map(gs)
            for key, prev_hp in self._last_priority_hp.items():
                mid = key.split(":", 1)[1] if ":" in key else key
                cur_hp = priority_hp.get(key, 0)
                dmg = max(0, prev_hp - cur_hp)
                if dmg > 0:
                    reward += dmg * PRIORITY_MONSTER_WEIGHTS.get(mid, 0.03)
                if prev_hp > 0 and cur_hp <= 0:
                    reward += PRIORITY_KILL_BONUS.get(mid, 1.0)

            # --- Boss fight shaping ---
            boss = _get_boss(gs)
            if boss is not None:
                boss_id = str(getattr(boss, "monster_id", "") or "")
                if not self._boss_id:
                    self._boss_fight_start_hp = self.last_hp
                    self._boss_id = boss_id
                    self._boss_last_hp = int(getattr(boss, "current_hp", 0) or 0)
                    self._boss_max_hp = int(getattr(boss, "max_hp", 0) or 0)

                boss_hp = int(getattr(boss, "current_hp", 0) or 0)

                # -- Act 1 --

                # Guardian: two-phase cycle. Defensive mode has Sharp Hide
                # (thorns). Penalize attacking into thorns; bonus for
                # dealing damage when Sharp Hide is down (offensive mode).
                if boss_id == "TheGuardian":
                    ret = _monster_has_retaliation(boss)
                    if ret > 0 and hp_lost > 0:
                        reward -= hp_lost * RETALIATION_PENALTY
                    boss_dmg = max(0, self._boss_last_hp - boss_hp)
                    if ret == 0 and boss_dmg > 0:
                        reward += boss_dmg * GUARDIAN_OPEN_DMG_BONUS

                # Hexaghost: Divider (turn 2, scales with player HP) and
                # Inferno (every ~7 turns, 6-hit multi-attack) deal huge
                # damage. Extra penalty for big hits teaches blocking.
                if boss_id == "Hexaghost" and hp_lost >= BIG_HIT_THRESHOLD:
                    reward -= hp_lost * BIG_HIT_EXTRA_PENALTY

                # Slime Boss: splits at 50% HP; spawned slimes inherit
                # current HP.  Reward overkill past the 50% threshold —
                # the further below 50% the boss is when it splits, the
                # weaker the slimes.
                if boss_id == "SlimeBoss" and self._boss_max_hp > 0:
                    half = self._boss_max_hp / 2.0
                    if self._boss_last_hp > 0 and boss_hp <= 0:
                        overkill = max(0.0, half - boss_hp)
                        reward += min(overkill * 0.05, 4.0)

                # -- Act 2 --

                # Bronze Automaton: Hyper Beam deals 45 damage, then
                # Stunned (does nothing). Penalize big hits (Hyper Beam);
                # the base damage reward during Stunned turn is enough
                # incentive to attack then.
                if boss_id == "BronzeAutomaton" and hp_lost >= BIG_HIT_THRESHOLD:
                    reward -= hp_lost * BIG_HIT_EXTRA_PENALTY

                # Champ: at <50% HP, enrages (+9 Str, clears debuffs)
                # then uses Execute (10x2). Penalize HP loss when Champ
                # has high strength (post-enrage).
                if boss_id == "Champ":
                    str_val = _monster_strength(boss)
                    if str_val >= 6 and hp_lost > 0:
                        reward -= hp_lost * BIG_HIT_EXTRA_PENALTY

                # -- Act 3 --

                # Donu & Deca: Donu buffs +3 Str to both every other
                # turn. Reward damage to Donu specifically via priority.
                # (Donu is added to PRIORITY_MONSTER_WEIGHTS below.)

                self._boss_last_hp = boss_hp

        if terminated:
            if victory:
                reward += VICTORY_REWARD
            else:
                reward -= DEFEAT_PENALTY
                reward += floor_num * DEFEAT_FLOOR_OFFSET

        act = int(getattr(gs, "act", 0) or 0)
        if act > self._last_act and act > 1:
            reward += ACT_ADVANCE_REWARD
            # Boss kill bonus: reward HP preserved through the boss fight
            if self._boss_id and hp > 0 and self._boss_fight_start_hp > 0:
                hp_ratio = hp / max(1, self._boss_fight_start_hp)
                reward += BOSS_KILL_REWARD + hp_ratio * BOSS_KILL_REWARD
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
            self._last_priority_hp = self._priority_hp_map(gs)
        else:
            self.last_enemy_hp, self.last_alive = None, 0
            self._last_priority_hp = {}
            self._boss_id = ""
            self._boss_fight_start_hp = 0
            self._boss_last_hp = 0
            self._boss_max_hp = 0

        return reward


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
_TERMINAL_SCREENS = {"GAME_OVER", "GAME_OVER_SCREEN"}


def is_terminal_state(gs: Any) -> bool:
    """Return True only for real run-ending screens.

    CommunicationMod's COMPLETE screen is a transient room/action completion
    state, not a completed Slay the Spire run.
    """
    return _screen_name(gs) in _TERMINAL_SCREENS


def is_victory_state(gs: Any) -> bool:
    """Return True when the real game-over screen reports a victory."""
    if not is_terminal_state(gs):
        return False
    scr_obj = getattr(gs, "screen", None)
    return bool(getattr(scr_obj, "victory", False))


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
        terminated = is_terminal_state(self._game_state)
        victory = is_victory_state(self._game_state)

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
