"""
behavior_clone.py — Collect heuristic demonstrations and pre-train the PPO network.

Runs inside Communication Mod (stdin/stdout protocol). The heuristic agent plays
games while we encode states with obs_encoder and record (obs, action_id, mask)
tuples. After collecting enough data, trains the PPO network via supervised
cross-entropy loss to imitate the heuristic, then saves the warm-started model.

Covers the full RL decision surface: combat card play, map navigation, card
drafting, campfire rest/smith, event choices, shop purchases, boss relic picks,
forced discard, grid selection, and combat reward priority.

Usage (in CommunicationMod config.properties):
  command=python behavior_clone.py --games 150 --save models/ppo_sts.pt
"""

from __future__ import annotations

import sys
import os

_real_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

from spirecomm_patches import apply_all as _apply_spirecomm_patches
_apply_spirecomm_patches(_real_stdout)

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
import random
import csv
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "bc_debug.log")
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

def log(msg: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
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
    NUM_ACTIONS, compute_action_mask,
    _PLAY_TARGETED_START, _PLAY_UNTARGETED_START, _END_TURN,
    _POTION_TARGETED_START, _POTION_UNTARGETED_START,
    _CHOOSE_START, _PROCEED, _LEAVE, _NOOP,
    MAX_HAND, MAX_MONSTERS, MAX_POTIONS,
    SPAWNER_IDS,
    is_terminal_state, is_victory_state,
)

from bc_stats import append_bc_stats
from fight_tracker import FightTracker
from game_data import POTION_EFFECTS
from game_data import CARD_MECHANICS
from screen_handler import (
    GOOD_CARDS, OK_CARDS, JUNK_CARDS,
    pick_card_reward, pick_combat_reward_obj, pick_combat_reward_str,
    pick_rest, pick_event_slot_and_choice,
    pick_boss_relic, pick_hand_select,
    pick_grid_card, _pick_grid_upgrade, _is_matching_grid, _pick_grid_match,
    _pick_from_unselected, _looks_like_smith_grid,
    event_choice_targets,
    pick_map,
    pick_shop_item,
    recover_from_command_error,
)

log("Imports done")


# ---------------------------------------------------------------------------
# Shop visit tracking (prevents SHOP_ROOM ↔ SHOP_SCREEN infinite loop)
# ---------------------------------------------------------------------------
_visited_shop_floors: set = set()
_skipped_card_reward_keys: set[tuple[int, int]] = set()


# ---------------------------------------------------------------------------
# Heuristic action picker
# ---------------------------------------------------------------------------
def _screen_name(gs) -> str:
    st = getattr(gs, "screen_type", None)
    name = getattr(st, "name", st) if st is not None else "NONE"
    return str(name) if name else "NONE"


def _screen_key(gs) -> tuple[int, int]:
    return (
        int(getattr(gs, "act", 0) or 0),
        int(getattr(gs, "floor", -1) or -1),
    )


URGENT_MONSTERS = (
    "GremlinWizard", "SnakeDagger", "Dagger", "SlaverRed", "Exploder",
    "SlaverBlue", "SlaverBoss", "GremlinNob", "BookOfStabbing",
    "SnakePlant", "Chosen", "Byrd", "ShelledParasite", "Mugger", "Looter",
)


def _norm(text: str) -> str:
    return (
        str(text or "").lower()
        .replace(" ", "").replace("_", "").replace("-", "").replace("'", "")
    )


_CARD_MECHANICS_NORM = {
    _norm(k): v for k, v in CARD_MECHANICS.items()
}
_POTION_EFFECTS_NORM = {
    _norm(k): v for k, v in POTION_EFFECTS.items()
}


def _monster_id(m) -> str:
    return str(getattr(m, "monster_id", "") or getattr(m, "name", "") or "")


def _room_type(gs) -> str:
    return str(getattr(gs, "room_type", "") or "")


def _is_elite_or_boss(gs) -> bool:
    room = _room_type(gs)
    if "Elite" in room or "Boss" in room:
        return True
    alive = living_monsters(getattr(gs, "monsters", []) or [])
    return any(_monster_id(m) in {
        "GremlinNob", "Lagavulin", "GremlinLeader", "SlaverBoss",
        "BookOfStabbing", "Reptomancer", "Nemesis", "GiantHead",
    } for m in alive)


def _potion_name(pot) -> str:
    return str(
        getattr(pot, "name", None)
        or getattr(pot, "potion_id", None)
        or ""
    )


def _potion_effects(pot) -> tuple:
    name = _potion_name(pot)
    return _POTION_EFFECTS_NORM.get(_norm(name), (0, 0, 0, 0, 0))


def _monster_damage(m) -> int:
    dmg = int(getattr(m, "move_base_damage", 0) or 0)
    hits = int(getattr(m, "move_hits", 1) or 1)
    return max(0, dmg) * max(1, hits)


def _is_attacking(m) -> bool:
    intent = str(getattr(getattr(m, "intent", None), "name",
                         getattr(m, "intent", "")) or "").upper()
    return "ATTACK" in intent or _monster_damage(m) > 0


def estimate_incoming(monsters) -> int:
    total = 0
    for m in (monsters or []):
        if getattr(m, "is_gone", False):
            continue
        hp = int(getattr(m, "current_hp", 0) or 0)
        if hp <= 0 and not getattr(m, "half_dead", False):
            continue
        dmg = int(getattr(m, "move_base_damage", 0) or 0)
        hits = int(getattr(m, "move_hits", 1) or 1)
        total += max(0, dmg) * max(1, hits)
    return total


def _card_mechanics(card) -> tuple:
    cid = str(getattr(card, "card_id", "") or "")
    name = str(getattr(card, "name", "") or "")
    return (
        _CARD_MECHANICS_NORM.get(_norm(cid))
        or _CARD_MECHANICS_NORM.get(_norm(name))
        or (0, False, False, 0.0, 0.0)
    )


def score_card(card, incoming: int, gs=None, target=None) -> float:
    name = (getattr(card, "name", "") or "").lower()
    cost = int(getattr(card, "cost", 1) or 1)
    ctype = str(getattr(card, "type", "") or "").upper()
    dmg = int(getattr(card, "damage", 0) or 0)
    blk = int(getattr(card, "block", 0) or 0)
    draws, is_aoe, is_multi_hit, strength_scale, quality = _card_mechanics(card)
    alive = living_monsters(getattr(gs, "monsters", []) or []) if gs is not None else []
    elite_or_boss = _is_elite_or_boss(gs) if gs is not None else False
    target_id = _monster_id(target) if target is not None else ""
    has_nob = any(_monster_id(m) == "GremlinNob" for m in alive)

    if dmg == 0 and ("strike" in name or "bash" in name):
        dmg = 6 if "strike" in name else 8
    if blk == 0 and "defend" in name:
        blk = 5

    s = quality * 0.6
    if incoming >= 10:
        s += blk * 0.9
    else:
        s += blk * 0.05
    if dmg > 0:
        s += (dmg / max(1, cost)) * 1.0
    if is_aoe and len(alive) >= 2:
        s += 5.0 + len(alive)
    if is_multi_hit and any(_monster_id(m) in {"Byrd", "ShelledParasite", "Shelled Parasite"} for m in alive):
        s += 2.5
    if draws:
        s += draws * 1.2
    if "bash" in name:
        s += 5.0 if elite_or_boss else 4.0
    if name in {"uppercut", "shockwave", "thunderclap", "intimidate", "disarm"}:
        s += 5.0 if elite_or_boss or incoming >= 12 else 2.5
    if name in {"feed", "hand of greed"} and target is not None:
        hp = int(getattr(target, "current_hp", 999) or 999)
        if dmg >= hp:
            s += 8.0
    if name in {"immolate", "whirlwind", "cleave", "reaper"} and len(alive) >= 2:
        s += 6.0
    if ctype == "POWER":
        s += 6.0 if elite_or_boss else 3.0
        if incoming >= 18:
            s -= 3.5
    if has_nob and ctype == "SKILL":
        s -= 7.0
        if incoming >= 18 and blk > 0:
            s += 4.0
    if target_id in {
        "GremlinWizard", "SlaverRed", "SlaverBlue", "SlaverBoss",
        "SnakeDagger", "Dagger", "Exploder", "GremlinNob",
        "BookOfStabbing", "SnakePlant", "Byrd", "Chosen",
        "ShelledParasite", "Shelled Parasite", "Mugger", "Looter",
    } and dmg > 0:
        s += 4.0
    s -= max(0, cost - 1) * 0.3
    return s


def pick_target(monsters, prefer_low_hp=True, gs=None):
    alive = living_monsters(monsters or [])
    if not alive:
        return None
    ids = {_monster_id(m) for m in alive}
    if "GremlinLeader" in ids:
        minions = [m for m in alive if _monster_id(m) != "GremlinLeader"]
        attacking = [m for m in minions if _is_attacking(m)]
        wizards = [m for m in minions if _monster_id(m) == "GremlinWizard"]
        if wizards:
            return min(wizards, key=lambda m: int(getattr(m, "current_hp", 999) or 999))
        if attacking:
            return max(attacking, key=_monster_damage)
        if minions:
            return min(minions, key=lambda m: int(getattr(m, "current_hp", 999) or 999))
    if "Reptomancer" in ids:
        daggers = [m for m in alive if _monster_id(m) in {"SnakeDagger", "Dagger"}]
        if daggers:
            attacking = [m for m in daggers if _is_attacking(m)]
            return max(attacking or daggers, key=lambda m: (_monster_damage(m), -int(getattr(m, "current_hp", 999) or 999)))
    for urgent in URGENT_MONSTERS:
        matches = [m for m in alive if _monster_id(m) == urgent]
        if matches:
            return min(matches, key=lambda m: int(getattr(m, "current_hp", 999) or 999))
    if "Sentry" in ids:
        return min(alive, key=lambda m: int(getattr(m, "current_hp", 999) or 999))
    spawners = [m for m in alive if _monster_id(m) in SPAWNER_IDS]
    if spawners and len(alive) == 1:
        return spawners[0]
    if prefer_low_hp:
        alive.sort(key=lambda m: int(getattr(m, "current_hp", 999) or 999))
    return alive[0]


# ---------------------------------------------------------------------------
# Main heuristic dispatcher
# ---------------------------------------------------------------------------
def heuristic_action(gs) -> Tuple[Optional[Action], Optional[int]]:
    """Return (spire_action, flat_action_id) using heuristic logic.

    Returns (None, None) if no action can be determined.
    Covers all decision screens so BC can provide demonstrations
    for the full RL surface.
    """
    screen = _screen_name(gs)
    choice_list = list(getattr(gs, "choice_list", []) or [])
    proceed_avail = bool(getattr(gs, "proceed_available", False))
    cancel_avail = bool(getattr(gs, "cancel_available", False))
    in_combat = bool(getattr(gs, "in_combat", False))
    play_avail = bool(getattr(gs, "play_available", False))
    end_avail = bool(getattr(gs, "end_available", False))
    hand = list(getattr(gs, "hand", []) or [])
    monsters = list(getattr(gs, "monsters", []) or [])
    potions = list(getattr(gs, "potions", []) or [])
    alive = living_monsters(monsters)
    scr = getattr(gs, "screen", None)

    # Terminal
    if is_terminal_state(gs):
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return Action("state"), _NOOP

    # --- Mechanical: CHEST — always open ---
    if screen == "CHEST":
        if scr and getattr(scr, "chest_open", False):
            return (Action("proceed"), _PROCEED) if proceed_avail else (Action("state"), _NOOP)
        return ChooseAction(name="open"), _CHOOSE_START

    # --- Mechanical: HAND_SELECT ---
    if screen == "HAND_SELECT":
        if scr and getattr(scr, "can_pick_zero", False) and proceed_avail:
            return Action("proceed"), _PROCEED
        scr_cards = list(getattr(scr, "cards", []) or [])
        selected = list(getattr(scr, "selected_cards", []) or [])
        num_needed = int(getattr(scr, "num_cards", 1) or 1) if scr else 1
        cards = choice_list or [c.name for c in scr_cards]
        log(f"HAND_SELECT: choice_list={choice_list} scr_cards={len(scr_cards)} "
            f"selected={len(selected)} num_needed={num_needed} "
            f"cards={cards} proceed={proceed_avail} cancel={cancel_avail} "
            f"scr_type={type(scr).__name__ if scr else None}")
        if len(selected) >= num_needed:
            if proceed_avail:
                return Action("proceed"), _PROCEED
            return Action("state"), _NOOP
        if cards:
            idx = pick_hand_select(cards)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if cancel_avail:
            return Action("leave"), _LEAVE
        return ChooseAction(choice_index=0), _CHOOSE_START

    # --- Mechanical: GRID ---
    if screen == "GRID":
        if scr and getattr(scr, "confirm_up", False):
            return Action("proceed"), _PROCEED
        num_needed = int(getattr(scr, "num_cards", 1) or 1) if scr else 1
        selected = list(getattr(scr, "selected_cards", []) or []) if scr else []
        already = len(selected)
        if already >= num_needed:
            if proceed_avail:
                return Action("proceed"), _PROCEED
            return Action("state"), _NOOP
        selected_names = {getattr(c, "name", "").lower() for c in selected}
        grid_choices = choice_list or [
            getattr(c, "name", c) for c in getattr(scr, "cards", []) or []
        ]
        if grid_choices:
            for_upgrade = bool(getattr(scr, "for_upgrade", False)) if scr else False
            for_transform = bool(getattr(scr, "for_transform", False)) if scr else False
            any_number = bool(getattr(scr, "any_number", False)) if scr else False
            smith_grid = _looks_like_smith_grid(gs, scr, grid_choices, cancel_avail)
            if _is_matching_grid(gs, scr, grid_choices):
                idx = _pick_grid_match(grid_choices, scr, gs)
                if idx is not None:
                    return ChooseAction(choice_index=idx), _CHOOSE_START + idx
                if proceed_avail:
                    return Action("proceed"), _PROCEED
                if cancel_avail:
                    return Action("leave"), _LEAVE
                return Action("state"), _NOOP
            if for_upgrade or smith_grid:
                idx = _pick_grid_upgrade(grid_choices)
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if for_transform:
                return ChooseAction(choice_index=0), _CHOOSE_START
            if any_number:
                if proceed_avail:
                    return Action("proceed"), _PROCEED
                return ChooseAction(choice_index=0), _CHOOSE_START
            unselected = [(i, c) for i, c in enumerate(grid_choices)
                          if str(c).lower() not in selected_names]
            if not unselected:
                unselected = list(enumerate(grid_choices))
            idx = _pick_from_unselected(unselected, scr)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        if cancel_avail:
            return Action("leave"), _LEAVE
        return Action("state"), _NOOP

    # --- Mechanical: MAP boss-only ---
    if screen == "MAP":
        boss_avail = scr and getattr(scr, "boss_available", False)
        if boss_avail and not choice_list:
            return ChooseAction(name="boss"), _CHOOSE_START
        next_nodes = list(getattr(scr, "next_nodes", []) or []) if scr else []
        n = len(choice_list) or len(next_nodes)
        if n > 0:
            idx = pick_map(choice_list, gs) if choice_list else 0
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if boss_avail:
            return ChooseAction(name="boss"), _CHOOSE_START
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- BOSS_REWARD ---
    if screen == "BOSS_REWARD":
        if choice_list:
            idx = pick_boss_relic(choice_list)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- CARD_REWARD ---
    if screen == "CARD_REWARD":
        if choice_list:
            idx = pick_card_reward(choice_list, gs)
            if "skip" in str(choice_list[idx]).lower():
                _skipped_card_reward_keys.add(_screen_key(gs))
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        if cancel_avail:
            return Action("leave"), _LEAVE
        return None, None

    # --- COMBAT_REWARD ---
    if screen == "COMBAT_REWARD":
        potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())
        scr_obj = getattr(gs, "screen", None)
        rewards = list(getattr(scr_obj, "rewards", []) or []) if scr_obj else []
        skip_card_reward = _screen_key(gs) in _skipped_card_reward_keys
        if rewards:
            idx = pick_combat_reward_obj(rewards, potions_full, skip_card_reward)
            if idx >= 0:
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if proceed_avail:
                return Action("proceed"), _PROCEED
        if choice_list:
            idx = pick_combat_reward_str(
                choice_list,
                potions_full=potions_full,
                skip_card=skip_card_reward,
            )
            if idx >= 0:
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            idx = pick_map(choice_list, gs)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- REST ---
    if screen == "REST":
        if choice_list:
            idx = pick_rest(choice_list, gs)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- EVENT ---
    if screen == "EVENT":
        if choice_list:
            if event_choice_targets(gs):
                slot, choice_idx = pick_event_slot_and_choice(choice_list, gs)
                return ChooseAction(choice_index=choice_idx), _CHOOSE_START + slot
        options = list(getattr(scr, "options", []) or []) if scr else []
        if options:
            if event_choice_targets(gs):
                slot, choice_idx = pick_event_slot_and_choice([], gs)
                return ChooseAction(choice_index=choice_idx), _CHOOSE_START + slot
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- SHOP_ROOM ---
    if screen == "SHOP_ROOM":
        floor = int(getattr(gs, "floor", 0) or 0)
        gold = int(getattr(gs, "gold", 0) or 0)
        if floor not in _visited_shop_floors and gold >= 50 and choice_list:
            _visited_shop_floors.add(floor)
            lower = [str(c).lower() for c in choice_list]
            for i, c in enumerate(lower):
                if "shop" in c or "merchant" in c:
                    return ChooseAction(choice_index=i), _CHOOSE_START + i
        if proceed_avail:
            return Action("proceed"), _PROCEED
        if cancel_avail:
            return Action("leave"), _LEAVE
        return None, None

    # --- SHOP_SCREEN ---
    if screen == "SHOP_SCREEN":
        gold = int(getattr(gs, "gold", 0) or 0)
        if scr is not None and gold >= 30:
            for card in (getattr(scr, "cards", None) or []):
                name = str(getattr(card, "name", "") or "")
                price = int(getattr(card, "price", 999) or 999)
                if name.lower() in GOOD_CARDS and gold >= price:
                    return ChooseAction(name=name), _CHOOSE_START
            if getattr(scr, "purge_available", False):
                purge_cost = int(getattr(scr, "purge_cost", 999) or 999)
                if gold >= purge_cost:
                    return ChooseAction(name="purge"), _CHOOSE_START
            for card in (getattr(scr, "cards", None) or []):
                name = str(getattr(card, "name", "") or "")
                price = int(getattr(card, "price", 999) or 999)
                if name.lower() in OK_CARDS and gold >= price:
                    return ChooseAction(name=name), _CHOOSE_START
            for relic in (getattr(scr, "relics", None) or []):
                name = str(getattr(relic, "name", "") or "")
                price = int(getattr(relic, "price", 999) or 999)
                if gold >= price:
                    return ChooseAction(name=name), _CHOOSE_START
            potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())
            if not potions_full:
                for pot in (getattr(scr, "potions", None) or []):
                    name = str(getattr(pot, "name", "") or "")
                    price = int(getattr(pot, "price", 999) or 999)
                    if gold >= price:
                        return ChooseAction(name=name), _CHOOSE_START
        return Action("cancel"), _LEAVE

    # --- COMBAT ---
    if in_combat:
        incoming = estimate_incoming(monsters)
        elite_or_boss = _is_elite_or_boss(gs)
        potion_avail = bool(getattr(gs, "potion_available", True))
        target = pick_target(monsters, prefer_low_hp=True, gs=gs)

        # -- Use potions when conditions warrant --
        hp = int(getattr(gs, "current_hp", 0) or 0)
        max_hp = max(1, int(getattr(gs, "max_hp", 1) or 1))
        hp_pct = hp / max_hp
        player_block = int(getattr(getattr(gs, "player", None), "block", 0) or 0)
        potion_slots_full = bool(getattr(gs, "are_potions_full", lambda: False)())
        for k, pot in enumerate(potions[:MAX_POTIONS]):
            pot_name = _potion_name(pot)
            if _norm(pot_name) in {"", "potionslot"}:
                continue
            if not potion_avail or not getattr(pot, "can_use", False):
                continue
            effects = _potion_effects(pot)
            deals_damage, gives_block, gives_str, gives_dex, heals = effects
            pname = _norm(pot_name)
            use = False
            if heals and (hp_pct < 0.35 or (elite_or_boss and hp_pct < 0.55)):
                use = True
            elif gives_block and incoming - player_block >= (8 if elite_or_boss else 12):
                use = True
            elif gives_str and (elite_or_boss or incoming > 15):
                use = True
            elif gives_dex and (elite_or_boss or incoming > 10):
                use = True
            elif deals_damage:
                tgt_hp = int(getattr(target, "current_hp", 999) or 999) if target else 999
                if elite_or_boss or incoming > 12 or tgt_hp <= 20:
                    use = True
            elif pname in {
                "fearpotion", "weakpotion", "ancientpotion", "cultistpotion",
                "powerpotion", "attackpotion", "skillpotion", "duplicationpotion",
                "energypotion", "distilledchaos", "liquidmemories",
                "blessingoftheforge", "gamblersbrew",
            } and (elite_or_boss or incoming > 18 or potion_slots_full):
                use = True
            elif pname == "smokebomb" and hp_pct < 0.2 and incoming >= hp:
                use = True
            if use:
                requires_target = getattr(pot, "requires_target", False)
                if requires_target:
                    tgt = target or pick_target(monsters, prefer_low_hp=True, gs=gs)
                    if tgt is not None:
                        tgt_idx = alive.index(tgt) if tgt in alive else 0
                        if tgt_idx < MAX_MONSTERS:
                            aid = _POTION_TARGETED_START + k * MAX_MONSTERS + tgt_idx
                            return PotionAction(use=True, potion=pot, target_monster=tgt), aid
                else:
                    aid = _POTION_UNTARGETED_START + k
                    return PotionAction(use=True, potion=pot), aid

        if play_avail and hand:
            best_card = None
            best_score = -1e9
            for card in hand:
                if not getattr(card, "is_playable", False):
                    continue
                s = score_card(card, incoming, gs=gs, target=target)
                if s > best_score:
                    best_score = s
                    best_card = card

            if best_card is not None:
                if best_score <= 0.0 and end_avail:
                    return Action("end"), _END_TURN
                card_idx = hand.index(best_card)
                if card_idx < MAX_HAND:
                    if getattr(best_card, "has_target", False):
                        tgt = target or pick_target(monsters, prefer_low_hp=True, gs=gs)
                        if tgt is not None:
                            tgt_idx = alive.index(tgt) if tgt in alive else 0
                            if tgt_idx < MAX_MONSTERS:
                                action_id = _PLAY_TARGETED_START + card_idx * MAX_MONSTERS + tgt_idx
                                return PlayCardAction(card=best_card, target_monster=tgt), action_id
                    else:
                        action_id = _PLAY_UNTARGETED_START + card_idx
                        return PlayCardAction(card=best_card), action_id

        if end_avail:
            return Action("end"), _END_TURN

    # --- Trivial fallback (mirrors train_ppo auto-handle) ---
    log(f"FALLBACK: screen={screen} choice_list={choice_list} "
        f"proceed={proceed_avail} cancel={cancel_avail} "
        f"scr_type={type(scr).__name__ if scr else None}")
    scr_cards = list(getattr(scr, "cards", []) or [])
    all_choices = choice_list or [c.name for c in scr_cards]
    if all_choices:
        return ChooseAction(choice_index=0), _CHOOSE_START
    if not choice_list and not cancel_avail:
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return Action("state"), _NOOP
    if not choice_list and cancel_avail and not proceed_avail:
        return Action("leave"), _LEAVE

    return Action("state"), _NOOP


def _action_desc(action: Optional[Action]) -> str:
    if action is None:
        return "None"
    command = str(getattr(action, "command", type(action).__name__))
    parts = [command]
    for attr in ("choice_index", "name", "card_index", "target_index"):
        value = getattr(action, attr, None)
        if value not in (None, -1):
            parts.append(f"{attr}={value}")
    return " ".join(parts)


def _tmp_npz_path(path: str) -> str:
    return path.replace(".npz", ".tmp.npz") if path.endswith(".npz") else path + ".tmp.npz"


def _remove_if_exists(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
            log(f"Removed BC progress checkpoint: {path}")
    except OSError as e:
        log(f"Failed to remove BC progress checkpoint {path}: {e}")


# ---------------------------------------------------------------------------
# Demonstration collector
# ---------------------------------------------------------------------------
class DemoCollector:
    def __init__(self, max_games: int, checkpoint_path: Optional[str] = None,
                 resume_checkpoint: bool = True, save_path: str = ""):
        self.max_games = max_games
        self.games_done = 0
        self.total_steps = 0
        self.initialized = False
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.skipped_samples = 0
        self._game_start_steps = 0
        self._game_start_sample_len = 0
        self._game_start_skipped = 0
        self.fight_tracker = FightTracker(source="bc", worker="bc", log=log)

        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.masks: List[np.ndarray] = []
        if resume_checkpoint and checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
            tmp = _tmp_npz_path(self.checkpoint_path)
            np.savez_compressed(
                tmp,
                observations=np.array(self.observations, dtype=np.float32),
                actions=np.array(self.actions, dtype=np.int64),
                action_masks=np.array(self.masks, dtype=np.bool_),
                games_done=np.array(self.games_done, dtype=np.int64),
                total_steps=np.array(self.total_steps, dtype=np.int64),
                skipped_samples=np.array(self.skipped_samples, dtype=np.int64),
                max_games=np.array(self.max_games, dtype=np.int64),
                run_id=np.array(self.run_id),
                saved_at=np.array(datetime.now().isoformat()),
            )
            os.replace(tmp, self.checkpoint_path)
            log(f"BC progress checkpoint saved: games={self.games_done}/{self.max_games} "
                f"samples={len(self.actions)} path={self.checkpoint_path}")
        except Exception as e:
            log(f"Failed to save BC progress checkpoint {self.checkpoint_path}: {e}")

    def _load_checkpoint(self, path: str) -> None:
        try:
            with np.load(path, allow_pickle=False) as data:
                self.observations = [x for x in data["observations"]]
                self.actions = [int(x) for x in data["actions"]]
                self.masks = [x for x in data["action_masks"]]
                self.games_done = int(data["games_done"].item())
                self.total_steps = int(data["total_steps"].item())
                if "skipped_samples" in data.files:
                    self.skipped_samples = int(data["skipped_samples"].item())
                if "run_id" in data.files:
                    self.run_id = str(data["run_id"].item())
            self.games_done = min(self.games_done, self.max_games)
            log(f"Resumed BC progress checkpoint: games={self.games_done}/{self.max_games} "
                f"samples={len(self.actions)} skipped={self.skipped_samples} path={path}")
        except Exception as e:
            log(f"Failed to load BC progress checkpoint {path}: {e}")

    def on_state_change(self, gs) -> Action:
        try:
            return self._handle(gs)
        except Exception as e:
            log(f"ERROR: {e}")
            log(traceback.format_exc())
            return Action("state")

    def _handle(self, gs) -> Action:
        screen = _screen_name(gs)
        terminal = is_terminal_state(gs)
        victory = is_victory_state(gs)
        self.fight_tracker.observe(
            gs, game=self.games_done + 1, terminal=terminal, victory=victory,
        )

        if not self.initialized:
            self.initialized = True
            self._game_start_steps = self.total_steps
            self._game_start_sample_len = len(self.observations)
            self._game_start_skipped = self.skipped_samples
            log(f"Demo game #{self.games_done + 1} started, floor={getattr(gs, 'floor', '?')}")

        if terminal:
            self.games_done += 1
            self.initialized = False
            _visited_shop_floors.clear()
            _skipped_card_reward_keys.clear()
            fight_stats = self.fight_tracker.finish_game(
                gs, game=self.games_done, victory=victory
            )
            steps_this_game = max(0, self.total_steps - self._game_start_steps)
            samples_this_game = max(0, len(self.observations) - self._game_start_sample_len)
            skipped_this_game = max(0, self.skipped_samples - self._game_start_skipped)
            log(f"Demo game #{self.games_done} ended. "
                f"Samples so far: {len(self.observations)}, total_steps={self.total_steps}")
            append_bc_stats({
                "run_id": self.run_id,
                "source": "bc",
                "game": self.games_done,
                "target_games": self.max_games,
                "steps": steps_this_game,
                "samples": samples_this_game,
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
                "checkpoint_path": self.checkpoint_path or "",
                "model_path": self.save_path,
            }, log=log)
            self._save_checkpoint()
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            if proceed_avail:
                return Action("proceed")
            return Action("state")

        spire_action, action_id = heuristic_action(gs)

        if spire_action is None or action_id is None:
            return Action("state")

        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)
        mask_ok = bool(action_id < NUM_ACTIONS and mask[action_id])

        if mask_ok:
            self.observations.append(obs)
            self.actions.append(action_id)
            self.masks.append(mask)
        else:
            self.skipped_samples += 1
            if VERBOSE:
                log(f"BC VERBOSE skipped sample: screen={screen} action_id={action_id} "
                    f"mask_sum={int(mask.sum())} action={_action_desc(spire_action)}")

        self.total_steps += 1
        if VERBOSE:
            choice_list = list(getattr(gs, "choice_list", []) or [])
            log(f"BC STEP {self.total_steps}: game={self.games_done + 1} "
                f"floor={getattr(gs, 'floor', '?')} "
                f"hp={getattr(gs, 'current_hp', '?')}/{getattr(gs, 'max_hp', '?')} "
                f"screen={screen} choices={choice_list[:8]} action_id={action_id} "
                f"action={_action_desc(spire_action)} mask_ok={mask_ok} "
                f"samples={len(self.observations)} skipped={self.skipped_samples}")
        elif self.total_steps % 100 == 0:
            log(f"  step={self.total_steps} samples={len(self.observations)} "
                f"skipped={self.skipped_samples} games={self.games_done}/{self.max_games}")

        return spire_action

    def on_out_of_game(self) -> Action:
        log(f"OUT OF GAME (demos: {self.games_done}/{self.max_games})")
        if self.games_done >= self.max_games:
            log("Enough demos collected, starting training...")
            raise StopIteration()
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        return recover_from_command_error(err)


# ---------------------------------------------------------------------------
# Supervised training
# ---------------------------------------------------------------------------
BC_TRAIN_STATS_CSV = os.path.join(_root, "logs", "bc_train_stats.csv")


def _action_group(action_id: int) -> str:
    """Coarse action group for BC diagnostics."""
    if _PLAY_TARGETED_START <= action_id < _PLAY_TARGETED_START + MAX_HAND * MAX_MONSTERS:
        return "card_targeted"
    if _PLAY_UNTARGETED_START <= action_id < _PLAY_UNTARGETED_START + MAX_HAND:
        return "card_untargeted"
    if action_id == _END_TURN:
        return "end_turn"
    if _POTION_TARGETED_START <= action_id < _POTION_TARGETED_START + MAX_POTIONS * MAX_MONSTERS:
        return "potion_targeted"
    if _POTION_UNTARGETED_START <= action_id < _POTION_UNTARGETED_START + MAX_POTIONS:
        return "potion_untargeted"
    if _CHOOSE_START <= action_id < _CHOOSE_START + 40:
        return "choice"
    if action_id == _PROCEED:
        return "proceed"
    if action_id == _LEAVE:
        return "leave"
    if action_id == _NOOP:
        return "noop"
    return "other"


def _append_bc_train_stats(row: dict) -> None:
    columns = [
        "timestamp", "epoch", "epochs", "samples", "train_samples", "val_samples",
        "train_loss", "train_acc", "val_loss", "val_acc", "best_val_loss",
        "lr", "batch_size", "weight_decay", "label_smoothing", "patience",
        "choice_acc", "card_targeted_acc", "card_untargeted_acc", "end_turn_acc",
        "potion_targeted_acc", "potion_untargeted_acc", "proceed_acc", "leave_acc",
    ]
    try:
        os.makedirs(os.path.dirname(BC_TRAIN_STATS_CSV), exist_ok=True)
        exists = os.path.exists(BC_TRAIN_STATS_CSV)
        with open(BC_TRAIN_STATS_CSV, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            out = {c: row.get(c, "") for c in columns}
            out["timestamp"] = out["timestamp"] or datetime.now().isoformat()
            writer.writerow(out)
    except Exception as e:
        log(f"bc train stats append failed: {e}")


def _eval_bc_metrics(trainer, obs_t, act_t, mask_t, indices, loss_fn, batch_size: int) -> tuple[float, float, dict]:
    if len(indices) == 0:
        return float("nan"), float("nan"), {}
    total_loss = 0.0
    total_examples = 0
    correct = 0
    group_total: dict[str, int] = {}
    group_correct: dict[str, int] = {}

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            b_obs = obs_t[idx]
            b_act = act_t[idx]
            b_mask = mask_t[idx]
            features = trainer.shared(b_obs)
            logits = trainer.policy_head(features).masked_fill(~b_mask, -1e8)
            loss = loss_fn(logits, b_act)
            pred = logits.argmax(dim=-1)
            n_batch = len(idx)
            total_loss += float(loss.item()) * n_batch
            total_examples += n_batch
            correct += int((pred == b_act).sum().item())

            for action, ok in zip(b_act.cpu().numpy().tolist(), (pred == b_act).cpu().numpy().tolist()):
                group = _action_group(int(action))
                group_total[group] = group_total.get(group, 0) + 1
                group_correct[group] = group_correct.get(group, 0) + int(bool(ok))

    group_acc = {
        f"{group}_acc": (group_correct.get(group, 0) / max(1, total) * 100.0)
        for group, total in group_total.items()
    }
    return total_loss / max(1, total_examples), correct / max(1, total_examples) * 100.0, group_acc


def _restore_trainer_state(trainer, state: dict) -> None:
    if not state:
        return
    trainer.shared.load_state_dict(state["shared"])
    trainer.policy_head.load_state_dict(state["policy_head"])
    trainer.value_head.load_state_dict(state["value_head"])


def train_supervised(obs_list, action_list, mask_list, save_path: str,
                     epochs: int = 50, lr: float = 5e-4, batch_size: int = 256,
                     val_split: float = 0.10, patience: int = 12,
                     weight_decay: float = 1e-5, label_smoothing: float = 0.02,
                     seed: int = 20260511):
    from ppo_model import PPOTrainer

    n = len(obs_list)
    rng = np.random.default_rng(seed)
    log(
        f"Training supervised on {n} samples, {epochs} epochs "
        f"lr={lr} batch={batch_size} val_split={val_split} "
        f"patience={patience} weight_decay={weight_decay} label_smoothing={label_smoothing}"
    )

    trainer = PPOTrainer(
        obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
        lr=lr, net_arch=(256, 256),
    )

    obs_arr = np.array(obs_list, dtype=np.float32)
    if obs_arr.ndim == 2 and obs_arr.shape[1] < OBS_SIZE:
        pad = np.zeros((obs_arr.shape[0], OBS_SIZE - obs_arr.shape[1]), dtype=np.float32)
        obs_arr = np.concatenate([obs_arr, pad], axis=1)
        log(f"Zero-padded BC observations from {obs_arr.shape[1] - pad.shape[1]} to {OBS_SIZE}")
    obs_t = torch.as_tensor(obs_arr)
    act_t = torch.as_tensor(np.array(action_list, dtype=np.int64))
    mask_t = torch.as_tensor(np.array(mask_list, dtype=np.bool_))

    all_indices = rng.permutation(n)
    val_n = int(round(n * max(0.0, min(0.4, val_split))))
    if n < 1000:
        val_n = min(val_n, max(0, n // 10))
    val_idx = all_indices[:val_n]
    train_idx = all_indices[val_n:] if val_n > 0 else all_indices

    params = (
        list(trainer.shared.parameters())
        + list(trainer.policy_head.parameters())
        + list(trainer.value_head.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        indices = rng.permutation(train_idx)
        total_loss = 0.0
        total_examples = 0
        correct = 0

        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            b_obs = obs_t[idx]
            b_act = act_t[idx]
            b_mask = mask_t[idx]

            features = trainer.shared(b_obs)
            logits = trainer.policy_head(features).masked_fill(~b_mask, -1e8)
            loss = loss_fn(logits, b_act)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            n_batch = len(idx)
            total_loss += float(loss.item()) * n_batch
            total_examples += n_batch
            correct += int((logits.argmax(dim=-1) == b_act).sum().item())

        train_loss = total_loss / max(1, total_examples)
        train_acc = correct / max(1, total_examples) * 100.0
        val_loss, val_acc, group_acc = _eval_bc_metrics(
            trainer, obs_t, act_t, mask_t, val_idx, loss_fn, batch_size
        ) if val_n > 0 else (float("nan"), float("nan"), {})

        metric = val_loss if val_n > 0 and val_loss == val_loss else train_loss
        if metric < best_metric - 1e-5:
            best_metric = metric
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                "shared": {k: v.detach().cpu().clone() for k, v in trainer.shared.state_dict().items()},
                "policy_head": {k: v.detach().cpu().clone() for k, v in trainer.policy_head.state_dict().items()},
                "value_head": {k: v.detach().cpu().clone() for k, v in trainer.value_head.state_dict().items()},
            }
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch + 1,
            "epochs": epochs,
            "samples": n,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.3f}",
            "val_loss": f"{val_loss:.6f}" if val_loss == val_loss else "",
            "val_acc": f"{val_acc:.3f}" if val_acc == val_acc else "",
            "best_val_loss": f"{best_metric:.6f}",
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "patience": patience,
        }
        row.update({k: f"{v:.3f}" for k, v in group_acc.items()})
        _append_bc_train_stats(row)

        if (epoch + 1) % 5 == 0 or epoch == 0 or stale_epochs == patience:
            val_part = f" val_loss={val_loss:.4f} val_acc={val_acc:.1f}%" if val_n > 0 else ""
            log(
                f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.1f}%{val_part} best_epoch={best_epoch}"
            )

        if patience > 0 and val_n > 0 and stale_epochs >= patience:
            log(f"Early stopping BC at epoch {epoch+1}; best_epoch={best_epoch} best_val_loss={best_metric:.4f}")
            break

    _restore_trainer_state(trainer, best_state)
    trainer.save(save_path)
    log(f"Warm-started model saved to {save_path} (best_epoch={best_epoch})")
    return trainer


def save_demo_dataset(obs_list, action_list, mask_list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = _tmp_npz_path(path)
    np.savez_compressed(
        tmp,
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(action_list, dtype=np.int64),
        action_masks=np.array(mask_list, dtype=np.bool_),
        created_at=np.array(datetime.now().isoformat()),
        samples=np.array(len(action_list), dtype=np.int64),
    )
    os.replace(tmp, path)
    log(f"BC demo dataset saved to {path} ({len(action_list)} samples)")


def load_demo_dataset(path: str) -> tuple[list, list, list]:
    """Load a saved BC demo .npz file."""
    with np.load(path, allow_pickle=False) as data:
        obs = [x for x in data["observations"]]
        actions = [int(x) for x in data["actions"]]
        masks = [x for x in data["action_masks"]]
    if len(obs) != len(actions) or len(obs) != len(masks):
        raise ValueError(
            f"demo length mismatch in {path}: obs={len(obs)} "
            f"actions={len(actions)} masks={len(masks)}"
        )
    return obs, actions, masks


def load_demo_dir(path: str) -> tuple[list, list, list]:
    """Merge every saved BC demo .npz file in a directory."""
    import glob

    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    obs_all: list = []
    action_all: list = []
    mask_all: list = []
    loaded = 0

    for f in files:
        name = os.path.basename(f).lower()
        # Progress checkpoints may contain compatible arrays, but they can
        # represent incomplete collection state. Only consume explicit demo
        # datasets so partial checkpoints do not silently enter training.
        if "progress" in name or "checkpoint" in name:
            continue
        try:
            obs, actions, masks = load_demo_dataset(f)
        except Exception as e:
            log(f"Skipping demo file {f}: {e}")
            continue
        obs_all.extend(obs)
        action_all.extend(actions)
        mask_all.extend(masks)
        loaded += 1
        log(f"Loaded demo file: {f} samples={len(actions)}")

    log(f"Merged {loaded} demo file(s) from {path}: samples={len(action_all)}")
    return obs_all, action_all, mask_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=150,
                        help="Number of heuristic games to collect")
    parser.add_argument("--save", type=str, default="models/ppo_sts.pt",
                        help="Where to save warm-started model")
    parser.add_argument("--demo-save", type=str, default=None,
                        help="Where to save BC demo dataset for PPO anchoring")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Per-game BC progress checkpoint path")
    parser.add_argument("--no-resume-bc", dest="resume_bc", action="store_false",
                        help="Ignore any existing BC progress checkpoint")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Supervised BC epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Supervised BC learning rate (default: 5e-4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Supervised BC batch size (default: 256)")
    parser.add_argument("--val-split", type=float, default=0.10,
                        help="Holdout validation split for BC training (default: 0.10)")
    parser.add_argument("--patience", type=int, default=12,
                        help="Early-stop patience on validation loss; 0 disables (default: 12)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Adam weight decay for BC training (default: 1e-5)")
    parser.add_argument("--label-smoothing", type=float, default=0.02,
                        help="Cross-entropy label smoothing for BC (default: 0.02)")
    parser.add_argument("--seed", type=int, default=20260511,
                        help="RNG seed for train/validation split and shuffling")
    parser.add_argument("--collect-only", action="store_true",
                        help="Collect demos and save them, but skip supervised training")
    parser.add_argument("--worker-id", type=str, default="1",
                        help="Identifier used in demo filenames for parallel BC collection")
    parser.add_argument("--demo-dir", type=str, default=None,
                        help="Directory for collect-only demo files")
    parser.add_argument("--train-demo-dir", type=str, default=None,
                        help="Skip live collection and train from all demo .npz files in this directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Write detailed per-state/per-action debug logs")
    args = parser.parse_args()
    VERBOSE = VERBOSE or args.verbose

    log("=== BEHAVIOR CLONE STARTING ===")

    save_path = os.path.join(_root, args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if args.train_demo_dir:
        demo_dir = args.train_demo_dir
        if not os.path.isabs(demo_dir):
            demo_dir = os.path.join(_root, demo_dir)
        obs_list, action_list, mask_list = load_demo_dir(demo_dir)
        if len(action_list) < 50:
            log(f"Too few samples in demo dir {demo_dir}, skipping training")
            return
        demo_save = args.demo_save
        if demo_save is None:
            demo_save = save_path.replace(".pt", "_bc_demos.npz")
        elif not os.path.isabs(demo_save):
            demo_save = os.path.join(_root, demo_save)
        save_demo_dataset(obs_list, action_list, mask_list, demo_save)
        train_supervised(
            obs_list, action_list, mask_list, save_path,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            val_split=args.val_split, patience=args.patience,
            weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
            seed=args.seed,
        )
        log("=== BEHAVIOR CLONE TRAIN-FROM-DEMO-DIR COMPLETE ===")
        return

    demo_save = args.demo_save
    if demo_save is None:
        if args.demo_dir:
            demo_dir = args.demo_dir
            if not os.path.isabs(demo_dir):
                demo_dir = os.path.join(_root, demo_dir)
            os.makedirs(demo_dir, exist_ok=True)
            demo_save = os.path.join(
                demo_dir,
                f"bc_demo_worker_{args.worker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
            )
        else:
            demo_save = save_path.replace(".pt", "_bc_demos.npz")
    elif not os.path.isabs(demo_save):
        demo_save = os.path.join(_root, demo_save)
    bc_checkpoint = args.bc_checkpoint
    if bc_checkpoint is None:
        bc_checkpoint = save_path.replace(".pt", "_bc_progress.npz")
    elif not os.path.isabs(bc_checkpoint):
        bc_checkpoint = os.path.join(_root, bc_checkpoint)

    collector = DemoCollector(
        max_games=args.games,
        checkpoint_path=bc_checkpoint,
        resume_checkpoint=args.resume_bc,
        save_path=save_path,
    )

    if collector.games_done < args.games:
        coord = Coordinator()
        coord.register_state_change_callback(collector.on_state_change)
        coord.register_out_of_game_callback(collector.on_out_of_game)
        coord.register_command_error_callback(collector.on_error)

        log(f"Starting demo collection: {args.games} games "
            f"save={args.save} epochs={args.epochs} lr={args.lr} "
            f"batch={args.batch_size} val_split={args.val_split} "
            f"patience={args.patience} collect_only={args.collect_only} "
            f"worker_id={args.worker_id} demo_save={demo_save} verbose={VERBOSE} "
            f"checkpoint={bc_checkpoint} resume={args.resume_bc}")
        coord.signal_ready()

        try:
            coord.run()
        except StopIteration:
            pass
    else:
        log(f"BC checkpoint already has {collector.games_done}/{args.games} games; "
            "training immediately")

    n = len(collector.observations)
    log(f"Collection done: {collector.games_done} games, {n} samples")

    if n < 50:
        log("Too few samples, skipping training")
        return

    save_demo_dataset(
        collector.observations, collector.actions, collector.masks,
        demo_save,
    )

    if args.collect_only:
        log(f"Collect-only mode complete. Demo file saved to {demo_save}")
        _remove_if_exists(bc_checkpoint)
        return

    train_supervised(
        collector.observations, collector.actions, collector.masks,
        save_path, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_split=args.val_split, patience=args.patience,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        seed=args.seed,
    )
    _remove_if_exists(bc_checkpoint)
    log("=== BEHAVIOR CLONE COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
