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
  command=python behavior_clone.py --games 50 --save models/ppo_sts.pt
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
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

os.makedirs(os.path.join(_root, "logs"), exist_ok=True)
DEBUG_LOG = os.path.join(_root, "logs", "bc_debug.log")

def log(msg: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
            f.flush()
    except Exception:
        pass

log("=== BEHAVIOR CLONE STARTING ===")

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
    is_terminal_state,
)

from game_data import POTION_EFFECTS
from game_data import CARD_MECHANICS
from screen_handler import (
    GOOD_CARDS, OK_CARDS, JUNK_CARDS,
    pick_card_reward, pick_combat_reward_obj, pick_combat_reward_str,
    pick_rest, pick_event, pick_boss_relic, pick_hand_select,
    pick_grid_card, _pick_grid_upgrade, _is_matching_grid, _pick_grid_match,
    _pick_from_unselected,
    pick_map,
    pick_shop_item,
)

log("Imports done")


# ---------------------------------------------------------------------------
# Shop visit tracking (prevents SHOP_ROOM ↔ SHOP_SCREEN infinite loop)
# ---------------------------------------------------------------------------
_visited_shop_floors: set = set()


# ---------------------------------------------------------------------------
# Heuristic action picker
# ---------------------------------------------------------------------------
def _screen_name(gs) -> str:
    st = getattr(gs, "screen_type", None)
    name = getattr(st, "name", st) if st is not None else "NONE"
    return str(name) if name else "NONE"


URGENT_MONSTERS = {
    "SlaverRed", "GremlinWizard", "Exploder", "Mugger", "Looter",
    "SnakeDagger", "Dagger", "Byrd", "Chosen",
}


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
    if is_multi_hit and any(_monster_id(m) in {"Byrd", "Shelled Parasite"} for m in alive):
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
    if target_id in {"GremlinWizard", "SlaverRed", "SnakeDagger", "Dagger", "Exploder"} and dmg > 0:
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
            if _is_matching_grid(gs, scr, grid_choices):
                idx = _pick_grid_match(grid_choices, scr, gs)
                if idx is not None:
                    return ChooseAction(choice_index=idx), _CHOOSE_START + idx
                if proceed_avail:
                    return Action("proceed"), _PROCEED
                if cancel_avail:
                    return Action("leave"), _LEAVE
                return Action("state"), _NOOP
            if for_upgrade:
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
        if rewards:
            idx = pick_combat_reward_obj(rewards, potions_full)
            if idx >= 0:
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if proceed_avail:
                return Action("proceed"), _PROCEED
        if choice_list:
            idx = pick_combat_reward_str(choice_list, potions_full=potions_full)
            if idx >= 0:
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if proceed_avail:
                return Action("proceed"), _PROCEED
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
            idx = pick_event(choice_list, gs)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        options = list(getattr(scr, "options", []) or []) if scr else []
        if options:
            enabled = [
                (i, opt) for i, opt in enumerate(options)
                if not bool(getattr(opt, "disabled", False))
            ]
            if enabled:
                idx, opt = enabled[0]
                choice_idx = getattr(opt, "choice_index", None)
                return ChooseAction(choice_index=choice_idx if choice_idx is not None else idx), _CHOOSE_START + idx
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


# ---------------------------------------------------------------------------
# Demonstration collector
# ---------------------------------------------------------------------------
class DemoCollector:
    def __init__(self, max_games: int):
        self.max_games = max_games
        self.games_done = 0
        self.total_steps = 0
        self.initialized = False

        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.masks: List[np.ndarray] = []

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

        if not self.initialized:
            self.initialized = True
            log(f"Demo game #{self.games_done + 1} started, floor={getattr(gs, 'floor', '?')}")

        if terminal:
            self.games_done += 1
            self.initialized = False
            _visited_shop_floors.clear()
            log(f"Demo game #{self.games_done} ended. "
                f"Samples so far: {len(self.observations)}, total_steps={self.total_steps}")
            proceed_avail = bool(getattr(gs, "proceed_available", False))
            if proceed_avail:
                return Action("proceed")
            return Action("state")

        spire_action, action_id = heuristic_action(gs)

        if spire_action is None or action_id is None:
            return Action("state")

        obs = encode_game_state(gs)
        mask = compute_action_mask(gs)

        if action_id < NUM_ACTIONS and mask[action_id]:
            self.observations.append(obs)
            self.actions.append(action_id)
            self.masks.append(mask)

        self.total_steps += 1
        if self.total_steps % 100 == 0:
            log(f"  step={self.total_steps} samples={len(self.observations)} "
                f"games={self.games_done}/{self.max_games}")

        return spire_action

    def on_out_of_game(self) -> Action:
        log(f"OUT OF GAME (demos: {self.games_done}/{self.max_games})")
        if self.games_done >= self.max_games:
            log("Enough demos collected, starting training...")
            raise StopIteration()
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    def on_error(self, err: str) -> Action:
        log(f"COMMAND ERROR: {err}")
        if "Possible commands" in err and "wait" in err:
            return Action("wait")
        if "proceed" in err and "choose" in err:
            return ChooseAction(choice_index=0)
        return Action("state")


# ---------------------------------------------------------------------------
# Supervised training
# ---------------------------------------------------------------------------
def train_supervised(obs_list, action_list, mask_list, save_path: str,
                     epochs: int = 30, lr: float = 1e-3, batch_size: int = 128):
    from ppo_model import PPOTrainer

    n = len(obs_list)
    log(f"Training supervised on {n} samples, {epochs} epochs...")

    trainer = PPOTrainer(
        obs_size=OBS_SIZE, n_actions=NUM_ACTIONS, device="cpu",
        lr=lr, net_arch=(256, 256),
    )

    obs_t = torch.as_tensor(np.array(obs_list, dtype=np.float32))
    act_t = torch.as_tensor(np.array(action_list, dtype=np.int64))
    mask_t = torch.as_tensor(np.array(mask_list, dtype=np.bool_))

    optimizer = torch.optim.Adam(
        list(trainer.shared.parameters())
        + list(trainer.policy_head.parameters())
        + list(trainer.value_head.parameters()),
        lr=lr,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
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

            features = trainer.shared(b_obs)
            logits = trainer.policy_head(features)
            logits = logits.masked_fill(~b_mask, -1e8)

            loss = loss_fn(logits, b_act)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(trainer.shared.parameters())
                + list(trainer.policy_head.parameters())
                + list(trainer.value_head.parameters()),
                1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == b_act).sum().item()
            batches += 1

        acc = correct / n * 100
        avg_loss = total_loss / max(1, batches)
        log(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.1f}%")

    trainer.save(save_path)
    log(f"Warm-started model saved to {save_path}")
    return trainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50,
                        help="Number of heuristic games to collect")
    parser.add_argument("--save", type=str, default="models/ppo_sts.pt",
                        help="Where to save warm-started model")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    save_path = os.path.join(_root, args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    collector = DemoCollector(max_games=args.games)

    coord = Coordinator()
    coord.register_state_change_callback(collector.on_state_change)
    coord.register_out_of_game_callback(collector.on_out_of_game)
    coord.register_command_error_callback(collector.on_error)

    log(f"Starting demo collection: {args.games} games")
    coord.signal_ready()

    try:
        coord.run()
    except StopIteration:
        pass

    n = len(collector.observations)
    log(f"Collection done: {collector.games_done} games, {n} samples")

    if n < 50:
        log("Too few samples, skipping training")
        return

    train_supervised(
        collector.observations, collector.actions, collector.masks,
        save_path, epochs=args.epochs,
    )
    log("=== BEHAVIOR CLONE COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
