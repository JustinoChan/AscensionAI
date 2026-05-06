"""
screen_handler.py — Shared non-combat screen handling for all training scripts.

Provides:
  - auto_handle_screen(): unified handler used by rollout_worker, train_ppo,
    and train_bc_ppo
  - Helper functions for picking optimal choices on each screen type,
    also used by behavior_clone's heuristic_action()
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from spirecomm.communication.action import Action, ChooseAction


# ---------------------------------------------------------------------------
# Card knowledge
# ---------------------------------------------------------------------------
GOOD_CARDS = frozenset({
    # Powers — scaling / build-defining
    "inflame", "demon form", "metallicize", "barricade", "corruption",
    "feel no pain", "dark embrace", "evolve", "berserk",
    # Skills — universally strong
    "shrug it off", "offering", "battle trance", "impervious",
    "disarm", "shockwave", "burning pact", "double tap",
    "spot weakness",
    # Attacks — high impact
    "uppercut", "headbutt", "feed", "reaper", "whirlwind",
    "immolate", "fiend fire", "limit break",
    # Colorless
    "apotheosis",
    "hand of greed", "master of strategy", "dark shackles", "panic button",
    "finesse", "flash of steel", "trip",
})
OK_CARDS = frozenset({
    # Common attacks
    "anger", "cleave", "thunderclap", "iron wave", "body slam",
    "heavy blade", "pommel strike", "sword boomerang",
    # Uncommon attacks
    "carnage", "pummel", "dropkick", "blood for blood",
    "hemokinesis", "sever soul", "bludgeon",
    # Skills
    "flame barrier", "second wind", "ghostly armor", "clothesline",
    "power through", "true grit", "entrench", "intimidate",
    "sentinel", "seeing red", "warcry", "armaments", "exhume",
    # Powers
    "fire breathing", "brutality", "rage", "juggernaut",
    "combust", "bloodletting", "flex", "twin strike", "perfected strike",
})
PREMIUM_CARDS = frozenset({
    "offering", "immolate", "feed", "reaper", "impervious",
    "demon form", "corruption", "barricade", "apotheosis",
})
EARLY_ATTACKS = frozenset({
    "pommel strike", "anger", "cleave", "twin strike", "headbutt",
    "carnage", "hemokinesis", "uppercut", "immolate", "bludgeon",
})
AVOID_CARDS = frozenset({
    "clash", "berserk", "havoc", "wild strike", "searing blow",
    "fire breathing", "rupture", "barricade",
})
JUNK_CARDS = frozenset({
    "wound", "burn", "dazed", "slimed", "void",
    "regret", "shame", "doubt", "pain", "decay", "parasite",
    "normality", "clumsy", "injury", "writhe", "pride",
    "ascendersbane", "necronomicurse",
})

_PURGE_PRIORITY = {
    "wound": 10, "burn": 10, "dazed": 10, "slimed": 10, "void": 10,
    "normality": 10, "pain": 10, "decay": 10, "parasite": 10,
    "regret": 9, "shame": 9, "doubt": 9, "clumsy": 9, "injury": 9,
    "writhe": 9, "pride": 9, "necronomicurse": 9, "ascendersbane": 9,
    "strike": 5, "defend": 4,
}


# ---------------------------------------------------------------------------
# Per-screen pick helpers
# ---------------------------------------------------------------------------
def _act_floor_index(gs) -> int:
    floor = int(getattr(gs, "floor", 0) or 0)
    act = int(getattr(gs, "act", 1) or 1)
    return max(1, floor - {1: 0, 2: 17, 3: 34}.get(act, 0))


def pick_card_reward(choice_list: list, gs=None) -> int:
    lower = [str(c).lower() for c in choice_list]
    act = int(getattr(gs, "act", 1) or 1) if gs is not None else 1
    act_floor = _act_floor_index(gs) if gs is not None else 1
    best_idx, best_score = 0, -999.0
    skip_idx = None
    for i, c in enumerate(lower):
        if "skip" in c:
            skip_idx = i
            continue
        name = c.rstrip("+").strip()
        score = 0.0
        if name in PREMIUM_CARDS:
            score += 10.0
        if name in GOOD_CARDS:
            score += 7.0
        if name in OK_CARDS:
            score += 4.0
        if name in EARLY_ATTACKS:
            score += 3.0 if act == 1 and act_floor <= 8 else 1.0
        if name in AVOID_CARDS:
            score -= 5.0
        if name in JUNK_CARDS:
            score -= 20.0
        if act == 1 and act_floor <= 5 and name in {
            "shrug it off", "armaments", "true grit", "ghostly armor",
            "power through",
        }:
            score -= 1.5
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score <= 0.0 and skip_idx is not None:
        return skip_idx
    return best_idx


_CARD_REWARD_OPEN_KEYS: set[tuple[int, int]] = set()
_SHOP_VISITED_KEYS: set[tuple[int, int]] = set()
_LAST_SCREEN_POS: tuple[int, int] | None = None


def _screen_memory_key(gs) -> tuple[int, int]:
    return (
        int(getattr(gs, "act", 0) or 0),
        int(getattr(gs, "floor", -1) or -1),
    )


def _reset_screen_memory_if_new_run(gs) -> None:
    """Clear per-run screen memory when a worker starts a new run."""
    global _LAST_SCREEN_POS
    pos = _screen_memory_key(gs)
    if _LAST_SCREEN_POS is not None:
        last_act, last_floor = _LAST_SCREEN_POS
        act, floor = pos
        if floor <= 1 and last_floor > 1:
            _CARD_REWARD_OPEN_KEYS.clear()
            _SHOP_VISITED_KEYS.clear()
            _MATCH_MEMORY.clear()
        elif act < last_act or (act == last_act and floor < last_floor):
            _CARD_REWARD_OPEN_KEYS.clear()
            _SHOP_VISITED_KEYS.clear()
            _MATCH_MEMORY.clear()
    _LAST_SCREEN_POS = pos


def _reward_type_name(reward) -> str:
    rt = getattr(reward, "reward_type", None)
    return str(getattr(rt, "name", rt) or "").upper()


def pick_combat_reward_obj(
    rewards: list,
    potions_full: bool = False,
    skip_card: bool = False,
) -> int:
    """Pick reward from CombatReward objects. Returns -1 if nothing pickable."""
    for p in ("RELIC", "SAPPHIRE_KEY", "EMERALD_KEY",
              "GOLD", "STOLEN_GOLD", "POTION", "CARD"):
        if p == "POTION" and potions_full:
            continue
        if p == "CARD" and skip_card:
            continue
        for i, r in enumerate(rewards):
            if _reward_type_name(r) == p:
                return i
    return -1


def pick_combat_reward_str(
    choice_list: list,
    potions_full: bool = False,
    skip_card: bool = False,
) -> int:
    """Pick reward from string choice_list. Returns -1 if nothing pickable."""
    for priority in ("relic", "gold", "potion", "card"):
        if priority == "potion" and potions_full:
            continue
        if priority == "card" and skip_card:
            continue
        for i, c in enumerate(choice_list):
            if priority in str(c).lower():
                return i
    return -1


def _available_commands(gs) -> set[str]:
    return set(getattr(gs, "available_commands", []) or [])


def _can_choose(gs) -> bool:
    commands = _available_commands(gs)
    return not commands or "choose" in commands


def _proceed_or_state(gs) -> Action:
    commands = _available_commands(gs)
    if "proceed" in commands:
        return Action("proceed")
    if "confirm" in commands:
        return Action("confirm")
    if not commands and bool(getattr(gs, "proceed_available", False)):
        return Action("proceed")
    return Action("state")


def _cancel_or_state(gs) -> Action:
    commands = _available_commands(gs)
    for cmd in ("cancel", "leave", "return", "skip"):
        if cmd in commands:
            return Action(cmd)
    if not commands and bool(getattr(gs, "cancel_available", False)):
        return Action("leave")
    return Action("state")


def recover_from_command_error(err: str) -> Action:
    """Pick a conservative valid command after CommunicationMod rejects one."""
    lower = str(err).lower()
    possible: set[str] = set()
    match = re.search(r"possible commands:\s*\[([^\]]+)\]", lower)
    if match:
        possible = {p.strip() for p in match.group(1).split(",") if p.strip()}

    # The common post-combat loop is invalid proceed while map choices are
    # available. Choosing index 0 advances; returning just reopens the loop.
    if "choose" in possible and ("invalid command: proceed" in lower or "proceed" in lower):
        return ChooseAction(choice_index=0)
    if "state" in possible:
        return Action("state")
    if "return" in possible:
        return Action("return")
    if "cancel" in possible:
        return Action("cancel")
    if "leave" in possible:
        return Action("leave")
    if "skip" in possible:
        return Action("skip")
    if "choose" in possible:
        return ChooseAction(choice_index=0)

    # Bare "wait" is not valid for this CommunicationMod build; it requires
    # an argument, so polling state is the safest no-op fallback.
    if "wait" in possible:
        return Action("state")
    if "proceed" in lower and "choose" in lower:
        return ChooseAction(choice_index=0)
    return Action("state")


def pick_rest(choice_list: list, gs) -> int:
    lower = [str(c).lower() for c in choice_list]
    hp_pct = int(getattr(gs, "current_hp", 0) or 0) / max(
        1, int(getattr(gs, "max_hp", 1) or 1))
    act = int(getattr(gs, "act", 0) or 0)
    floor = int(getattr(gs, "floor", 0) or 0)
    pre_boss = floor >= {1: 15, 2: 32, 3: 49}.get(act, 999)
    heal_threshold = 0.7 if pre_boss else 0.6
    if hp_pct < heal_threshold and "rest" in lower:
        return lower.index("rest")
    if "smith" in lower:
        return lower.index("smith")
    if "rest" in lower:
        return lower.index("rest")
    return 0


def pick_event(choice_list: list, gs) -> int:
    lower = [str(c).lower() for c in choice_list]
    hp_pct = int(getattr(gs, "current_hp", 0) or 0) / max(
        1, int(getattr(gs, "max_hp", 1) or 1))
    if len(lower) > 3 and any(c.startswith("card") for c in lower):
        return pick_hand_select(choice_list)
    if hp_pct < 0.35:
        for i, c in enumerate(lower):
            if "leave" in c:
                return i
    for i, c in enumerate(lower):
        if "leave" not in c:
            return i
    return 0


def pick_boss_relic(choice_list: list) -> int:
    lower = [str(c).lower() for c in choice_list]
    avoid = {
        "busted crown", "coffee dripper", "sozu", "runic dome",
        "ectoplasm", "snecko eye", "velvet choker",
    }
    prefer = {
        "cursed key", "fusion hammer", "slavers collar", "mark of pain",
        "black star", "calling bell", "empty cage", "astrolabe",
        "pandoras box", "runic pyramid",
    }
    for i, c in enumerate(lower):
        if c in prefer:
            return i
    non_avoid = [i for i, c in enumerate(lower) if c not in avoid]
    return non_avoid[0] if non_avoid else 0


def pick_hand_select(choice_list: list) -> int:
    """Pick worst card for forced discard."""
    for i, c in enumerate(choice_list):
        name = str(c).lower().rstrip("+")
        if name in JUNK_CARDS:
            return i
    for i, c in enumerate(choice_list):
        name = str(c).lower().rstrip("+")
        if "strike" in name or "defend" in name:
            return i
    return 0


def pick_grid_card(choice_list: list) -> int:
    """Pick card for GRID purge (worst card first)."""
    best_idx, best_score = 0, -1
    for i, c in enumerate(choice_list):
        name = str(c).lower().rstrip("+")
        for key, score in _PURGE_PRIORITY.items():
            if key in name and score > best_score:
                best_score = score
                best_idx = i
    return best_idx


def _pick_grid_upgrade(choice_list: list) -> int:
    """Pick best card to upgrade from GRID."""
    upgrade_priority = [
        "bash", "uppercut", "shockwave", "offering", "armaments",
        "inflame", "demon form", "immolate", "feed", "limit break",
        "impervious", "corruption", "apotheosis",
    ]
    lower = [str(c).lower().rstrip("+") for c in choice_list]
    for card in upgrade_priority:
        if card in lower:
            return lower.index(card)
    for i, c in enumerate(choice_list):
        if str(c).lower().rstrip("+") in GOOD_CARDS:
            return i
    for i, c in enumerate(choice_list):
        if str(c).lower().rstrip("+") in OK_CARDS:
            return i
    return 0


_HIDDEN_MATCH_NAMES = frozenset({
    "", "?", "unknown", "face down", "facedown", "card", "hidden",
})
_MATCH_MEMORY: dict[tuple, dict] = {}


def _card_label(card) -> str:
    return str(getattr(card, "name", card) or "").lower().rstrip("+").strip()


def _is_hidden_match_label(name: str) -> bool:
    return name in _HIDDEN_MATCH_NAMES or name.startswith("unknown")


def _is_matching_grid(gs, scr, choice_list: list) -> bool:
    """Detect Match and Keep-style grids without catching normal card selects."""
    if scr is None:
        return False
    num_needed = int(getattr(scr, "num_cards", 1) or 1)
    if num_needed != 2:
        return False
    if any(bool(getattr(scr, attr, False)) for attr in (
        "any_number", "for_upgrade", "for_transform", "for_purge",
    )):
        return False
    n_choices = len(choice_list) or len(getattr(scr, "cards", []) or [])
    if n_choices < 4:
        return False
    current_action = str(getattr(gs, "current_action", "") or "").lower()
    if "match" in current_action or "keep" in current_action:
        return True
    # Fallback for CommunicationMod states that omit current_action: most
    # non-flagged 2-card GRID screens with a large board are matching events.
    return n_choices >= 6


def _match_key(gs, scr, n_choices: int) -> tuple:
    return (
        int(getattr(gs, "floor", -1) or -1),
        str(getattr(gs, "current_action", "") or ""),
        n_choices,
    )


def _pick_grid_match(choice_list: list, scr, gs=None) -> Optional[int]:
    """Pick one slot for a matching event (e.g. Match and Keep).

    CommunicationMod exposes this event as a GRID screen. If card names are
    visible, take known pairs. If they are hidden, rotate through unrevealed
    slots and remember revealed names from selected_cards instead of clicking
    the first two slots forever.
    """
    cards = list(getattr(scr, "cards", []) or []) if scr else []
    names = [_card_label(c) for c in (choice_list or cards)]
    n = len(names)
    if n == 0:
        return None

    if gs is None:
        key = ("fallback", n)
    else:
        key = _match_key(gs, scr, n)
    mem = _MATCH_MEMORY.setdefault(key, {
        "seen": {},
        "tried": set(),
        "pending": [],
    })

    selected = list(getattr(scr, "selected_cards", []) or []) if scr else []
    if not selected and len(mem["pending"]) >= 2:
        mem["pending"] = []
    for i, card in enumerate(selected):
        if i >= len(mem["pending"]):
            break
        name = _card_label(card)
        if name and not _is_hidden_match_label(name):
            mem["seen"][mem["pending"][i]] = name

    available = [i for i in range(n) if i not in mem["pending"]]
    if not available:
        return None

    if selected:
        target = _card_label(selected[-1])
        if target and not _is_hidden_match_label(target):
            for i in available:
                if i < len(names) and names[i] == target:
                    mem["pending"].append(i)
                    mem["tried"].add(i)
                    return i
            for i, name in mem["seen"].items():
                if i in available and name == target:
                    mem["pending"].append(i)
                    mem["tried"].add(i)
                    return i

        unknown = [i for i in available if i not in mem["tried"]]
        if not unknown:
            mem["tried"].difference_update(available)
            unknown = available
        idx = unknown[0]
        mem["pending"].append(idx)
        mem["tried"].add(idx)
        return idx

    visible = [
        (i, name) for i, name in enumerate(names)
        if name and not _is_hidden_match_label(name)
    ]
    counts = Counter(name for _, name in visible)
    for i, name in visible:
        if counts[name] >= 2:
            mem["pending"] = [i]
            mem["tried"].add(i)
            return i

    seen_by_name: dict[str, list[int]] = {}
    for i, name in mem["seen"].items():
        if 0 <= i < n:
            seen_by_name.setdefault(name, []).append(i)
    for indices in seen_by_name.values():
        if len(indices) >= 2:
            idx = sorted(indices)[0]
            mem["pending"] = [idx]
            mem["tried"].add(idx)
            return idx

    unknown = [i for i in range(n) if i not in mem["tried"]]
    if not unknown:
        mem["tried"].clear()
        unknown = list(range(n))
    idx = unknown[0]
    mem["pending"] = [idx]
    mem["tried"].add(idx)
    return idx


def _pick_from_unselected(unselected: list, scr) -> int:
    """From (index, card_name) pairs, pick the best card to remove/select."""
    for_purge = bool(getattr(scr, "for_purge", False)) if scr else False
    for_upgrade = bool(getattr(scr, "for_upgrade", False)) if scr else False
    if for_upgrade:
        for i, c in unselected:
            if str(c).lower().rstrip("+") in GOOD_CARDS:
                return i
        return unselected[0][0]
    best_idx, best_score = unselected[0][0], -1
    for i, c in unselected:
        name = str(c).lower().rstrip("+")
        for key, score in _PURGE_PRIORITY.items():
            if key in name and score > best_score:
                best_score = score
                best_idx = i
    return best_idx


_MAP_SYMBOL_SCORES = {
    "E": 5, "?": 2, "$": 2, "M": 1, "T": 4, "R": 0,
}


def pick_map(choice_list: list, gs) -> int:
    """Elite-seeking, health-aware map pathing for Ironclad."""
    hp = int(getattr(gs, "current_hp", 0) or 0)
    mhp = max(1, int(getattr(gs, "max_hp", 1) or 1))
    hp_pct = hp / mhp
    act = int(getattr(gs, "act", 0) or 0)
    scr = getattr(gs, "screen", None)
    next_nodes = list(getattr(scr, "next_nodes", []) or []) if scr else []
    n = len(choice_list)
    if not next_nodes or n == 0:
        return 0

    best_idx, best_score = 0, -999.0
    for i in range(min(n, len(next_nodes))):
        sym = getattr(next_nodes[i], "symbol", "?")
        score = float(_MAP_SYMBOL_SCORES.get(sym, 1))

        act_floor = _act_floor_index(gs)
        easy_budget = 3 if act == 1 else 2

        if sym == "M":
            if act_floor <= easy_budget:
                score += 5
            elif hp_pct < 0.45:
                score -= 3
            else:
                score += 1
        elif sym == "?":
            if act_floor <= easy_budget:
                score -= 3
            elif hp_pct < 0.45:
                score += 2
        elif sym == "E":
            if hp_pct < 0.4:
                score -= 8
            elif act == 1 and hp_pct > 0.6:
                score += 6
            elif hp_pct > 0.6:
                score += 4
        elif sym == "R":
            if hp_pct < 0.4:
                score += 10
            elif hp_pct < 0.6:
                score += 6
            else:
                score -= 2
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def pick_shop_item(choice_list: list, gs, scr) -> Optional[int]:
    """Pick an item to buy in the shop, or None to leave."""
    gold = int(getattr(gs, "gold", 0) or 0)
    if scr is None or gold < 50:
        return None
    prices: dict = {}
    for attr in ("cards", "potions", "relics"):
        for it in (getattr(scr, attr, None) or []):
            nm = getattr(it, "name", None)
            pr = getattr(it, "price", None)
            if nm is not None and pr is not None:
                prices[str(nm)] = int(pr)
    for i, item in enumerate(choice_list):
        item_str = str(item)
        name_lower = item_str.lower()
        if name_lower in ("purge",):
            continue
        price = prices.get(item_str)
        if price is None or price <= 0 or gold < price:
            continue
        if name_lower in GOOD_CARDS:
            return i
    for it in (getattr(scr, "relics", None) or []):
        relic_name = str(getattr(it, "name", ""))
        price = prices.get(relic_name)
        if price is not None and gold >= price:
            for i, c in enumerate(choice_list):
                if str(c) == relic_name:
                    return i
    return None


# ---------------------------------------------------------------------------
# Unified screen handler
# ---------------------------------------------------------------------------
def auto_handle_screen(
    gs, screen_name: str, *, heuristic_all: bool = False,
) -> Optional[Action]:
    """Handle non-combat STS screens.

    heuristic_all=True:  handle every screen heuristically (BC/demo fallback).
    heuristic_all=False: handle only mechanical screens; return None for
                         decision screens so the RL policy can choose.

    Decision screens (RL-only when heuristic_all=False):
      CARD_REWARD, REST, BOSS_REWARD, HAND_SELECT, MAP, EVENT
    """
    in_combat = bool(getattr(gs, "in_combat", False))
    choice_list = list(getattr(gs, "choice_list", []) or [])
    proceed_avail = bool(getattr(gs, "proceed_available", False))
    cancel_avail = bool(getattr(gs, "cancel_available", False))
    scr = getattr(gs, "screen", None)
    _reset_screen_memory_if_new_run(gs)

    if in_combat and screen_name == "NONE":
        return None

    # ---- CHEST ----
    if screen_name == "CHEST":
        if scr and getattr(scr, "chest_open", False):
            return Action("proceed") if proceed_avail else Action("state")
        return ChooseAction(name="open")

    # ---- HAND_SELECT ----
    if screen_name == "HAND_SELECT":
        if scr and getattr(scr, "can_pick_zero", False) and proceed_avail:
            return Action("proceed")
        selected = list(getattr(scr, "selected_cards", []) or [])
        num_needed = int(getattr(scr, "num_cards", 1) or 1) if scr else 1
        if len(selected) >= num_needed:
            if proceed_avail:
                return Action("proceed")
            return Action("state")
        cards = choice_list or [c.name for c in getattr(scr, "cards", []) or []]
        if heuristic_all:
            if cards:
                return ChooseAction(choice_index=pick_hand_select(cards))
            if cancel_avail:
                return Action("leave")
            return ChooseAction(choice_index=0)
        if not cards:
            if cancel_avail:
                return Action("leave")
        return None

    # ---- GRID ----
    if screen_name == "GRID":
        if scr and getattr(scr, "confirm_up", False):
            return Action("proceed")
        num_needed = int(getattr(scr, "num_cards", 1) or 1) if scr else 1
        selected = list(getattr(scr, "selected_cards", []) or []) if scr else []
        already = len(selected)
        if already >= num_needed:
            if proceed_avail:
                return Action("proceed")
            return Action("state")
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
                    return ChooseAction(choice_index=idx)
                if proceed_avail:
                    return Action("proceed")
                if cancel_avail:
                    return Action("leave")
                return Action("state")
            if for_upgrade:
                unsel = [(i, c) for i, c in enumerate(grid_choices)
                         if str(c).lower() not in selected_names]
                if unsel:
                    best_idx = _pick_from_unselected(unsel, scr)
                    return ChooseAction(choice_index=best_idx)
                return ChooseAction(choice_index=_pick_grid_upgrade(grid_choices))
            if for_transform:
                return ChooseAction(choice_index=0)
            if any_number:
                if proceed_avail:
                    return Action("proceed")
                return ChooseAction(choice_index=0)
            unselected = [(i, c) for i, c in enumerate(grid_choices)
                          if str(c).lower() not in selected_names]
            if not unselected:
                unselected = list(enumerate(grid_choices))
            best_idx = _pick_from_unselected(unselected, scr)
            return ChooseAction(choice_index=best_idx)
        if proceed_avail:
            return Action("proceed")
        if cancel_avail:
            return Action("leave")
        return Action("state")

    # ---- MAP ----
    if screen_name == "MAP":
        boss_avail = scr and getattr(scr, "boss_available", False)
        if boss_avail and not choice_list:
            return ChooseAction(name="boss")
        if not choice_list:
            if proceed_avail:
                return _proceed_or_state(gs)
            return Action("state")
        if heuristic_all:
            return ChooseAction(choice_index=pick_map(choice_list, gs))
        return None

    # ---- COMBAT_REWARD ----
    if screen_name == "COMBAT_REWARD":
        potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())
        rewards = list(getattr(scr, "rewards", []) or []) if scr else []
        floor_key = _screen_memory_key(gs)
        skip_card_reward = floor_key in _CARD_REWARD_OPEN_KEYS
        if rewards:
            idx = pick_combat_reward_obj(rewards, potions_full, skip_card_reward)
            if idx >= 0:
                if _reward_type_name(rewards[idx]) == "CARD":
                    _CARD_REWARD_OPEN_KEYS.add(floor_key)
                return ChooseAction(choice_index=idx)
            return _proceed_or_state(gs) if proceed_avail else Action("state")
        if choice_list:
            idx = pick_combat_reward_str(choice_list, potions_full, skip_card_reward)
            if idx >= 0:
                if "card" in str(choice_list[idx]).lower():
                    _CARD_REWARD_OPEN_KEYS.add(floor_key)
                return ChooseAction(choice_index=idx)
            # After clicking Proceed on the reward screen, CommunicationMod can
            # expose map node choices while still reporting COMBAT_REWARD. Pick
            # a node instead of bouncing map -> return -> reward forever.
            if _can_choose(gs):
                return ChooseAction(choice_index=pick_map(choice_list, gs))
            return _proceed_or_state(gs) if proceed_avail else Action("state")
        return _proceed_or_state(gs) if proceed_avail else Action("state")

    # ---- SHOP ----
    if screen_name == "SHOP_ROOM":
        shop_key = _screen_memory_key(gs)
        if choice_list and shop_key not in _SHOP_VISITED_KEYS:
            lower = [str(c).lower() for c in choice_list]
            for i, c in enumerate(lower):
                if "shop" in c or "merchant" in c:
                    _SHOP_VISITED_KEYS.add(shop_key)
                    return ChooseAction(choice_index=i)
        if proceed_avail:
            return Action("proceed")
        if cancel_avail:
            return Action("cancel")
        return Action("proceed")

    if screen_name == "SHOP_SCREEN":
        gold = int(getattr(gs, "gold", 0) or 0)
        if scr is not None and gold >= 30:
            for card in (getattr(scr, "cards", None) or []):
                name = str(getattr(card, "name", "") or "")
                price = int(getattr(card, "price", 999) or 999)
                if name.lower() in GOOD_CARDS and gold >= price:
                    return ChooseAction(name=name)
            if getattr(scr, "purge_available", False):
                purge_cost = int(getattr(scr, "purge_cost", 999) or 999)
                if gold >= purge_cost:
                    return ChooseAction(name="purge")
            for card in (getattr(scr, "cards", None) or []):
                name = str(getattr(card, "name", "") or "")
                price = int(getattr(card, "price", 999) or 999)
                if name.lower() in OK_CARDS and gold >= price:
                    return ChooseAction(name=name)
            for relic in (getattr(scr, "relics", None) or []):
                name = str(getattr(relic, "name", "") or "")
                price = int(getattr(relic, "price", 999) or 999)
                if gold >= price:
                    return ChooseAction(name=name)
            potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())
            if not potions_full:
                for pot in (getattr(scr, "potions", None) or []):
                    name = str(getattr(pot, "name", "") or "")
                    price = int(getattr(pot, "price", 999) or 999)
                    if gold >= price:
                        return ChooseAction(name=name)
        return Action("cancel")

    # ---- EVENT ----
    if screen_name == "EVENT":
        if choice_list:
            if heuristic_all:
                return ChooseAction(choice_index=pick_event(choice_list, gs))
            return None
        options = list(getattr(scr, "options", []) or []) if scr else []
        if options:
            enabled = [
                (i, opt) for i, opt in enumerate(options)
                if not bool(getattr(opt, "disabled", False))
            ]
            if not enabled:
                if proceed_avail:
                    return Action("proceed")
                if cancel_avail:
                    return Action("leave")
                return Action("state")
            if heuristic_all:
                idx, opt = enabled[0]
                choice_idx = getattr(opt, "choice_index", None)
                return ChooseAction(choice_index=choice_idx if choice_idx is not None else idx)
            return None
        if proceed_avail:
            return Action("proceed")
        return Action("state")

    # ---- Decision screens: heuristic_all handles, otherwise RL decides ----

    if screen_name == "BOSS_REWARD":
        if heuristic_all and choice_list:
            return ChooseAction(choice_index=pick_boss_relic(choice_list))
        if heuristic_all or not choice_list:
            return Action("proceed") if proceed_avail else Action("state")
        return None

    if screen_name == "CARD_REWARD":
        if choice_list:
            _CARD_REWARD_OPEN_KEYS.add(_screen_memory_key(gs))
        if heuristic_all and choice_list:
            return ChooseAction(choice_index=pick_card_reward(choice_list, gs))
        if heuristic_all or not choice_list:
            return Action("proceed") if proceed_avail else Action("state")
        return None

    if screen_name == "REST":
        if heuristic_all and choice_list:
            return ChooseAction(choice_index=pick_rest(choice_list, gs))
        if heuristic_all or not choice_list:
            return Action("proceed") if proceed_avail else Action("state")
        return None

    # ---- Generic fallback ----
    if heuristic_all and not in_combat:
        if choice_list:
            return ChooseAction(choice_index=0)
        if proceed_avail:
            return Action("proceed")
        if cancel_avail:
            return Action("leave")
        return Action("state")

    if not choice_list and not cancel_avail:
        if proceed_avail:
            return Action("proceed")
        return Action("state")
    if not choice_list and cancel_avail and not proceed_avail:
        return Action("leave")

    if heuristic_all:
        if choice_list:
            return ChooseAction(choice_index=0)
        if proceed_avail:
            return Action("proceed")

    return None
