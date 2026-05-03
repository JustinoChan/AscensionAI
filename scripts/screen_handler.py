"""
screen_handler.py — Shared non-combat screen handling for all training scripts.

Provides:
  - auto_handle_screen(): unified handler used by rollout_worker, train_ppo,
    and train_bc_ppo
  - Helper functions for picking optimal choices on each screen type,
    also used by behavior_clone's heuristic_action()
"""
from __future__ import annotations

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
def pick_card_reward(choice_list: list) -> int:
    lower = [str(c).lower() for c in choice_list]
    for i, c in enumerate(lower):
        if c in GOOD_CARDS:
            return i
    for i, c in enumerate(lower):
        if c in OK_CARDS:
            return i
    for i, c in enumerate(lower):
        if "skip" in c:
            return i
    return 0


def pick_combat_reward_obj(rewards: list, potions_full: bool = False) -> int:
    """Pick reward from CombatReward objects. Returns -1 if nothing pickable."""
    for p in ("RELIC", "SAPPHIRE_KEY", "EMERALD_KEY",
              "GOLD", "STOLEN_GOLD", "POTION", "CARD"):
        if p == "POTION" and potions_full:
            continue
        for i, r in enumerate(rewards):
            rt = getattr(r, "reward_type", None)
            if rt is not None and rt.name == p:
                return i
    return -1


def pick_combat_reward_str(choice_list: list, potions_full: bool = False) -> int:
    """Pick reward from string choice_list. Returns -1 if nothing pickable."""
    for priority in ("relic", "gold", "potion", "card"):
        if priority == "potion" and potions_full:
            continue
        for i, c in enumerate(choice_list):
            if priority in str(c).lower():
                return i
    return -1


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
    avoid = {"busted crown", "coffee dripper", "sozu", "runic dome"}
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
    for i, c in enumerate(choice_list):
        if str(c).lower().rstrip("+") in GOOD_CARDS:
            return i
    for i, c in enumerate(choice_list):
        if str(c).lower().rstrip("+") in OK_CARDS:
            return i
    return 0


def _pick_grid_match(choice_list: list, scr) -> Optional[int]:
    """Pick card for a matching event (e.g. Match and Keep).

    If a card was already selected, find its duplicate in the remaining choices.
    Otherwise pick the first card that has a duplicate somewhere in the list.
    Returns None if no match is possible (caller should leave).
    """
    selected = list(getattr(scr, "selected_cards", []) or []) if scr else []
    lower = [str(c).lower().rstrip("+") for c in choice_list]

    if selected:
        last = selected[-1]
        target = str(getattr(last, "name", last)).lower().rstrip("+")
        for i, name in enumerate(lower):
            if name == target:
                return i

    from collections import Counter
    counts = Counter(lower)
    for i, name in enumerate(lower):
        if counts[name] >= 2:
            return i
    return None


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
    "E": 5, "?": 3, "$": 2, "M": 1, "T": 4, "R": 0,
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

        if sym == "E":
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
        elif sym == "M":
            if hp_pct < 0.3:
                score -= 3

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

    heuristic_all=True:  handle every screen heuristically (rollout workers).
    heuristic_all=False: handle only mechanical screens; return None for
                         decision screens so the RL policy can choose.

    Decision screens (RL-only when heuristic_all=False):
      CARD_REWARD, REST, BOSS_REWARD, HAND_SELECT, MAP
    """
    in_combat = bool(getattr(gs, "in_combat", False))
    choice_list = list(getattr(gs, "choice_list", []) or [])
    proceed_avail = bool(getattr(gs, "proceed_available", False))
    cancel_avail = bool(getattr(gs, "cancel_available", False))
    scr = getattr(gs, "screen", None)

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
        if choice_list:
            for_upgrade = bool(getattr(scr, "for_upgrade", False)) if scr else False
            for_transform = bool(getattr(scr, "for_transform", False)) if scr else False
            any_number = bool(getattr(scr, "any_number", False)) if scr else False
            if for_upgrade:
                unsel = [(i, c) for i, c in enumerate(choice_list)
                         if str(c).lower() not in selected_names]
                if unsel:
                    best_idx = _pick_from_unselected(unsel, scr)
                    return ChooseAction(choice_index=best_idx)
                return ChooseAction(choice_index=_pick_grid_upgrade(choice_list))
            if for_transform:
                return ChooseAction(choice_index=0)
            if any_number:
                if proceed_avail:
                    return Action("proceed")
                return ChooseAction(choice_index=0)
            unselected = [(i, c) for i, c in enumerate(choice_list)
                          if str(c).lower() not in selected_names]
            if not unselected:
                unselected = list(enumerate(choice_list))
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
                return Action("proceed")
            return Action("state")
        if heuristic_all:
            return ChooseAction(choice_index=pick_map(choice_list, gs))
        return None

    # ---- COMBAT_REWARD ----
    if screen_name == "COMBAT_REWARD":
        potions_full = bool(getattr(gs, "are_potions_full", lambda: False)())
        rewards = list(getattr(scr, "rewards", []) or []) if scr else []
        if rewards:
            idx = pick_combat_reward_obj(rewards, potions_full)
            if idx >= 0:
                return ChooseAction(choice_index=idx)
            return Action("proceed") if proceed_avail else Action("state")
        if choice_list:
            idx = pick_combat_reward_str(choice_list, potions_full)
            if idx >= 0:
                return ChooseAction(choice_index=idx)
            return Action("proceed") if proceed_avail else Action("state")
        return Action("proceed") if proceed_avail else Action("state")

    # ---- SHOP ----
    if screen_name == "SHOP_ROOM":
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
            return ChooseAction(choice_index=0)
        options = list(getattr(scr, "options", []) or []) if scr else []
        if options:
            return ChooseAction(choice_index=0)
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
        if heuristic_all and choice_list:
            return ChooseAction(choice_index=pick_card_reward(choice_list))
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
