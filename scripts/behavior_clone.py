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
sys.stdout = sys.stderr

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

DEBUG_LOG = os.path.join(_root, "bc_debug.log")

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
)

log("Imports done")


# ---------------------------------------------------------------------------
# Heuristic action picker
# ---------------------------------------------------------------------------
def _screen_name(gs) -> str:
    st = getattr(gs, "screen_type", None)
    name = getattr(st, "name", st) if st is not None else "NONE"
    return str(name) if name else "NONE"


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


def score_card(card, incoming: int) -> float:
    name = (getattr(card, "name", "") or "").lower()
    cost = int(getattr(card, "cost", 1) or 1)
    ctype = str(getattr(card, "type", "") or "").upper()
    dmg = int(getattr(card, "damage", 0) or 0)
    blk = int(getattr(card, "block", 0) or 0)

    if dmg == 0 and ("strike" in name or "bash" in name):
        dmg = 6 if "strike" in name else 8
    if blk == 0 and "defend" in name:
        blk = 5

    s = 0.0
    if incoming >= 10:
        s += blk * 0.9
    else:
        s += blk * 0.2
    if dmg > 0:
        s += (dmg / max(1, cost)) * 1.0
    if "bash" in name:
        s += 4.0
    if ctype == "POWER":
        s += 3.0
        if incoming >= 12:
            s -= 3.5
    s -= max(0, cost - 1) * 0.3
    return s


def pick_target(monsters, prefer_low_hp=True):
    alive = living_monsters(monsters or [])
    if not alive:
        return None
    if prefer_low_hp:
        alive.sort(key=lambda m: int(getattr(m, "current_hp", 999) or 999))
    return alive[0]


# ---------------------------------------------------------------------------
# Decision-screen heuristic helpers
# ---------------------------------------------------------------------------
_GOOD_CARDS = {
    "inflame", "shrug it off", "anger", "uppercut", "offering",
    "battle trance", "headbutt", "feed", "impervious", "demon form",
    "metallicize", "reaper", "limit break", "barricade", "corruption",
}
_OK_CARDS = {
    "cleave", "thunderclap", "iron wave", "body slam", "carnage",
    "pummel", "flame barrier", "feel no pain", "dark embrace",
    "second wind", "ghostly armor", "disarm", "clothesline",
    "power through", "true grit", "fire breathing",
}
_JUNK_CARDS = {
    "wound", "burn", "dazed", "slimed", "void",          # status
    "regret", "shame", "doubt", "pain", "decay", "parasite",  # curse
}


def _pick_card_reward_idx(choice_list: list) -> int:
    """Pick the best card from a reward screen."""
    lower = [str(c).lower() for c in choice_list]
    for i, c in enumerate(lower):
        if c in _GOOD_CARDS:
            return i
    for i, c in enumerate(lower):
        if c in _OK_CARDS:
            return i
    # Nothing exciting — skip if available, else take first
    for i, c in enumerate(lower):
        if "skip" in c:
            return i
    return 0


def _pick_combat_reward_idx(choice_list: list) -> int:
    """Pick reward in priority: relic > gold > potion > card."""
    for priority in ("relic", "gold", "potion", "card"):
        for i, c in enumerate(choice_list):
            if priority in str(c).lower():
                return i
    return 0


def _pick_rest_idx(choice_list: list, gs) -> int:
    """Campfire: rest if low HP, else smith."""
    lower = [str(c).lower() for c in choice_list]
    hp_pct = int(getattr(gs, "current_hp", 0) or 0) / max(1, int(getattr(gs, "max_hp", 1) or 1))
    if hp_pct < 0.55 and "rest" in lower:
        return lower.index("rest")
    if "smith" in lower:
        return lower.index("smith")
    if "rest" in lower:
        return lower.index("rest")
    return 0


def _pick_event_idx(choice_list: list, gs) -> int:
    """Event: avoid risky options when low HP, else pick first non-leave."""
    lower = [str(c).lower() for c in choice_list]
    hp_pct = int(getattr(gs, "current_hp", 0) or 0) / max(1, int(getattr(gs, "max_hp", 1) or 1))
    if hp_pct < 0.35:
        for i, c in enumerate(lower):
            if "leave" in c:
                return i
    for i, c in enumerate(lower):
        if "leave" not in c:
            return i
    return 0


def _pick_boss_relic_idx(choice_list: list) -> int:
    """Boss relic: prefer energy relics, avoid Busted Crown/Coffee Dripper."""
    lower = [str(c).lower() for c in choice_list]
    _avoid = {"busted crown", "coffee dripper", "sozu", "runic dome"}
    # Prefer relics NOT in _avoid
    non_avoid = [i for i, c in enumerate(lower) if c not in _avoid]
    if non_avoid:
        return non_avoid[0]
    return 0


def _pick_hand_select_idx(choice_list: list) -> int:
    """Pick worst card from HAND_SELECT (forced discard)."""
    for i, c in enumerate(choice_list):
        name = str(c).lower().rstrip("+")
        if name in _JUNK_CARDS:
            return i
    for i, c in enumerate(choice_list):
        name = str(c).lower().rstrip("+")
        if "strike" in name or "defend" in name:
            return i
    return 0


def _pick_grid_card_idx(choice_list: list) -> int:
    """Pick a card for GRID (purge = pick worst, upgrade = pick best).

    Without knowing the operation type, prioritise removing junk/basics.
    If nothing junk-like, pick the first card (reasonable for upgrades too).
    """
    _PURGE_PRIORITY = {
        "wound": 10, "burn": 10, "dazed": 10, "slimed": 10, "void": 10,
        "regret": 9, "shame": 9, "doubt": 9, "pain": 9, "decay": 9,
        "strike": 5, "defend": 4,
    }
    best_idx, best_score = 0, -1
    for i, c in enumerate(choice_list):
        name = str(c).lower().rstrip("+")
        for key, score in _PURGE_PRIORITY.items():
            if key in name and score > best_score:
                best_score = score
                best_idx = i
    return best_idx


def _pick_shop_item_idx(choice_list: list, gs, scr) -> Optional[int]:
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

    # First pass: buy a good card/relic we can afford
    for i, item in enumerate(choice_list):
        item_str = str(item)
        name_lower = item_str.lower()
        if name_lower in ("purge",):
            continue
        price = prices.get(item_str)
        if price is None or price <= 0 or gold < price:
            continue
        if name_lower in _GOOD_CARDS:
            return i

    # Second pass: relics (any relic is usually worth buying)
    for it in (getattr(scr, "relics", None) or []):
        relic_name = str(getattr(it, "name", ""))
        price = prices.get(relic_name)
        if price is not None and gold >= price:
            for i, c in enumerate(choice_list):
                if str(c) == relic_name:
                    return i

    return None


def _pick_map_idx(choice_list: list, gs) -> int:
    """MAP path selection: avoid elites when low, prefer middle when healthy."""
    hp = int(getattr(gs, "current_hp", 0) or 0)
    mhp = max(1, int(getattr(gs, "max_hp", 1) or 1))
    n = len(choice_list)
    # Low HP → safest path (first), healthy → middle path
    if hp / mhp < 0.45:
        return 0
    return min(n // 2, n - 1)


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
    if screen in {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}:
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return Action("state"), _NOOP

    # --- Mechanical: CHEST — always open ---
    if screen == "CHEST":
        if scr and getattr(scr, "chest_open", False):
            return (Action("proceed"), _PROCEED) if proceed_avail else (Action("state"), _NOOP)
        return ChooseAction(name="open"), _CHOOSE_START

    # --- Mechanical: HAND_SELECT with can_pick_zero ---
    if screen == "HAND_SELECT":
        if scr and getattr(scr, "can_pick_zero", False) and proceed_avail:
            return Action("proceed"), _PROCEED
        # Forced selection — pick worst card
        if choice_list:
            idx = _pick_hand_select_idx(choice_list)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        return ChooseAction(choice_index=0), _CHOOSE_START

    # --- Mechanical: GRID confirmation ---
    if screen == "GRID":
        if scr and getattr(scr, "confirm_up", False):
            return Action("proceed"), _PROCEED
        if proceed_avail and not choice_list:
            return Action("proceed"), _PROCEED
        # Card selection (purge/upgrade/transform)
        if choice_list:
            idx = _pick_grid_card_idx(choice_list)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        return ChooseAction(choice_index=0), _CHOOSE_START

    # --- Mechanical: MAP boss-only ---
    if screen == "MAP":
        boss_avail = scr and getattr(scr, "boss_available", False)
        if boss_avail and not choice_list:
            return ChooseAction(name="boss"), _CHOOSE_START
        if choice_list:
            idx = _pick_map_idx(choice_list, gs)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- BOSS_REWARD ---
    if screen == "BOSS_REWARD":
        if choice_list:
            idx = _pick_boss_relic_idx(choice_list)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- CARD_REWARD ---
    if screen == "CARD_REWARD":
        if choice_list:
            idx = _pick_card_reward_idx(choice_list)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        if cancel_avail:
            return Action("leave"), _LEAVE
        return None, None

    # --- COMBAT_REWARD ---
    if screen == "COMBAT_REWARD":
        if choice_list:
            idx = _pick_combat_reward_idx(choice_list)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- REST ---
    if screen == "REST":
        if choice_list:
            idx = _pick_rest_idx(choice_list, gs)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- EVENT ---
    if screen == "EVENT":
        if choice_list:
            idx = _pick_event_idx(choice_list, gs)
            return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return None, None

    # --- SHOP_ROOM ---
    if screen == "SHOP_ROOM":
        gold = int(getattr(gs, "gold", 0) or 0)
        if choice_list and gold >= 75:
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
        if choice_list:
            idx = _pick_shop_item_idx(choice_list, gs, scr)
            if idx is not None:
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
        return Action("leave"), _LEAVE

    # --- COMBAT ---
    if in_combat:
        incoming = estimate_incoming(monsters)

        if play_avail and hand:
            best_card = None
            best_score = -1e9
            for card in hand:
                if not getattr(card, "is_playable", False):
                    continue
                s = score_card(card, incoming)
                if s > best_score:
                    best_score = s
                    best_card = card

            if best_card is not None:
                card_idx = hand.index(best_card)
                if card_idx < MAX_HAND:
                    if getattr(best_card, "has_target", False):
                        tgt = pick_target(monsters, prefer_low_hp=True)
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
    if not choice_list and not cancel_avail:
        if proceed_avail:
            return Action("proceed"), _PROCEED
        return Action("state"), _NOOP
    if not choice_list and cancel_avail and not proceed_avail:
        return Action("leave"), _LEAVE
    if choice_list:
        return ChooseAction(choice_index=0), _CHOOSE_START

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
        terminal = screen in {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}

        if not self.initialized:
            self.initialized = True
            log(f"Demo game #{self.games_done + 1} started, floor={getattr(gs, 'floor', '?')}")

        if terminal:
            self.games_done += 1
            self.initialized = False
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
