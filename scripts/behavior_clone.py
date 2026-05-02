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
)

from game_data import POTION_EFFECTS
from screen_handler import (
    GOOD_CARDS, OK_CARDS, JUNK_CARDS,
    pick_card_reward, pick_combat_reward_obj, pick_combat_reward_str,
    pick_rest, pick_event, pick_boss_relic, pick_hand_select,
    pick_grid_card, _pick_grid_upgrade, _pick_grid_match, pick_map,
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
    spawners = [m for m in alive
                if str(getattr(m, "monster_id", "")) in SPAWNER_IDS]
    if spawners:
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
    if screen in {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}:
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
        already = len(getattr(scr, "selected_cards", []) or []) if scr else 0
        if already >= num_needed:
            if proceed_avail:
                return Action("proceed"), _PROCEED
        if choice_list:
            for_purge = bool(getattr(scr, "for_purge", False)) if scr else False
            for_upgrade = bool(getattr(scr, "for_upgrade", False)) if scr else False
            for_transform = bool(getattr(scr, "for_transform", False)) if scr else False
            if for_purge:
                idx = pick_grid_card(choice_list)
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if for_upgrade:
                idx = _pick_grid_upgrade(choice_list)
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if for_transform:
                return ChooseAction(choice_index=0), _CHOOSE_START
            any_number = bool(getattr(scr, "any_number", False)) if scr else False
            if any_number:
                if proceed_avail:
                    return Action("proceed"), _PROCEED
                return ChooseAction(choice_index=0), _CHOOSE_START
            idx = _pick_grid_match(choice_list, scr)
            if idx is not None:
                return ChooseAction(choice_index=idx), _CHOOSE_START + idx
            if cancel_avail:
                return Action("leave"), _LEAVE
            if proceed_avail:
                return Action("proceed"), _PROCEED
        if proceed_avail:
            return Action("proceed"), _PROCEED
        if cancel_avail:
            return Action("leave"), _LEAVE
        return Action("proceed"), _PROCEED

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
            idx = pick_card_reward(choice_list)
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
            return ChooseAction(choice_index=0), _CHOOSE_START
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

        # -- Use potions when conditions warrant --
        hp = int(getattr(gs, "current_hp", 0) or 0)
        max_hp = max(1, int(getattr(gs, "max_hp", 1) or 1))
        hp_pct = hp / max_hp
        for k, pot in enumerate(potions[:MAX_POTIONS]):
            pot_name = str(getattr(pot, "name", "") or "")
            if pot_name in ("Potion Slot", "") or not getattr(pot, "can_use", False):
                continue
            effects = POTION_EFFECTS.get(pot_name, (0, 0, 0, 0, 0))
            deals_damage, gives_block, gives_str, gives_dex, heals = effects
            use = False
            if heals and hp_pct < 0.35:
                use = True
            elif gives_block and hp_pct < 0.5 and incoming > 10:
                use = True
            elif (deals_damage or gives_str) and incoming > 15:
                use = True
            elif gives_dex and incoming > 10:
                use = True
            if use:
                requires_target = getattr(pot, "requires_target", False)
                if requires_target:
                    tgt = pick_target(monsters, prefer_low_hp=True)
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
        terminal = screen in {"GAME_OVER", "VICTORY", "COMPLETE", "CREDITS"}

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
