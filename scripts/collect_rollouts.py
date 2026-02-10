from datetime import datetime
import time
import json
from typing import Any
import os


from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, PlayCardAction, ChooseAction, PotionAction, StartGameAction
from spirecomm.spire.character import PlayerClass
import random


LOG_PATH = r"C:\AscensionAI\rollouts.jsonl"
DEBUG_LOG_PATH = r"C:\AscensionAI\rollout_debug.log"

def debug_log(msg: str):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
        f.flush()


# ----------------------------
# GameState access helpers
# (works for spirecomm objects OR dict payloads)
# ----------------------------
def gs_attr(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def gs_screen_type(gs: Any) -> str | None:
    st = gs_attr(gs, "screen_type", None)
    if isinstance(st, dict):
        return st.get("name")
    return getattr(st, "name", None) if st is not None else None

def gs_choice_list(gs: Any) -> list:
    return gs_attr(gs, "choice_list", []) or []

def gs_proceed_available(gs: Any) -> bool:
    return bool(gs_attr(gs, "proceed_available", False))

def gs_in_combat(gs: Any) -> bool:
    # Prefer explicit in_combat if present; otherwise infer from room_phase/screen.
    v = gs_attr(gs, "in_combat", None)
    if v is not None:
        return bool(v)
    return str(gs_attr(gs, "room_phase", "")).upper() == "COMBAT"

def gs_combat_state(gs: Any) -> Any:
    return gs_attr(gs, "combat_state", None) or gs_attr(gs, "combatState", None)

def gs_hand(gs: Any) -> list:
    cs = gs_combat_state(gs)
    if cs is not None:
        return gs_attr(cs, "hand", []) or []
    return gs_attr(gs, "hand", []) or []

def gs_monsters(gs: Any) -> list:
    cs = gs_combat_state(gs)
    if cs is not None:
        return gs_attr(cs, "monsters", []) or []
    return gs_attr(gs, "monsters", []) or []

def gs_potions(gs: Any) -> list:
    return gs_attr(gs, "potions", []) or []

def gs_play_available(gs: Any) -> bool:
    return bool(gs_attr(gs, "play_available", False)) or bool(gs_attr(gs, "playAvailable", False))

def gs_end_available(gs: Any) -> bool:
    return bool(gs_attr(gs, "end_available", False)) or bool(gs_attr(gs, "endAvailable", False))

def gs_potion_available(gs: Any) -> bool:
    return bool(gs_attr(gs, "potion_available", False)) or bool(gs_attr(gs, "potionAvailable", False))

def gs_gold(gs: Any) -> int:
    return int(gs_attr(gs, "gold", 0) or 0)

def gs_floor(gs: Any) -> int:
    return int(gs_attr(gs, "floor", 0) or 0)

def gs_current_hp(gs: Any) -> int:
    return int(gs_attr(gs, "current_hp", 0) or 0)

def gs_max_hp(gs: Any) -> int:
    return int(gs_attr(gs, "max_hp", 0) or 0)

def gs_deck(gs: Any) -> list:
    return gs_attr(gs, "deck", []) or []

def gs_relics(gs: Any) -> list:
    return gs_attr(gs, "relics", []) or []

def gs_screen_state(gs: Any) -> Any:
    return gs_attr(gs, "screen", None) or gs_attr(gs, "screen_state", None) or gs_attr(gs, "screenState", None)
debug_log("ROLLOUT COLLECTOR STARTED (PHASE 1)")


def fingerprint(obs: dict) -> tuple:
    """Create a hashable fingerprint of observation to detect state changes"""
    cl = obs.get("choice_list") or []
    return (
        obs.get("screen_type"),
        obs.get("floor", 0),
        obs.get("gold", 0),
        obs.get("current_hp", 0),
        obs.get("deck_size", 0),
        obs.get("num_relics", 0),
        obs.get("in_combat", False),
        obs.get("num_monsters", 0),
        obs.get("hand_size", 0),
        obs.get("enemy_total_hp", 0),
        obs.get("alive_monsters", 0),
        tuple(cl),
    )


class RolloutCollector:
    def __init__(self):
        self.transitions = []

        # Delta reward tracking (not baseline)
        self.last_gold = 0
        self.last_hp = 0
        self.last_max_hp = 0
        self.last_deck_size = 0
        self.last_relics = 0

        # Reward accumulation across unlogged intermediate state updates
        self.pending_reward = 0.0

        # Combat shaping trackers
        self.last_in_combat = False
        self.last_enemy_total_hp = None
        self.last_alive_monsters = 0

        # Shop tracking
        self.shop_visited_floor = -1
        self.shop_bought_items = set()
        self.last_shop_action_time = 0.0
        self.last_screen = None  # Track screen changes for gating shop clears

        # --- exploration / anti-determinism ---
        self.epsilon = 0.08
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.02

        # --- potion throttling ---
        self.last_potion_use_time = 0.0
        self.min_seconds_between_potions = 6.0

    
    # ----------------------------
    # Target legality
    # ----------------------------
    @staticmethod
    def is_targetable_monster(m: Any) -> bool:
        if m is None:
            return False
        if getattr(m, "is_gone", False):
            return False
        hp = int(getattr(m, "current_hp", 0) or 0)
        return hp > 0

    def living_monsters(self, monsters: list) -> list:
        return [m for m in (monsters or []) if self.is_targetable_monster(m)]

    def pick_target(self, monsters: list, prefer_low_hp: bool = True):
        living = self.living_monsters(monsters)
        if not living:
            return None
        if prefer_low_hp:
            return min(living, key=lambda m: int(getattr(m, "current_hp", 0) or 0))
        return living[0]

    # ----------------------------
    # Incoming damage estimation
    # ----------------------------
    def estimate_incoming_damage(self, monsters: list) -> int:
        """Best-effort using move damage fields. Treat positive base/adjusted damage as incoming."""
        total = 0
        for m in self.living_monsters(monsters):
            hits = int(getattr(m, "move_hits", 1) or 1)
            dmg = getattr(m, "move_adjusted_damage", None)
            if dmg is None:
                dmg = getattr(m, "move_base_damage", None)
            try:
                dmg = int(dmg)
            except Exception:
                dmg = 0
            if dmg > 0:
                total += dmg * max(1, hits)
        return max(0, total)

    # ----------------------------
    # Potion gating
    # ----------------------------
    def should_use_potion_now(self, game_state, monsters: list) -> bool:
        now = time.time()
        if now - self.last_potion_use_time < self.min_seconds_between_potions:
            return False

        hp = int(getattr(game_state, "current_hp", 0) or 0)
        mhp = max(1, int(getattr(game_state, "max_hp", 1) or 1))
        hp_pct = hp / mhp

        incoming = self.estimate_incoming_damage(monsters)
        if hp_pct <= 0.35:
            return True
        if incoming >= max(10, int(0.18 * mhp)):
            return True
        return False

    # ----------------------------
    # Exploration / epsilon-greedy
    # ----------------------------
    def maybe_random_action(self, game_state, screen, choice_list, in_combat, play_avail, end_avail, hand, monsters):
        if random.random() > self.epsilon:
            return None

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if choice_list and screen in {"EVENT", "MAP", "CARD_REWARD", "COMBAT_REWARD", "SHOP_ROOM"}:
            return ChooseAction(choice_index=random.randrange(len(choice_list)))

        if in_combat and play_avail and hand:
            playable = [c for c in hand if getattr(c, "is_playable", False)]
            if playable:
                card = random.choice(playable)
                if getattr(card, "has_target", False):
                    tgt = self.pick_target(monsters, prefer_low_hp=True)
                    if tgt is not None:
                        return PlayCardAction(card=card, target_monster=tgt)
                return PlayCardAction(card=card)

        if in_combat and end_avail and random.random() < 0.25:
            return Action("end")

        return None

    # ----------------------------
    # Map selection
    # ----------------------------
    def pick_map_choice_index(self, game_state, choice_list):
        hp = int(getattr(game_state, "current_hp", 0) or 0)
        mhp = max(1, int(getattr(game_state, "max_hp", 1) or 1))
        hp_pct = hp / mhp

        n = len(choice_list)
        if n <= 1:
            return 0

        if hp_pct < 0.45:
            return 0

        mid = n // 2
        if n >= 3 and random.random() < 0.25:
            return random.choice([0, mid, n - 1])
        return mid

    # ----------------------------
    # Card scoring (cheap heuristic)
    # ----------------------------
    def score_card(self, card, game_state, incoming_dmg: int) -> float:
        name = (getattr(card, "name", "") or "").lower()
        cost = getattr(card, "cost", 1)
        try:
            cost = int(cost)
        except Exception:
            cost = 1

        ctype = str(getattr(card, "type", "") or "").upper()

        base_damage = int(getattr(card, "damage", 0) or 0)
        base_block = int(getattr(card, "block", 0) or 0)

        # fallbacks for vanilla cards
        if base_damage == 0 and ("strike" in name or "bash" in name):
            base_damage = 6 if "strike" in name else 8
        if base_block == 0 and "defend" in name:
            base_block = 5

        score = 0.0

        # defend when threatened
        if incoming_dmg >= 10:
            score += base_block * 0.9
        else:
            score += base_block * 0.2  # small value even when safe

        # damage is generally good, energy efficiency matters
        if base_damage > 0:
            score += (base_damage / max(1, cost)) * 1.0

        # bash is high value early
        if "bash" in name:
            score += 4.0

        # powers: mild preference unless danger is high
        if ctype == "POWER":
            score += 3.0
            if incoming_dmg >= 12:
                score -= 3.5

        # small penalty for expensive cards
        score -= max(0, cost - 1) * 0.3
        return score


    # ----------------------------
    # Combat helpers (static)
    # ----------------------------
    @staticmethod
    def is_effectively_alive(m: Any) -> bool:
        if getattr(m, "is_gone", False):
            return False
        if getattr(m, "half_dead", False):
            return True
        return getattr(m, "current_hp", 0) > 0

    @staticmethod
    def compute_enemy_stats(monsters: list) -> tuple[int, int]:
        total_hp = 0
        alive = 0
        for m in monsters:
            if RolloutCollector.is_effectively_alive(m):
                alive += 1
                total_hp += max(0, int(getattr(m, "current_hp", 0)))
        return total_hp, alive
    
    # ----------------------------
    # Helper Functions
    # ----------------------------
    def _lower_choices(self, choice_list):
        return [(c or "").strip().lower() for c in (choice_list or [])]

    def _find_choice_index_contains(self, choice_list, needles):
        lower = self._lower_choices(choice_list)
        for i, c in enumerate(lower):
            for n in needles:
                if n in c:
                    return i
        return None
    
    def has_empty_potion_slot(self, game_state) -> bool:
        potions = gs_potions(game_state)
        for p in potions:
            pid = (getattr(p, "id", "") or "").lower()
            name = (getattr(p, "name", "") or "").lower()
            # common empty slot markers
            if "potion slot" in pid or "potion slot" in name or name == "empty":
                return True
        return False



    def serialize_state(self, game_state):
        """Convert game state to serializable dict"""
        st = getattr(game_state, "screen_type", None)
        in_combat = gs_in_combat(game_state)

        monsters = gs_monsters(game_state)
        enemy_total_hp = 0
        alive_monsters = 0
        if in_combat and monsters:
            enemy_total_hp, alive_monsters = self.compute_enemy_stats(monsters)

        return {
            "screen_type": gs_screen_type(game_state),
            "floor": gs_floor(game_state),
            "gold": gs_gold(game_state),
            "current_hp": gs_current_hp(game_state),
            "max_hp": gs_max_hp(game_state),
            "deck_size": len(gs_deck(game_state)),
            "num_relics": len(gs_relics(game_state)),
            "in_combat": in_combat,

            # IMPORTANT: use alive_monsters, not raw list length
            "num_monsters": alive_monsters if in_combat else 0,
            "alive_monsters": alive_monsters if in_combat else 0,
            "enemy_total_hp": enemy_total_hp if in_combat else 0,

            "hand_size": len(gs_hand(game_state)) if in_combat else 0,
            "choice_list": gs_choice_list(game_state),
        }

    def serialize_action(self, action, game_state_where_chosen):
        """Convert action to serializable dict, using the state where it was chosen"""
        action_info = {"type": type(action).__name__}

        if isinstance(action, PlayCardAction):
            if action.card:
                action_info["card_name"] = action.card.name
                action_info["has_target"] = getattr(action.card, "has_target", False)
            if action.target_monster:
                action_info["target_monster"] = action.target_monster.name

        elif isinstance(action, PotionAction):
            if action.potion:
                action_info["potion_name"] = action.potion.name
            action_info["use"] = action.use

        elif isinstance(action, ChooseAction):
            action_info["choice_index"] = action.choice_index
            # Use actual choice from the state where choice was made
            if hasattr(game_state_where_chosen, "choice_list") and game_state_where_chosen.choice_list:
                if 0 <= action.choice_index < len(game_state_where_chosen.choice_list):
                    action_info["choice_name"] = game_state_where_chosen.choice_list[action.choice_index]
                else:
                    action_info["choice_name"] = None
            else:
                action_info["choice_name"] = None

        elif isinstance(action, StartGameAction):
            action_info["player_class"] = action.player_class.name
            action_info["ascension_level"] = action.ascension_level

        else:
            action_info["command"] = getattr(action, "command", str(action))

        return action_info

    def reset_baseline(self, game_state):
        """Initialize tracking values to current state (no delta on first step)"""
        self.last_gold = getattr(game_state, "gold", 0)
        self.last_hp = getattr(game_state, "current_hp", 0)
        self.last_max_hp = getattr(game_state, "max_hp", 0)
        self.last_deck_size = len(getattr(game_state, "deck", []) or [])
        self.last_relics = len(getattr(game_state, "relics", []) or [])

        self.last_in_combat = gs_in_combat(game_state)

        # Initialize combat stats if in combat
        monsters = gs_monsters(game_state)
        if self.last_in_combat and monsters:
            enemy_total_hp, alive_monsters = self.compute_enemy_stats(monsters)
            self.last_enemy_total_hp = enemy_total_hp
            self.last_alive_monsters = alive_monsters
        else:
            self.last_enemy_total_hp = None
            self.last_alive_monsters = 0
        
        self.pending_reward = 0.0


    def compute_reward_delta(self, game_state):
        current_gold = getattr(game_state, "gold", 0)
        current_hp = getattr(game_state, "current_hp", 0)

        deck = getattr(game_state, "deck", []) or []
        relics = getattr(game_state, "relics", []) or []
        current_deck_size = len(deck)
        current_relics = len(relics)

        in_combat = bool(getattr(game_state, "in_combat", False))

        enemy_total_hp = None
        alive_monsters = 0
        if in_combat:
            monsters = gs_monsters(game_state)
            if monsters:
                enemy_total_hp, alive_monsters = self.compute_enemy_stats(monsters)

        reward = 0.0

        # A) Progression / loot
        gold_delta = current_gold - self.last_gold
        reward += gold_delta * 0.01

        relic_delta = current_relics - self.last_relics
        reward += relic_delta * 1.0

        deck_delta = current_deck_size - self.last_deck_size
        if deck_delta < 0:
            reward += (-deck_delta) * 0.20

        # B) Combat shaping
        if in_combat:
            hp_loss = max(0, self.last_hp - current_hp)
            reward -= hp_loss * 0.05

            if enemy_total_hp is not None and self.last_enemy_total_hp is not None:
                dmg = max(0, self.last_enemy_total_hp - enemy_total_hp)
                reward += dmg * 0.02

            kills = max(0, self.last_alive_monsters - alive_monsters)
            reward += kills * 0.50

        return reward


    def update_reward_tracker(self, game_state):
        self.last_gold = getattr(game_state, "gold", 0)
        self.last_hp = getattr(game_state, "current_hp", 0)
        self.last_max_hp = getattr(game_state, "max_hp", 0)

        deck = getattr(game_state, "deck", []) or []
        relics = getattr(game_state, "relics", []) or []
        self.last_deck_size = len(deck)
        self.last_relics = len(relics)

        in_combat = bool(getattr(game_state, "in_combat", False))
        self.last_in_combat = in_combat

        # Update combat trackers
        if in_combat:
            monsters = gs_monsters(game_state)
            if monsters:
                enemy_total_hp, alive_monsters = self.compute_enemy_stats(monsters)
                self.last_enemy_total_hp = enemy_total_hp
                self.last_alive_monsters = alive_monsters
            else:
                # In combat but monsters not loaded yet
                self.last_enemy_total_hp = None
                self.last_alive_monsters = 0
        else:
            self.last_enemy_total_hp = None
            self.last_alive_monsters = 0

    def record_transition(self, episode_id, t, obs, action, reward, next_obs, done):
        """Record a single transition"""
        transition = {
            "episode_id": episode_id,
            "t": t,
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        }
        self.transitions.append(transition)

        # Periodic flush to prevent memory buildup
        if len(self.transitions) >= 2000:
            self.save_rollout()

    def save_rollout(self):
        """Save all transitions to jsonl file"""
        if self.transitions:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                for transition in self.transitions:
                    f.write(json.dumps(transition) + "\n")
            debug_log(f"SAVED: {len(self.transitions)} transitions")
        self.transitions = []

    def get_next_action(self, game_state):
            # Normalize all attributes at entry (use gs_* so it works for dict payloads too)
            screen = gs_screen_type(game_state)
            choice_list = gs_choice_list(game_state)
            proceed = gs_proceed_available(game_state)
            in_combat = gs_in_combat(game_state)
            play_avail = gs_play_available(game_state)
            end_avail = gs_end_available(game_state)
            potion_avail = gs_potion_available(game_state)
    
            hand = gs_hand(game_state)
            potions = gs_potions(game_state)
            monsters = gs_monsters(game_state)
            gold = gs_gold(game_state)
            floor = gs_floor(game_state)
    
            # --- epsilon-greedy exploration early (dataset diversity) ---
            ra = self.maybe_random_action(game_state, screen, choice_list, in_combat, play_avail, end_avail, hand, monsters)
            if ra is not None:
                return ra
    
            # ----------------------------
            # SHOP_SCREEN
            # ----------------------------
            if screen == "SHOP_SCREEN":
                now = time.time()
                if now - self.last_shop_action_time < 0.5:
                    return Action("state")
    
                if choice_list and gold:
                    screen_state = gs_screen_state(game_state)
                    items_with_prices = {}
    
                    if screen_state is not None:
                        for attr in ("cards", "potions", "relics"):
                            lst = getattr(screen_state, attr, None)
                            if lst is None and isinstance(screen_state, dict):
                                lst = screen_state.get(attr)
                            if not lst:
                                continue
                            for it in lst:
                                nm = getattr(it, "name", None) if not isinstance(it, dict) else it.get("name")
                                pr = getattr(it, "price", None) if not isinstance(it, dict) else it.get("price")
                                if nm is not None and pr is not None:
                                    items_with_prices[str(nm)] = int(pr)
    
                    for i, item in enumerate(choice_list):
                        if item != "purge" and item not in self.shop_bought_items:
                            price = items_with_prices.get(item)
                            if price is None:
                                debug_log(f"SHOP: no price match for '{item}'")
                                continue
                            if price > 0 and gold >= price:
                                self.shop_bought_items.add(item)
                                self.last_shop_action_time = now
                                return ChooseAction(choice_index=i)
    
                self.last_shop_action_time = now
                return Action("leave")
    
            # ----------------------------
            # SHOP_ROOM
            # ----------------------------
            if screen == "SHOP_ROOM":
                # Only clear shop state when first entering the room
                if self.last_screen != "SHOP_ROOM":
                    self.shop_bought_items.clear()
                    self.last_shop_action_time = 0.0

                lower = self._lower_choices(choice_list)

                # If we haven't opened the shop yet on this floor, try to open it using flexible matching.
                if self.shop_visited_floor != floor and choice_list:
                    open_idx = self._find_choice_index_contains(choice_list, needles=[
                        "shop", "merchant", "open", "enter", "talk"
                    ])
                    if open_idx is not None:
                        self.shop_visited_floor = floor
                        return ChooseAction(choice_index=open_idx)

                # If we can't open it (or already did), proceed/leave
                if proceed:
                    return Action("proceed")

                # Many connectors expose "leave" as a choice instead of proceed
                leave_idx = self._find_choice_index_contains(choice_list, needles=["leave", "back", "return"])
                if leave_idx is not None:
                    return ChooseAction(choice_index=leave_idx)

                # fallback
                if choice_list:
                    return ChooseAction(choice_index=0)

            # ----------------------------
            # CARD_REWARD (simple draft policy)
            # ----------------------------
            if screen == "CARD_REWARD" and choice_list:
                picks_good = {"inflame", "shrug it off", "anger", "uppercut", "offering", "battle trance", "headbutt"}
                picks_ok   = {"cleave", "thunderclap", "iron wave", "body slam"}
                lower = [c.lower() for c in choice_list]
    
                for i, c in enumerate(lower):
                    if c in picks_good:
                        return ChooseAction(choice_index=i)
                for i, c in enumerate(lower):
                    if c in picks_ok:
                        return ChooseAction(choice_index=i)
                if "skip" in lower:
                    return ChooseAction(choice_index=lower.index("skip"))
                return ChooseAction(choice_index=0)
    
            # ----------------------------
            # COMBAT_REWARD
            # ----------------------------
            if screen == "COMBAT_REWARD" and choice_list:
                lower = self._lower_choices(choice_list)

                # Prefer guaranteed-safe picks first
                if "gold" in lower:
                    return ChooseAction(choice_index=lower.index("gold"))
                if "card" in lower:
                    return ChooseAction(choice_index=lower.index("card"))

                # Potion: only take if we have an empty slot
                if "potion" in lower:
                    if self.has_empty_potion_slot(game_state):
                        return ChooseAction(choice_index=lower.index("potion"))
                    # otherwise skip potion if possible
                    if "skip" in lower:
                        return ChooseAction(choice_index=lower.index("skip"))
                    # if no explicit skip exists, avoid potion by picking something else
                    for i, c in enumerate(lower):
                        if c not in ("potion",):
                            return ChooseAction(choice_index=i)
                    return Action("proceed") if proceed else Action("state")

                # If there's a "skip" option in general, it's safe
                if "skip" in lower:
                    return ChooseAction(choice_index=lower.index("skip"))

                return ChooseAction(choice_index=0)

    
            # ----------------------------
            # MAP
            # ----------------------------
            if screen == "MAP" and choice_list:
                idx = self.pick_map_choice_index(game_state, choice_list)
                return ChooseAction(choice_index=idx)
    
            # Any other choice_list screens (default)
            if choice_list:
                return ChooseAction(choice_index=0)
    
            # ----------------------------
            # Potions (gated)
            # ----------------------------
            if in_combat and potion_avail and potions and self.should_use_potion_now(game_state, monsters):
                for potion in potions:
                    if not getattr(potion, "can_use", False):
                        continue
                    if getattr(potion, "requires_target", False):
                        target = self.pick_target(monsters, prefer_low_hp=True)
                        if target is not None:
                            self.last_potion_use_time = time.time()
                            return PotionAction(use=True, potion=potion, target_monster=target)
                    else:
                        self.last_potion_use_time = time.time()
                        return PotionAction(use=True, potion=potion)
    
            # ----------------------------
            # Combat: play best card (heuristic)
            # ----------------------------
            if in_combat and play_avail and hand:
                incoming = self.estimate_incoming_damage(monsters)
    
                best = None
                best_score = -1e9
    
                for card in hand:
                    if not getattr(card, "is_playable", False):
                        continue
                    s = self.score_card(card, game_state, incoming)
                    if s > best_score:
                        best_score = s
                        best = card
    
                if best is not None:
                    if getattr(best, "has_target", False):
                        target = self.pick_target(monsters, prefer_low_hp=True)
                        if target is not None:
                            return PlayCardAction(card=best, target_monster=target)
                    else:
                        return PlayCardAction(card=best)
    
                if end_avail:
                    return Action("end")
    
            # Wait for hand to load
            if in_combat and play_avail and not hand:
                return Action("state")
    
            # End turn if play not available
            if in_combat and end_avail and not play_avail:
                return Action("end")
    
            if proceed:
                return Action("proceed")
    
            return Action("state")


# Global state for rollout collection
collector = RolloutCollector()
episode_id = None
episode_counter = 0
t = 0
total_games = 0
last_obs = None
last_action = None
last_game_state = None
last_state_fingerprint = None
prev_screen = None


def on_state_change(game_state):
    global episode_id, episode_counter, t, last_obs, last_action, last_game_state, last_state_fingerprint, prev_screen

    try:
        obs = collector.serialize_state(game_state)

        # Episode initialization
        if episode_id is None:
            episode_counter += 1
            episode_id = f"{datetime.utcnow().isoformat()}_{episode_counter:04d}"
            t = 0
            collector.reset_baseline(game_state)
            collector.last_screen = None
            collector.pending_reward = 0.0
            prev_screen = gs_screen_type(game_state)

            last_obs = obs
            last_action = collector.get_next_action(game_state)
            last_game_state = game_state
            last_state_fingerprint = fingerprint(obs)
            debug_log(f"EPISODE START: {episode_id}")
            return last_action

        # Current screen + done detection
        screen = gs_screen_type(game_state)
        end_screens = {"GAME_OVER", "COMPLETE", "VICTORY", "CREDITS", "GAME_OVER_SCREEN"}
        done = screen in end_screens

        # Decide if we should log (avoid "state" spam and identical-state duplicates)
        fp = fingerprint(obs)
        should_log = fp != last_state_fingerprint

        # 1) Compute reward delta for THIS callback
        delta_reward = collector.compute_reward_delta(game_state)

        # 2) Add combat-end bonus ONLY when we ENTER the combat reward screen
        # (prevents spurious bonus from intermediate screen flips)
        # Note: adjust the literal string if your screen naming differs.
        if collector.last_in_combat and not gs_in_combat(game_state):
            if prev_screen != "COMBAT_REWARD" and screen == "COMBAT_REWARD":
                # survived the fight and reached reward screen
                if getattr(game_state, "current_hp", 0) > 0:
                    delta_reward += 1.0

        # 3) Accumulate reward even if we don't log this step
        collector.pending_reward += delta_reward

        # 4) Update trackers so deltas stay correct
        collector.update_reward_tracker(game_state)

        # 5) Only log when the state actually changed — using accumulated reward
        if should_log:
            logged_reward = collector.pending_reward
            collector.pending_reward = 0.0

            collector.record_transition(
                episode_id=episode_id,
                t=t,
                obs=last_obs,
                action=collector.serialize_action(last_action, last_game_state),
                reward=logged_reward,
                next_obs=obs,
                done=done
            )
            debug_log(f"T={t}: r={logged_reward:.3f}, done={done}")
            t += 1

        # Episode end
        if done:
            debug_log(f"EPISODE END: {episode_id}")
            if collector.pending_reward != 0.0:
                debug_log(f"Flushing pending_reward={collector.pending_reward:.3f} (terminal)")
                collector.pending_reward = 0.0
            collector.save_rollout()

            episode_id = None
            last_obs = None
            last_action = None
            last_game_state = None
            last_state_fingerprint = None
            prev_screen = None
            collector.pending_reward = 0.0
            return Action("state")

        # Prepare for next callback
        last_obs = obs
        last_action = collector.get_next_action(game_state)
        last_game_state = game_state
        last_state_fingerprint = fp
        collector.last_screen = screen
        prev_screen = screen

        return last_action

    except Exception as e:
        debug_log(f"ERROR: {e}")
        import traceback
        debug_log(traceback.format_exc())
        return Action("state")



def on_out_of_game():
    global episode_id, last_obs, last_action, last_game_state, last_state_fingerprint, prev_screen, total_games

    debug_log("OUT OF GAME")

    # If we were mid-episode, flush what we have.
    if episode_id is not None:
        if collector.pending_reward != 0.0:
            debug_log(f"Flushing pending_reward={collector.pending_reward:.3f} (terminal)")
            collector.pending_reward = 0.0
        collector.save_rollout()

    episode_id = None
    last_obs = None
    last_action = None
    last_game_state = None
    last_state_fingerprint = None
    prev_screen = None
    collector.pending_reward = 0.0

    if total_games == 0:
        total_games += 1
        return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)

    # After first full run ends, stop hard to avoid OUT_OF_GAME spam.
    debug_log("Completed 1 run; exiting collector.")
    os._exit(0)




def on_error(err: str):
    debug_log(f"ERROR: {err}")
    return Action("state")


def main():
    coord = Coordinator()
    coord.register_state_change_callback(on_state_change)
    coord.register_command_error_callback(on_error)
    coord.register_out_of_game_callback(on_out_of_game)
    coord.signal_ready()
    coord.run()


if __name__ == "__main__":
    main()
