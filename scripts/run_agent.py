from datetime import datetime
import time

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, PlayCardAction, EventOptionAction, ChooseAction, PotionAction, StartGameAction
from spirecomm.spire.character import PlayerClass
LOG_PATH = r"C:\AscensionAI\agent_debug.log"

def log(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
        f.flush()

log("AGENT STARTED")

class NoOpAction(Action):
    """An action that sends nothing (prevents command spam)."""
    def __init__(self):
        super().__init__("noop")

    def can_be_executed(self, coordinator):
        return True

    def execute(self, coordinator):
        # Do nothing
        return

LAST_STATE_SENT = 0.0
SHOP_VISITED_FLOOR = -1
SHOP_ITEM_BOUGHT = False

def send_state_throttled(min_interval_sec: float):
    global LAST_STATE_SENT
    now = time.time()
    if now - LAST_STATE_SENT >= min_interval_sec:
        LAST_STATE_SENT = now
        log("SENDING: state")
        return Action("state")
    return NoOpAction()

def on_out_of_game():
    # Out of game: only valid commands are usually [start, state]
    log("OUT_OF_GAME")
    return send_state_throttled(1.5)
    """return StartGameAction(player_class=PlayerClass.IRONCLAD, ascension_level=0)"""

def on_state_change(game_state):
    global SHOP_VISITED_FLOOR, SHOP_ITEM_BOUGHT
    log(f"STATE RECEIVED - Screen: {getattr(game_state, 'screen_type', 'None')}")
    try:
        # Handle SHOP_SCREEN - buy once then leave
        if game_state.screen_type.name == "SHOP_SCREEN":
            if not SHOP_ITEM_BOUGHT and game_state.choice_list:
                # Skip "purge", buy first purchasable item
                for i, item in enumerate(game_state.choice_list):
                    if item != "purge":
                        log(f"SHOP - Buying: {item}")
                        SHOP_ITEM_BOUGHT = True
                        return ChooseAction(choice_index=i)
            # Already bought or no items, leave shop
            log("SHOP - Leaving")
            return Action("leave")
        
        # Reset SHOP_ITEM_BOUGHT when leaving SHOP_SCREEN
        if game_state.screen_type.name == "SHOP_ROOM":
            SHOP_ITEM_BOUGHT = False
            # Only choose shop if we haven't visited it on this floor yet
            if SHOP_VISITED_FLOOR != game_state.floor and "shop" in game_state.choice_list:
                log("SHOP_ROOM - Opening shop")
                SHOP_VISITED_FLOOR = game_state.floor
                return ChooseAction(choice_index=game_state.choice_list.index("shop"))
            # Already visited shop this floor, proceed to leave
            elif game_state.proceed_available:
                log("SHOP_ROOM - Proceeding (already visited)")
                return Action("proceed")
        
        # Handle other choice lists (events, rewards)
        if game_state.choice_list:
            log(f"CHOOSING: {game_state.choice_list[0]}")
            return ChooseAction(choice_index=0)

        # Use potions whenever possible during combat
        if game_state.in_combat and game_state.potion_available and game_state.potions:
            for potion in game_state.potions:
                if potion.can_use:
                    if potion.requires_target:
                        # Use on first living monster
                        living = [m for m in game_state.monsters if not getattr(m, "is_gone", False)]
                        if living:
                            target = living[0]
                            log(f"USE POTION {potion.name} -> {target.name} (idx={target.monster_index})")
                            return PotionAction(use=True, potion=potion, target_monster=target)
                    else:
                        log(f"USE POTION {potion.name}")
                        return PotionAction(use=True, potion=potion)

        if game_state.in_combat and game_state.play_available and game_state.hand:
            # pick first playable card in hand
            for card in game_state.hand:
                if getattr(card, "is_playable", False):
                    if getattr(card, "has_target", False):
                        # pick first living monster
                        living = [m for m in game_state.monsters if not getattr(m, "is_gone", False)]
                        if living:
                            target = living[0]
                            log(f"PLAY {card.name} -> {target.name} (idx={target.monster_index})")
                            return PlayCardAction(card=card, target_monster=target)
                        else:
                            log(f"Wanted to target with {card.name}, but no monsters found")
                            break
                    else:
                        log(f"PLAY {card.name} (no target)")
                        return PlayCardAction(card=card)
            # No playable cards found, end turn if possible
            if game_state.end_available:
                log("END TURN")
                return Action("end")
        
        # If in combat but hand is empty/not loaded, wait for state
        elif game_state.in_combat and game_state.play_available and not game_state.hand:
            log("COMBAT - Waiting for hand to load")
            return send_state_throttled(0.3)
        
        # Only end turn if truly no cards and play is not available
        elif game_state.in_combat and game_state.end_available and not game_state.play_available:
            log("END TURN")
            return Action("end")
        
        # Proceed when available (and not in combat)
        if game_state.proceed_available:
            return Action("proceed")

    except Exception as e:
        log(f"PLAY_ACTION_ERROR: {e}")

    return send_state_throttled(1.0)

def on_error(err: str):
    log(f"ERROR: {err}")
    # After an error, just stop sending commands briefly
    return NoOpAction()

def main():
    coord = Coordinator()
    coord.register_state_change_callback(on_state_change)
    coord.register_command_error_callback(on_error)
    coord.register_out_of_game_callback(on_out_of_game)
    coord.signal_ready()
    coord.run()

if __name__ == "__main__":
    main()
