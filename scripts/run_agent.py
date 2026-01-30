from datetime import datetime
import time

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, PlayCardAction, EventOptionAction, ChooseAction
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

def on_state_change(game_state):
    log(f"STATE RECEIVED - Screen: {getattr(game_state, 'screen_type', 'None')}")
    try:
        
        # game gets stuck in shop so auto leave for now [SCREEN_TYPE: SHOP]
        if game_state.screen_type.name == "SHOP_SCREEN":
            if game_state.cancel_available:
                return Action("leave")
        
        # when available, advances the screen 
        if game_state.proceed_available:
            return Action("proceed")
        
        # picks first available choice list
        if game_state.choice_list:
            log(f"PICKED ({game_state.choice_list[0]})")
            return ChooseAction(choice_index=0)
        

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

        # nothing to play -> end turn if possible
        if game_state.in_combat and game_state.end_available:
            log("END TURN")
            return Action("end")

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
