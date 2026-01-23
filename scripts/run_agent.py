from datetime import datetime
import time

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action

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
    log("STATE RECEIVED")
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
