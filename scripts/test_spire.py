import sys
sys.path.append(r"D:\StS AI Proj\AscensionAI\spirecomm\spirecomm")

from communication.coordinator import Coordinator
from communication.action import Action, StartGameAction

# Initialize Coordinator
coordinator = Coordinator()

# Signal ready to Communication Mod
coordinator.signal_ready()

# Optional: define dummy callbacks
def state_callback(game_state):
    print("Received game state:", game_state)
    return Action("pass")  # dummy action

def out_of_game_callback():
    print("Out of game")
    return Action("start_game")  # dummy start action

coordinator.register_state_change_callback(state_callback)
coordinator.register_out_of_game_callback(out_of_game_callback)

# Add an initial dummy action (optional)
coordinator.add_action_to_queue(Action("pass"))

# Run coordinator loop
coordinator.run()  # this will loop forever, updating state and executing actions
