"""Monkey-patches for spirecomm that keep the external library untouched.

Call apply_all(real_stdout) early in any script that uses Coordinator,
BEFORE constructing a Coordinator instance (the stdin/stdout reader
threads are started in __init__).
"""
import sys
import os
import json
import traceback

import spirecomm.communication.coordinator as _coord_module
from spirecomm.spire.game import Game

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_bug_log = os.path.join(_root, "logs", "bug_debug.log")


def _make_patched_write_stdout(real_stdout):
    def _patched_write_stdout(output_queue):
        while True:
            output = output_queue.get()
            real_stdout.write(output + "\n")
            real_stdout.flush()
    return _patched_write_stdout


def _patched_read_stdin(input_queue):
    """Detect stdin EOF (STS died) and exit instead of spinning forever."""
    while True:
        stdin_input = ""
        while True:
            ch = sys.stdin.read(1)
            if ch == "":
                os._exit(1)
            if ch == "\n":
                break
            stdin_input += ch
        input_queue.put(stdin_input)


def _patched_receive_game_state_update(self, block=False, perform_callbacks=True):
    """Wraps Game.from_json in try/except so a bad game state doesn't crash
    the entire worker.  Logs the traceback to bug_debug.log and requests
    a fresh state from CommunicationMod."""
    message = self.get_next_raw_message(block)
    if message is not None:
        communication_state = json.loads(message)
        self.last_error = communication_state.get("error", None)
        self.game_is_ready = communication_state.get("ready_for_command")
        if self.last_error is None:
            self.in_game = communication_state.get("in_game")
            if self.in_game:
                try:
                    self.last_game_state = Game.from_json(
                        communication_state.get("game_state"),
                        communication_state.get("available_commands"),
                    )
                except Exception:
                    try:
                        with open(_bug_log, "a") as f:
                            f.write(f"Game.from_json CRASH:\n{traceback.format_exc()}\n")
                            gs_json = communication_state.get("game_state", {})
                            f.write(f"screen_type={gs_json.get('screen_type')}\n")
                            f.write(f"screen_state keys="
                                    f"{list((gs_json.get('screen_state') or {}).keys())}\n\n")
                    except Exception:
                        pass
                    self.send_message("state")
                    return True
        if perform_callbacks:
            if self.last_error is not None:
                self.action_queue.clear()
                new_action = self.error_callback(self.last_error)
                self.add_action_to_queue(new_action)
            elif self.in_game:
                if len(self.action_queue) == 0 and perform_callbacks:
                    new_action = self.state_change_callback(self.last_game_state)
                    self.add_action_to_queue(new_action)
            elif self.stop_after_run:
                self.clear_actions()
            else:
                new_action = self.out_of_game_callback()
                self.add_action_to_queue(new_action)
        return True
    return False


def apply_all(real_stdout=None):
    """Apply all spirecomm monkey-patches. Call once at module init."""
    if real_stdout is not None:
        _coord_module.write_stdout = _make_patched_write_stdout(real_stdout)
    _coord_module.read_stdin = _patched_read_stdin
    _coord_module.Coordinator.receive_game_state_update = _patched_receive_game_state_update
