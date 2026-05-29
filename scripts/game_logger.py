"""
game_logger.py — Passive CommunicationMod state recorder.

Plugs into CommunicationMod like any other agent, but performs no gameplay
actions. On every state update from the mod, it dumps a structured snapshot
of the game state to a JSONL file under logs/. You play the run manually
via mouse/keyboard; this script just observes and records what the mod sees.

Useful for auditing the action space: play a run touching every room type
(shop buy+leave, event branches, card reward skip, rest upgrade vs heal,
boss node, chest, elite, unknown, etc.) and then inspect the trace to see
exactly which screen_type / choice_list / *_available flag combinations
the mod exposes for each situation.
"""

from __future__ import annotations

import sys
import os

# ---- stdout belongs to Communication Mod ----
_real_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

import spirecomm.communication.coordinator as _coord_module


def _patched_write_stdout(output_queue):
    while True:
        output = output_queue.get()
        _real_stdout.write(output + "\n")
        _real_stdout.flush()


_coord_module.write_stdout = _patched_write_stdout

def _patched_read_stdin(input_queue):
    while True:
        stdin_input = ""
        while True:
            ch = sys.stdin.read(1)
            if ch == '':
                os._exit(1)
            if ch == '\n':
                break
            stdin_input += ch
        input_queue.put(stdin_input)
_coord_module.read_stdin = _patched_read_stdin
# ---- End stdout fix ----

_scripts = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_scripts)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

import json
import time
import argparse
from datetime import datetime
from typing import Any

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action


LOG_DIR = os.path.join(_root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
TRACE_PATH = os.path.join(
    LOG_DIR, f"game_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)
DEBUG_LOG = os.path.join(_root, "logs", "game_logger_debug.log")
VERBOSE = os.environ.get("ASCENSION_VERBOSE", "0") == "1"

POLL_THROTTLE_SEC = 0.1  # cap the passive poll loop at ~10 Hz


def dlog(msg: str) -> None:
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {msg}\n")
    except Exception:
        pass


dlog(f"=== LOGGER STARTING -> {TRACE_PATH} ===")


def _shallow(obj: Any, depth: int = 2) -> Any:
    """Serialize SpireComm object graphs to JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_shallow(x, depth - 1) if depth > 0 else repr(x) for x in obj]
    if isinstance(obj, dict):
        return {
            str(k): (_shallow(v, depth - 1) if depth > 0 else repr(v))
            for k, v in obj.items()
        }
    if hasattr(obj, "__dict__") and depth > 0:
        out = {"__type__": type(obj).__name__}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            try:
                out[k] = _shallow(v, depth - 1)
            except Exception:
                out[k] = repr(v)
        return out
    name = getattr(obj, "name", None)
    if name is not None:
        return f"<{type(obj).__name__}.{name}>"
    return repr(obj)


def _enum_name(obj: Any) -> Any:
    if obj is None:
        return None
    name = getattr(obj, "name", None)
    return name if name is not None else str(obj)


def snapshot(gs: Any) -> dict:
    if gs is None:
        return {"ts": datetime.now().isoformat(), "in_game": False}
    scr = getattr(gs, "screen", None)
    return {
        "ts": datetime.now().isoformat(),
        "in_game": True,
        "floor": getattr(gs, "floor", None),
        "act": getattr(gs, "act", None),
        "screen_type": _enum_name(getattr(gs, "screen_type", None)),
        "room_phase": _enum_name(getattr(gs, "room_phase", None)),
        "room_type": getattr(gs, "room_type", None),
        "current_hp": getattr(gs, "current_hp", None),
        "max_hp": getattr(gs, "max_hp", None),
        "gold": getattr(gs, "gold", None),
        "in_combat": getattr(gs, "in_combat", False),
        "play_available": getattr(gs, "play_available", False),
        "end_available": getattr(gs, "end_available", False),
        "potion_available": getattr(gs, "potion_available", False),
        "proceed_available": getattr(gs, "proceed_available", False),
        "cancel_available": getattr(gs, "cancel_available", False),
        "choice_available": getattr(gs, "choice_available", False),
        "choice_list": [str(c) for c in (getattr(gs, "choice_list", []) or [])],
        "deck_size": len(getattr(gs, "deck", []) or []),
        "relics": [
            getattr(r, "name", str(r)) for r in (getattr(gs, "relics", []) or [])
        ],
        "hand": _shallow(getattr(gs, "hand", []), depth=1),
        "monsters": _shallow(getattr(gs, "monsters", []), depth=2),
        "potions": _shallow(getattr(gs, "potions", []), depth=1),
        "screen": _shallow(scr, depth=2),
    }


class StateLogger:
    def __init__(self) -> None:
        self.last_sig: Any = None
        self.count = 0
        self.errors = 0

    @staticmethod
    def _sig(rec: dict) -> tuple:
        return (
            rec.get("in_game", False),
            rec.get("screen_type"),
            rec.get("room_phase"),
            rec.get("floor"),
            rec.get("current_hp"),
            rec.get("max_hp"),
            rec.get("gold"),
            rec.get("deck_size"),
            rec.get("in_combat"),
            rec.get("play_available"),
            rec.get("end_available"),
            rec.get("potion_available"),
            rec.get("proceed_available"),
            rec.get("cancel_available"),
            rec.get("choice_available"),
            tuple(rec.get("choice_list", []) or []),
            len(rec.get("hand", []) or []) if isinstance(rec.get("hand"), list) else 0,
            len(rec.get("monsters", []) or []) if isinstance(rec.get("monsters"), list) else 0,
        )

    def _write(self, record: dict) -> None:
        try:
            with open(TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            self.errors += 1
            dlog(f"write failed: {e}")

    def on_state_change(self, gs: Any) -> Action:
        try:
            rec = snapshot(gs)
            sig = self._sig(rec)
            if VERBOSE or sig != self.last_sig:
                self.last_sig = sig
                self.count += 1
                self._write(rec)
                self._log_state(rec, gs)
        except Exception as e:
            dlog(f"on_state_change error: {e}")
        time.sleep(POLL_THROTTLE_SEC)
        return Action("state")

    def _log_state(self, rec: dict, gs: Any) -> None:
        screen = rec.get("screen_type", "?")
        choices = rec.get("choice_list") or []
        flags = []
        for flag in ("proceed", "cancel", "choice", "play", "end", "potion"):
            if rec.get(f"{flag}_available"):
                flags.append(flag)
        flag_str = ",".join(flags) if flags else "none"

        dlog(
            f"#{self.count} screen={screen} floor={rec.get('floor')} "
            f"hp={rec.get('current_hp')}/{rec.get('max_hp')} "
            f"gold={rec.get('gold')} flags=[{flag_str}] "
            f"choices({len(choices)})={choices[:8]}"
        )

        scr = getattr(gs, "screen", None)
        if screen in ("SHOP_ROOM", "SHOP_SCREEN") and scr is not None:
            cards = getattr(scr, "cards", None) or []
            relics = getattr(scr, "relics", None) or []
            potions = getattr(scr, "potions", None) or []
            purge = getattr(scr, "purge_available", False)
            purge_cost = getattr(scr, "purge_cost", None)
            dlog(f"  SHOP cards={[(str(getattr(c,'name',c)), getattr(c,'price','?')) for c in cards]}")
            dlog(f"  SHOP relics={[(str(getattr(r,'name',r)), getattr(r,'price','?')) for r in relics]}")
            if potions:
                dlog(f"  SHOP potions={[(str(getattr(p,'name',p)), getattr(p,'price','?')) for p in potions]}")
            dlog(f"  SHOP purge_available={purge} purge_cost={purge_cost}")

        if screen in ("COMBAT_REWARD",) and scr is not None:
            rewards = getattr(scr, "rewards", None) or []
            dlog(f"  REWARDS={[_shallow(r, depth=1) for r in rewards]}")

        if (VERBOSE and scr is not None) or screen in (
            "CARD_REWARD", "BOSS_REWARD", "REST", "EVENT", "HAND_SELECT", "GRID"
        ):
            dlog(f"  DETAIL screen_obj={_shallow(scr, depth=2)}")

    def on_out_of_game(self) -> Action:
        rec = {
            "ts": datetime.now().isoformat(),
            "in_game": False,
            "screen_type": "MAIN_MENU",
        }
        sig = self._sig(rec)
        if sig != self.last_sig:
            self.last_sig = sig
            self._write(rec)
            dlog("out of game (main menu)")
        time.sleep(POLL_THROTTLE_SEC * 5)
        return Action("state")

    def on_error(self, err: str) -> Action:
        self.errors += 1
        self._write(
            {"ts": datetime.now().isoformat(), "error": str(err), "in_game": None}
        )
        dlog(f"mod error: {err}")
        time.sleep(POLL_THROTTLE_SEC)
        return Action("state")


def main() -> None:
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Write every observed state, including repeats")
    args = parser.parse_args()
    VERBOSE = VERBOSE or args.verbose
    dlog(f"Config: verbose={VERBOSE} trace={TRACE_PATH}")

    logger = StateLogger()
    coord = Coordinator()
    coord.register_state_change_callback(logger.on_state_change)
    coord.register_out_of_game_callback(logger.on_out_of_game)
    coord.register_command_error_callback(logger.on_error)
    coord.signal_ready()
    dlog("Coordinator signaled ready; entering run loop")
    try:
        coord.run()
    except Exception as e:
        dlog(f"FATAL in run loop: {e}")
        raise


if __name__ == "__main__":
    main()
