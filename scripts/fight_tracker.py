"""Elite and boss fight tracking shared by training/eval agents.

The important edge case is death during combat: CommunicationMod may still
report ``in_combat=True`` on the terminal state, so a tracker that only ends a
fight after combat disappears will count wins but silently miss losses.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Callable


_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
_LOG_DIR = os.path.join(_ROOT, "logs")
_FIGHT_CSV = os.path.join(_LOG_DIR, "fight_stats.csv")
_LEGACY_ELITE_CSV = os.path.join(_LOG_DIR, "elite_stats.csv")

_FIGHT_COLUMNS = [
    "timestamp", "source", "worker", "game", "floor", "act",
    "fight_type", "room_type", "monsters", "hp_before", "hp_after",
    "max_hp", "won", "ended_by",
]

_LEGACY_COLUMNS = [
    "timestamp", "worker", "game", "floor", "act", "room_type",
    "monsters", "hp_before", "hp_after", "max_hp", "won",
]


def _screen_name(gs: Any) -> str:
    st = getattr(gs, "screen_type", None)
    name = getattr(st, "name", st) if st is not None else "NONE"
    return str(name or "NONE")


def _room_fight_type(room_type: str) -> str | None:
    if "Boss" in room_type:
        return "boss"
    if "Elite" in room_type:
        return "elite"
    return None


def _living_monster_names(gs: Any) -> str:
    monsters = list(getattr(gs, "monsters", []) or [])
    names = [
        str(getattr(m, "name", "?"))
        for m in monsters
        if not getattr(m, "is_gone", False)
    ]
    return ";".join(names)


def _ensure_csv(path: str, columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def _append_csv(path: str, columns: list[str], row: dict) -> None:
    _ensure_csv(path, columns)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerow(row)


class FightTracker:
    """Tracks per-game elite/boss fights and writes per-fight CSV rows."""

    def __init__(
        self,
        source: str,
        worker: str = "",
        log: Callable[[str], None] | None = None,
    ):
        self.source = source
        self.worker = worker
        self.log = log
        self.reset_game()

    def reset_game(self) -> None:
        self.active_type: str | None = None
        self.room_type = ""
        self.floor = 0
        self.act = 0
        self.monsters = ""
        self.hp_before = 0
        self.elites_fought = 0
        self.elites_won = 0
        self.bosses_fought = 0
        self.bosses_won = 0

    def summary(self) -> dict[str, int]:
        return {
            "elites_fought": self.elites_fought,
            "elites_won": self.elites_won,
            "bosses_fought": self.bosses_fought,
            "bosses_won": self.bosses_won,
        }

    def observe(
        self,
        gs: Any,
        game: int,
        terminal: bool = False,
        victory: bool = False,
    ) -> None:
        in_combat = bool(getattr(gs, "in_combat", False))
        room_type = str(getattr(gs, "room_type", "") or "")
        fight_type = _room_fight_type(room_type)

        if self.active_type is None and in_combat and fight_type:
            self.active_type = fight_type
            self.room_type = room_type
            self.floor = int(getattr(gs, "floor", 0) or 0)
            self.act = int(getattr(gs, "act", 0) or 0)
            self.hp_before = int(getattr(gs, "current_hp", 0) or 0)
            self.monsters = _living_monster_names(gs)
            self._log(
                f"{fight_type.upper()} FIGHT started: {self.monsters} "
                f"floor={self.floor} hp={self.hp_before}"
            )

        if self.active_type is not None and (terminal or not in_combat):
            ended_by = "terminal" if terminal else "post_combat"
            self._finish(gs, game=game, terminal=terminal,
                         victory=victory, ended_by=ended_by)

    def finish_game(self, gs: Any, game: int, victory: bool = False) -> dict[str, int]:
        if self.active_type is not None:
            self._finish(gs, game=game, terminal=True,
                         victory=victory, ended_by="terminal")
        summary = self.summary()
        self.reset_game()
        return summary

    def _finish(
        self,
        gs: Any,
        game: int,
        terminal: bool,
        victory: bool,
        ended_by: str,
    ) -> None:
        hp_after = int(getattr(gs, "current_hp", 0) or 0)
        screen = _screen_name(gs)
        won = bool(victory) or (not terminal and hp_after > 0)

        if self.active_type == "elite":
            self.elites_fought += 1
            self.elites_won += int(won)
        elif self.active_type == "boss":
            self.bosses_fought += 1
            self.bosses_won += int(won)

        row = {
            "timestamp": datetime.now().isoformat(),
            "source": self.source,
            "worker": self.worker,
            "game": game,
            "floor": self.floor,
            "act": self.act,
            "fight_type": self.active_type,
            "room_type": self.room_type,
            "monsters": self.monsters,
            "hp_before": self.hp_before,
            "hp_after": hp_after,
            "max_hp": int(getattr(gs, "max_hp", 0) or 0),
            "won": int(won),
            "ended_by": ended_by,
        }

        try:
            _append_csv(_FIGHT_CSV, _FIGHT_COLUMNS, row)
            _append_csv(_LEGACY_ELITE_CSV, _LEGACY_COLUMNS, row)
        except Exception as e:
            self._log(f"fight csv append failed: {e}")

        self._log(
            f"{str(self.active_type).upper()} FIGHT ended: won={won} "
            f"hp={self.hp_before}->{hp_after} screen={screen} "
            f"ended_by={ended_by} ({self.monsters})"
        )
        self.active_type = None

    def _log(self, msg: str) -> None:
        if self.log is not None:
            try:
                self.log(msg)
            except Exception:
                pass
