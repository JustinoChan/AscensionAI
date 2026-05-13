from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "spirecomm"))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_handler import auto_handle_screen, event_choice_targets
from sts_gym_env import (
    _CHOOSE_START,
    _LEAVE,
    _PROCEED,
    compute_action_mask,
    flat_action_to_spire_action,
)


def _option(label: str, *, disabled=False, choice_index=0):
    return SimpleNamespace(label=label, text=label, disabled=disabled,
                           choice_index=choice_index)


def _event_state(choice_list, options, *, hp=50, commands=None):
    if commands is None:
        commands = ["choose", "proceed", "leave"]
    return SimpleNamespace(
        screen_type=SimpleNamespace(name="EVENT"),
        screen=SimpleNamespace(
            event_id="Cursed Tome",
            event_name="Cursed Tome",
            body_text="Continue reading.",
            options=list(options),
        ),
        choice_list=list(choice_list),
        current_hp=hp,
        max_hp=80,
        in_combat=False,
        hand=[],
        monsters=[],
        potions=[],
        play_available=False,
        end_available=False,
        potion_available=False,
        proceed_available="proceed" in commands,
        cancel_available="leave" in commands,
        available_commands=list(commands),
        relics=[],
        are_potions_full=lambda: False,
    )


class EventChoiceHandlingTests(unittest.TestCase):
    def test_single_option_event_is_auto_handled_and_cannot_proceed(self):
        gs = _event_state(["Continue"], [_option("Continue")])

        action = auto_handle_screen(gs, "EVENT", heuristic_all=False)
        self.assertEqual("choose", action.command)
        self.assertEqual(0, action.choice_index)

        mask = compute_action_mask(gs)
        self.assertTrue(mask[_CHOOSE_START + 0])
        self.assertFalse(mask[_PROCEED])
        self.assertFalse(mask[_LEAVE])

    def test_compact_choice_list_maps_to_enabled_option_choice_index(self):
        gs = _event_state(
            ["Offer relic"],
            [
                _option("Locked", disabled=True, choice_index=0),
                _option("Offer relic", choice_index=2),
            ],
        )

        self.assertEqual({0: 2}, event_choice_targets(gs))
        action = flat_action_to_spire_action(_CHOOSE_START + 0, gs)
        self.assertEqual("choose", action.command)
        self.assertEqual(2, action.choice_index)

    def test_two_option_event_keeps_rl_decision_but_blocks_generic_exit(self):
        gs = _event_state(
            ["Leave", "Take the book"],
            [_option("Leave", choice_index=0), _option("Take the book", choice_index=1)],
        )

        self.assertIsNone(auto_handle_screen(gs, "EVENT", heuristic_all=False))

        mask = compute_action_mask(gs)
        self.assertTrue(mask[_CHOOSE_START + 0])
        self.assertTrue(mask[_CHOOSE_START + 1])
        self.assertFalse(mask[_PROCEED])
        self.assertFalse(mask[_LEAVE])

        action = flat_action_to_spire_action(_LEAVE, gs)
        self.assertEqual("choose", action.command)
        self.assertEqual(0, action.choice_index)


if __name__ == "__main__":
    unittest.main()
