"""Introspect the fork's bindings: what is exposed for constructing fights?
Pure introspection + one safe GameContext construction; no getNNInterface calls.
"""
import os

import gym_sts_lightspeed  # noqa: F401
import slaythespire as sts

print("== module surface ==")
print(sorted(n for n in dir(sts) if not n.startswith("_")))

CLASSES = [
    "GameContext", "BattleContext", "MonsterEncounter", "Player", "Monster",
    "Card", "CardInstance", "Deck", "Relic", "RelicInstance", "RelicId",
    "CardId", "Room", "GameOutcome", "Outcome", "CharacterClass",
    "NNInterface", "Action", "SearchAction",
]
for cls_name in CLASSES:
    cls = getattr(sts, cls_name, None)
    if cls is None:
        print("\n== %s: NOT PRESENT ==" % cls_name)
        continue
    print("\n== %s ==" % cls_name)
    names = [n for n in dir(cls) if not n.startswith("__") or n == "__init__"]
    for n in sorted(names):
        try:
            m = getattr(cls, n)
        except Exception as e:
            print("  %s: <getattr failed: %s>" % (n, e))
            continue
        doc = (getattr(m, "__doc__", "") or "").strip().splitlines()
        sig = doc[0] if doc else type(m).__name__
        print("  %s: %s" % (n, sig))

# Enum sizes (don't dump hundreds of values, just count + a sample)
for enum_name in ["MonsterEncounter", "CardId", "RelicId"]:
    e = getattr(sts, enum_name, None)
    if e is None:
        continue
    try:
        members = e.__members__
        print("\n== %s: %d members, sample: %s ==" % (
            enum_name, len(members), list(members)[:12]))
    except Exception as ex:
        print("\n== %s: no __members__ (%s) ==" % (enum_name, ex))

print("\n== gym_sts_lightspeed package location ==")
print(os.path.dirname(gym_sts_lightspeed.__file__))

print("\n== GameContext construction smoke ==")
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
print("constructed:", gc)
attrs = {}
for n in ["deck", "relics", "cur_hp", "max_hp", "gold", "floor_num", "act",
          "boss", "cur_room", "outcome", "screen_state"]:
    try:
        v = getattr(gc, n)
        attrs[n] = (type(v).__name__, str(v)[:60])
    except Exception as e:
        attrs[n] = ("ERR", str(e)[:60])
for k, v in attrs.items():
    print("  gc.%s -> %s %s" % (k, v[0], v[1]))

# Is anything settable?
print("\n== settability checks ==")
for n, val in [("cur_hp", 50), ("max_hp", 99), ("gold", 123),
               ("floor_num", 7), ("act", 1)]:
    try:
        setattr(gc, n, val)
        print("  gc.%s = %r OK (now %r)" % (n, val, getattr(gc, n)))
    except Exception as e:
        print("  gc.%s SET FAILED: %s" % (n, str(e)[:80]))
