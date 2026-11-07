"""Dump the bindings' introspection API so we can decide the observation strategy:
guard their getObservation, or build our own obs from exposed combat state."""
import numpy as np
import slaythespire as sts


def pub(o):
    return [a for a in dir(o) if not a.startswith("_")]


rng = np.random.default_rng(1)
CH = sts.CharacterClass.IRONCLAD

print("MODULE:", pub(sts))
gc = sts.GameContext(CH, 1, 0)
print("\nGAMECONTEXT:", pub(gc))

# drive into a battle
bc = None
for _ in range(400):
    if gc.outcome != sts.GameOutcome.UNDECIDED:
        break
    acts = gc.get_possible_actions()
    if not acts:
        break
    acts[int(rng.integers(0, len(acts)))].execute(gc)
    if gc.screen_state == sts.ScreenState.BATTLE:
        bc = sts.BattleContext()
        bc.init(gc)
        break

if bc is None:
    print("\n!! never reached battle")
else:
    print("\nBATTLECONTEXT:", pub(bc))
    print("\nPLAYER:", pub(bc.player))
    # monster group
    mg = bc.monsters
    print("\nMONSTERGROUP:", pub(mg))
    # try common access patterns
    mons = None
    for attr in ("monsters", "arr", "group"):
        v = getattr(mg, attr, None)
        if v is not None:
            mons = v
            print("  monsters via .%s, len=%s" % (attr, getattr(v, "__len__", lambda: "?")()))
            break
    if mons is None:
        try:
            mons = [mg[i] for i in range(5)]
        except Exception as e:
            print("  monster index err:", e)
    if mons:
        print("\nMONSTER[0]:", pub(mons[0]))
    # a possible action object
    a = bc.get_possible_actions()[0]
    print("\nACTION:", pub(a))
    # hand cards
    for attr in ("cards", "hand", "cards_in_hand", "draw_pile", "discard_pile", "potions"):
        v = getattr(bc, attr, None)
        if v is not None:
            try:
                first = v[0] if len(v) else None
                print("\nbc.%s len=%d, elem attrs=%s" % (attr, len(v), pub(first) if first is not None else []))
            except Exception as e:
                print("\nbc.%s present but: %s" % (attr, str(e)[:50]))
    # player powers / energy / block
    p = bc.player
    for attr in ("energy", "block", "cur_hp", "max_hp", "powers", "creature_powers"):
        v = getattr(p, attr, "MISSING")
        print("  player.%s = %r" % (attr, v if attr != "powers" else type(v)))
