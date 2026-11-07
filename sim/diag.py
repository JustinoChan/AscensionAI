"""Diagnose the getObservation map::at failures: which combat states break it?

First prints the introspection API (dir) of the key classes so we know what we
can read, then drives random games and, on each obs failure, dumps everything
readable about the combat state to categorize the missing key.
"""
import numpy as np
import slaythespire as sts

nn = sts.getNNInterface()
CH = sts.CharacterClass.IRONCLAD
rng = np.random.default_rng(0)


def pub(o):
    return [a for a in dir(o) if not a.startswith("_")]


# --- 1. Introspection API ---
print("=== module ===");        print(pub(sts)[:60])
gc0 = sts.GameContext(CH, 1, 0)
print("=== GameContext ===");   print(pub(gc0))
# get into a battle to introspect BattleContext
bc0 = None
guard = 0
while gc0.outcome == sts.GameOutcome.UNDECIDED and guard < 300:
    guard += 1
    acts = gc0.get_possible_actions()
    if not acts:
        break
    acts[int(rng.integers(0, len(acts)))].execute(gc0)
    if gc0.screen_state == sts.ScreenState.BATTLE:
        bc0 = sts.BattleContext(); bc0.init(gc0); break
if bc0 is not None:
    print("=== BattleContext ==="); print(pub(bc0))
    print("=== bc.player ===");     print(pub(bc0.player))
    try:
        ms = bc0.monsters
        print("=== monsters type ===", type(ms))
        m0 = ms[0] if hasattr(ms, "__getitem__") else None
        if m0 is not None:
            print("=== monster[0] ==="); print(pub(m0))
    except Exception as e:
        print("monsters introspect err:", e)


def dump_state(gc, bc):
    d = {}
    try: d["floor"] = gc.floor_num
    except Exception: pass
    try: d["encounter"] = str(gc.encounter)
    except Exception: pass
    try: d["deck"] = sorted(str(c.id) for c in gc.deck)
    except Exception as e: d["deck_err"] = str(e)[:40]
    try: d["relics"] = sorted(str(r.id) for r in gc.relics)
    except Exception as e: d["relics_err"] = str(e)[:40]
    if bc is not None:
        try: d["bc_monsters"] = [str(getattr(m, "id", "?")) for m in bc.monsters]
        except Exception as e: d["mon_err"] = str(e)[:40]
        for attr in ("cards_in_hand", "hand", "draw_pile", "discard_pile", "potions", "player"):
            try:
                v = getattr(bc, attr, None)
                if v is not None and attr != "player":
                    d[attr] = [str(getattr(c, "id", c)) for c in v][:12]
            except Exception as e:
                d[attr + "_err"] = str(e)[:40]
    return d


# --- 2. Hunt failures ---
print("\n=== HUNTING OBS FAILURES ===")
fails = 0
total_combat_obs = 0
seen = []
for g in range(4000):
    gc = sts.GameContext(CH, int(rng.integers(0, 2**31)), 0)
    in_combat = False; bc = None; guard = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and guard < 4000:
        guard += 1
        if in_combat:
            acts = bc.get_possible_actions()
            if not acts: break
            acts[int(rng.integers(0, len(acts)))].execute(bc)
            total_combat_obs += 1
            try:
                nn.getObservation(gc, bc)
            except Exception as e:
                fails += 1
                if len(seen) < 8:
                    seen.append((type(e).__name__, dump_state(gc, bc)))
            if bc.outcome != sts.BattleOutcome.UNDECIDED:
                bc.exit_battle(gc); in_combat = False; bc = None
        else:
            acts = gc.get_possible_actions()
            if not acts: break
            acts[int(rng.integers(0, len(acts)))].execute(gc)
            if gc.screen_state == sts.ScreenState.BATTLE:
                bc = sts.BattleContext(); bc.init(gc); in_combat = True
    if fails >= 8 and g > 50:
        break

print("combat obs failures: %d / %d" % (fails, total_combat_obs))
for i, (etype, st) in enumerate(seen):
    print("\n--- failure %d (%s) ---" % (i + 1, etype))
    for k, v in st.items():
        print("   %-14s %s" % (k, v))
