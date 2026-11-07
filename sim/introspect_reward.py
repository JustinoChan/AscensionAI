"""What state do the bindings expose for reward computation? Run on VM."""
import slaythespire as s

for name in ("BattleContext", "Player", "Monster", "GameContext", "Deck", "Card"):
    cls = getattr(s, name, None)
    if cls is None:
        print(f"{name}: NOT EXPOSED")
        continue
    attrs = [a for a in dir(cls) if not a.startswith("_")]
    print(f"{name}: {attrs}")
    print()

# Live poke: start a game, enter first combat, see what we can actually read.
gc = s.GameContext(s.CharacterClass.IRONCLAD, 12345, 0)
steps = 0
while gc.outcome == s.GameOutcome.UNDECIDED and steps < 200:
    if gc.screen_state == s.ScreenState.BATTLE:
        bc = s.BattleContext()
        bc.init(gc)
        print("IN BATTLE — live reads:")
        for attr in [a for a in dir(bc) if not a.startswith("_")]:
            try:
                v = getattr(bc, attr)
                if not callable(v):
                    print(f"  bc.{attr} = {v}")
            except Exception as e:
                print(f"  bc.{attr} -> ERR {e}")
        p = bc.player
        for attr in [a for a in dir(p) if not a.startswith("_")]:
            try:
                v = getattr(p, attr)
                if not callable(v):
                    print(f"  bc.player.{attr} = {v}")
            except Exception as e:
                print(f"  bc.player.{attr} -> ERR {e}")
        break
    acts = gc.get_possible_actions()
    if not acts:
        break
    acts[0].execute(gc)
    steps += 1
else:
    print("never reached battle in", steps, "steps")

print()
print("GC live attrs:")
for attr in [a for a in dir(gc) if not a.startswith("_")]:
    try:
        v = getattr(gc, attr)
        if not callable(v):
            print(f"  gc.{attr} = {v}")
    except Exception as e:
        print(f"  gc.{attr} -> ERR {e}")
