"""Feasibility probe for train-time planning: can we clone a mid-fight state
to do lookahead (try action A, roll out, measure outcome), and how fast?
Tests: (1) Python copy.deepcopy of BattleContext+GameContext; (2) the fork's
own copy constructor / copy methods if any; (3) timing of a 1-ply lookahead
via re-init from seed (fallback). No getNNInterface early."""
import copy
import time
import slaythespire as sts

print("== BattleContext / GameContext introspection for copy support ==")
for cls_name in ("BattleContext", "GameContext"):
    cls = getattr(sts, cls_name)
    methods = [n for n in dir(cls) if not n.startswith("__")]
    copyish = [n for n in methods if "copy" in n.lower() or "clone" in n.lower()]
    print("%s copy-ish methods: %s" % (cls_name, copyish))
    print("  has __copy__: %s  __deepcopy__: %s" % (
        hasattr(cls, "__copy__"), hasattr(cls, "__deepcopy__")))

# set up a mid-fight state
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc, sts.MonsterEncounter.GREMLIN_NOB)
import random
rng = random.Random(0)
for _ in range(3):  # advance a few actions into the fight
    acts = bc.get_possible_actions()
    if not acts or bc.outcome != sts.BattleOutcome.UNDECIDED:
        break
    rng.choice(acts).execute(bc)
print("\nmid-fight: turn=%d player_hp=%d enemy0_hp=%d" % (
    bc.turn, bc.player.cur_hp, bc.monsters.monsters[0].cur_hp))

print("\n== (1) copy.deepcopy(bc) ==")
try:
    t = time.time()
    bc2 = copy.deepcopy(bc)
    dt = (time.time() - t) * 1e6
    # mutate the copy, check the original is untouched
    a = bc2.get_possible_actions()[0]
    a.execute(bc2)
    print("deepcopy OK in %.0f us | orig turn=%d copy turn=%d (independent=%s)"
          % (dt, bc.turn, bc2.turn, bc.player.cur_hp != bc2.player.cur_hp
             or bc.turn != bc2.turn))
except Exception as e:
    print("deepcopy FAILED:", repr(e))

print("\n== (2) copy constructor BattleContext(bc) ==")
try:
    bc3 = sts.BattleContext(bc)
    print("copy-ctor OK:", bc3)
except Exception as e:
    print("copy-ctor not available:", repr(e)[:90])

print("\n== (3) fallback timing: re-init a fight from scratch (per candidate) ==")
t = time.time()
M = 200
for _ in range(M):
    g = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    b = sts.BattleContext()
    b.init(g, sts.MonsterEncounter.GREMLIN_NOB)
dt = (time.time() - t) / M * 1e6
print("re-init from scratch: %.0f us/fight (%.0f fights/s)" % (dt, 1e6 / dt))

print("\n== (4) full random rollout-to-end timing (for MC action-value) ==")
t = time.time()
R = 100
for _ in range(R):
    g = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    b = sts.BattleContext()
    b.init(g, sts.MonsterEncounter.GREMLIN_NOB)
    steps = 0
    while b.outcome == sts.BattleOutcome.UNDECIDED and steps < 200:
        acts = b.get_possible_actions()
        if not acts:
            break
        rng.choice(acts).execute(b)
        steps += 1
dt = (time.time() - t) / R * 1e3
print("full random fight rollout: %.2f ms/fight (%.0f rollouts/s)" % (dt, 1e3 / dt))
