"""Shadow test for the Variant C WORKER-side reward terms. Drives SafeStsEnv
directly with random legal actions and re-derives the expected reward for every
step from an independent re-implementation of the spec, asserting equality.
(The deck potential is trainer-side, computed from the obs deck region -- see
discover_slots.py for its validation.)

Run on the VM: python test_reward_varc.py [attempt]   (~200 games, prints PASS)

The fork's native getObservation segfaults (startup heisenbug + state-dependent
crashes) can kill the test process mid-run; the attempt arg varies the seeds so
a retry wrapper can accumulate verified games across attempts. PROGRESS lines
report how many games were fully verified before any crash.
"""
import random
import sys
from sim_worker import SafeStsEnv
import slaythespire

attempt = int(sys.argv[1]) if len(sys.argv) > 1 else 0

_V = slaythespire.GameOutcome.PLAYER_VICTORY
_L = slaythespire.GameOutcome.PLAYER_LOSS

env = SafeStsEnv(ascension=0)
rng = random.Random(7 + attempt)
obs, info = env.reset(seed=4242 + 1000 * attempt)

games, checked, elite_b, boss_b, deaths, fight_wins = 0, 0, 0, 0, 0, 0
was_in_combat = bool(env.in_combat)
fight_room = ""
while games < 200:
    floor_b = env.last_floor
    hp_b = env.last_hp
    mask = info["action_mask"]
    legal = [i for i in range(len(mask)) if mask[i]]
    a = rng.choice(legal)
    obs, r, term, trunc, info = env.step(a)

    # --- independent expectation, straight from the spec ---
    exp = 0.0
    exp += (env.gc.floor_num - floor_b) * 10.0
    hp_a = env.bc.player.cur_hp if (env.in_combat and env.bc) else env.gc.cur_hp
    exp += (hp_a - hp_b) * 1.0
    if env.gc.outcome == _V:
        exp += 1000.0
    elif env.gc.outcome == _L:
        exp -= 100.0 / max(1, int(env.gc.act))
        deaths += 1
    if env.in_combat and not was_in_combat:
        fight_room = str(env.gc.cur_room)
    elif was_in_combat and not env.in_combat:
        if env.gc.outcome != _L:
            fight_wins += 1
            if "ELITE" in fight_room:
                exp += 30.0
                elite_b += 1
            elif "BOSS" in fight_room:
                exp += 60.0
                boss_b += 1
        fight_room = ""
    was_in_combat = bool(env.in_combat)
    assert abs(float(r) - exp) < 1e-3, (
        "game %d: step reward %.4f != expected %.4f" % (games, r, exp))
    checked += 1

    if term or trunc:
        games += 1
        if games % 10 == 0:
            print("PROGRESS games=%d steps=%d" % (games, checked), flush=True)
        obs, info = env.reset()
        was_in_combat = bool(env.in_combat)
        fight_room = ""

print("PASS: %d steps / %d games | fight wins %d | elite +30 paid %d | "
      "boss +60 paid %d | deaths %d" % (checked, games, fight_wins, elite_b, boss_b, deaths))
if elite_b == 0 and boss_b == 0:
    print("WARN: random play never won an elite/boss fight; +30/+60 constants "
          "not exercised here (same code path as the +0 case; CSV diagnostics "
          "cover it during the real run)")
