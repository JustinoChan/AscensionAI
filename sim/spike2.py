"""Step-0 spike v2: isolate what works in sts_lightspeed via the fork's bindings.

Drives GameContext directly with a random agent. Goals:
  1. Is the SIMULATOR driveable to game-end without the (buggy) full-run obs?
  2. Can we ENTER combat, and does the COMBAT observation getObservation(gc,bc) work?
  3. Throughput: full-run sim steps/s and combat steps/s.
We wrap getObservation in try/except so a non-combat obs bug can't abort the run.
"""
import time
import numpy as np
import slaythespire as sts

CH = sts.CharacterClass.IRONCLAD
rng = np.random.default_rng(0)
nn = sts.getNNInterface()


def try_obs(gc, bc):
    try:
        o = np.array(nn.getObservation(gc, bc), dtype=np.int32)
        return o.shape[0], None
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__ + ": " + str(e)[:60]


games = 0
gc_steps = 0
combat_steps = 0
battles_entered = 0
combat_obs_ok = 0
combat_obs_fail = 0
noncombat_obs_ok = 0
noncombat_obs_fail = 0
floors = []
outcomes = {"WIN": 0, "LOSS": 0, "STUCK": 0}
first_combat_obs_len = None

BUDGET_S = 20.0
t0 = time.time()
while time.time() - t0 < BUDGET_S:
    gc = sts.GameContext(CH, int(rng.integers(0, 2**31)), 0)
    in_combat = False
    bc = None
    guard = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and guard < 5000:
        guard += 1
        if in_combat:
            acts = bc.get_possible_actions()
            if not acts:
                break
            a = acts[int(rng.integers(0, len(acts)))]
            a.execute(bc)
            combat_steps += 1
            n, err = try_obs(gc, bc)
            if n is not None:
                combat_obs_ok += 1
                if first_combat_obs_len is None:
                    first_combat_obs_len = n
            else:
                combat_obs_fail += 1
            if bc.outcome != sts.BattleOutcome.UNDECIDED:
                bc.exit_battle(gc)
                in_combat = False
                bc = None
        else:
            acts = gc.get_possible_actions()
            if not acts:
                break
            a = acts[int(rng.integers(0, len(acts)))]
            a.execute(gc)
            gc_steps += 1
            if gc.screen_state == sts.ScreenState.BATTLE:
                bc = sts.BattleContext()
                bc.init(gc)
                in_combat = True
                battles_entered += 1
            else:
                n, err = try_obs(gc, None)
                if n is not None:
                    noncombat_obs_ok += 1
                else:
                    noncombat_obs_fail += 1
    games += 1
    floors.append(gc.floor_num)
    if gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
        outcomes["WIN"] += 1
    elif gc.outcome == sts.GameOutcome.PLAYER_LOSS:
        outcomes["LOSS"] += 1
    else:
        outcomes["STUCK"] += 1

dt = time.time() - t0
print("=== SPIKE v2 (%.1fs random agent) ===" % dt)
print("games completed     =", games, "-> %.1f games/s" % (games / dt))
print("gc (run) steps       =", gc_steps, "-> %d steps/s" % round(gc_steps / dt))
print("combat steps         =", combat_steps, "-> %d steps/s" % round(combat_steps / dt))
print("battles entered      =", battles_entered)
print("combat obs ok/fail   =", combat_obs_ok, "/", combat_obs_fail,
      "(first combat obs len = %s)" % first_combat_obs_len)
print("noncombat obs ok/fail=", noncombat_obs_ok, "/", noncombat_obs_fail)
fl = [f for f in floors if isinstance(f, (int, float))]
if fl:
    print("floor mean/max       =", round(sum(fl) / len(fl), 2), "/", max(fl))
print("outcomes             =", outcomes)
