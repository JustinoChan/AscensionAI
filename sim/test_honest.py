"""Tests for GameContext/BattleContext copy() + determinize() (HONEST search).

(a) parity : an exact copy reproduces the replay-from-seed rollout bit-for-bit
             (combat 1-ply rollout values for every candidate, and a long
             full-run continuation from a copied map-screen state).
(b) hidden : determinize(s) leaves the visible observation unchanged and
             different s give different futures (combat rollouts, encounter
             sequence of a run continuation).
(c) is run separately (toggles-off 5-seed byte-identity vs sim_logs/vdraft3_A).
"""
import os
import sys
os.environ.setdefault("HONEST", "1")
import numpy as np
import eval_fullrun_analytics_v2 as A
from eval_fullrun_search import FullRunEval, GAMMA, TOPK, ROLL_CAP
from honest import make_honest
import slaythespire

H = make_honest(A.V2, GAMMA, TOPK, ROLL_CAP)
ev = H("policy_varN.npz", "policy_varE.npz", "macro")
_UND = slaythespire.GameOutcome.UNDECIDED


def obs(ev):
    return ev._obs1235() if ev.env.in_combat else ev._obs1200()


def cont(ev, nsteps):
    """Continue with fixed deterministic policies; return trajectory signature."""
    sig = []
    for _ in range(nsteps):
        if ev.env.gc.outcome != _UND:
            break
        if ev.env.in_combat:
            a = ev.net.greedy(ev._obs1235(), ev._mask())
        else:
            a = ev.vare.act(ev._obs1200(), ev._mask())
        ev.env.step(int(a))
        hp = ev.env.bc.player.cur_hp if ev.env.in_combat else ev.env.gc.cur_hp
        enc = str(ev.env.gc.encounter).split(".")[-1] if ev.env.in_combat else ""
        sig.append((int(ev.env.gc.floor_num), int(hp), enc))
    return sig


fails = 0
n_q = n_q_eq = 0
n_run = n_run_eq = 0
n_vis = n_vis_eq = 0
n_div_combat = n_div_combat_tot = 0
n_div_run = n_div_run_tot = 0

for seed in [10007, 17926, 25845]:
    ev.seed = seed
    ev.history = []
    ev.env.reset(seed=seed)
    combat_checks = 0
    map_checks = 0
    for step in range(1500):
        if ev.env.gc.outcome != _UND:
            break
        ss = str(ev.env.gc.screen_state).split(".")[-1]
        if ev.env.in_combat and step % 7 == 0 and combat_checks < 12:
            combat_checks += 1
            mask = ev._mask()
            legal = [i for i in range(128) if mask[i]][:TOPK]
            live = ev._hsnap()
            for a in legal:
                q_rep = FullRunEval._rollout_q(ev, a)        # replay from TRUE seed
                ev._hload(live)
                q_cpy = ev._hrollout_q(a)                    # exact copy
                n_q += 1
                n_q_eq += int(q_rep == q_cpy)
                if q_rep != q_cpy:
                    print("  PARITY-FAIL combat seed %d step %d a=%d rep=%r copy=%r" % (seed, step, a, q_rep, q_cpy))
            # (b) visible unchanged + different futures
            o0 = obs(ev) if ev._hload(live) else None
            qs = []
            for d in (1, 2, 3, 4, 5):
                ev._hload(live, ev._hdet(d))
                n_vis += 1
                n_vis_eq += int(np.array_equal(o0, obs(ev)))
                qs.append(ev._hrollout_q(legal[0]))
            n_div_combat_tot += 1
            n_div_combat += int(len(set(qs)) > 1)
            ev._hload(live)
        if (not ev.env.in_combat) and ss == "MAP_SCREEN" and map_checks < 4 and step % 3 == 0:
            map_checks += 1
            live = ev._hsnap()
            hist = list(ev.history)
            # replay reference continuation
            ok = FullRunEval._restore(ev)
            ref = cont(ev, 400)
            ev._hload(live)
            cpy = cont(ev, 400)
            n_run += 1
            n_run_eq += int(ref == cpy)
            if ref != cpy:
                print("  PARITY-FAIL run seed %d step %d (len %d vs %d)" % (seed, step, len(ref), len(cpy)))
            ev._hload(live)
            o0 = obs(ev)
            futs = []
            for d in (11, 12, 13):
                ev._hload(live, ev._hdet(d))
                n_vis += 1
                n_vis_eq += int(np.array_equal(o0, obs(ev)))
                futs.append(tuple(cont(ev, 400)))
            n_div_run_tot += 1
            n_div_run += int(len(set(futs)) > 1)
            ev._hload(live)
            ev.history = hist
        # live step (fixed policies) keeping replay history
        if ev.env.in_combat:
            a = ev.net.greedy(ev._obs1235(), ev._mask())
        else:
            a = ev.vare.act(ev._obs1200(), ev._mask())
        ev.env.step(int(a))
        ev.history.append(int(a))
    print("seed %d: floor %d combat_checks %d map_checks %d" % (seed, ev.env.gc.floor_num, combat_checks, map_checks))

print("(a) combat rollout parity copy==replay : %d/%d" % (n_q_eq, n_q))
print("(a) run continuation parity copy==replay: %d/%d" % (n_run_eq, n_run))
print("(b) visible obs unchanged by determinize : %d/%d" % (n_vis_eq, n_vis))
print("(b) combat futures differ across 5 samples: %d/%d states" % (n_div_combat, n_div_combat_tot))
print("(b) run futures differ across 3 samples   : %d/%d states" % (n_div_run, n_div_run_tot))
ok = (n_q_eq == n_q and n_run_eq == n_run and n_vis_eq == n_vis
      and n_div_run == n_div_run_tot and n_div_combat >= 0.5 * n_div_combat_tot)
print("RESULT", "PASS" if ok else "FAIL")
