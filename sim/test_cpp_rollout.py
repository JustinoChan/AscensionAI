"""Phase 2 step 2: native C++ rollouts vs the Python honest rollout.
Same states, same candidates, same determinization seeds.
Reports exact-score agreement, |diff|, chosen-action agreement and speed."""
import os
import sys
import time
os.environ.setdefault("HONEST", "1")
import numpy as np
import eval_fullrun_analytics_v2 as A
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
from honest import make_honest
import slaythespire as S

H = make_honest(A.V2, GAMMA, TOPK, ROLL_CAP)
ev = H("policy_varN.npz", "policy_varE.npz", "macro")
n = ev.net
cnet = S.CombatNet([w for w, _ in n.pbody] + [n.pi[0]], [b for _, b in n.pbody] + [n.pi[1]],
                   ev.maxima.astype(np.float32))
_UND = S.GameOutcome.UNDECIDED
K = 8

# feature parity on live states first
obs_eq = obs_n = 0
states = 0
eq = tot = 0
maxdiff = 0.0
pick_eq = pick_n = 0
t_py = t_cpp = t_rand = t_batch = 0.0
b_eq = b_tot = b_pick = 0
steps_cpp = 0
for seed in [10007, 17926, 25845, 33764, 41683]:
    ev.seed = seed
    ev.history = []
    ev.env.reset(seed=seed)
    for step in range(3000):
        if ev.env.gc.outcome != _UND:
            break
        if ev.env.in_combat and step % 5 == 0 and states < 60:
            o_py = ev._obs1235()
            o_c = np.asarray(cnet.obs(ev.env.gc, ev.env.bc), np.float32)
            obs_n += 1
            obs_eq += int(np.array_equal(o_py, o_c))
            mask = ev._mask()
            legal = [i for i in range(128) if mask[i]]
            if len(legal) > TOPK:
                prior = ev.net.logits(o_py, mask)
                legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
            live = ev._hsnap()
            dets = [ev._hdet(k) for k in range(K)]
            t0 = time.time()
            py = []
            for a in legal:
                row = []
                for d in dets:
                    ev._hload(live, d)
                    row.append(ev._hrollout_q(a))
                py.append(row)
            t_py += time.time() - t0
            ev._hload(live)
            t0 = time.time()
            cc, st = S.rollout_q(ev.env.gc, ev.env.bc, legal, dets, cnet, 0, GAMMA, ROLL_CAP, False)
            t_cpp += time.time() - t0
            steps_cpp += st
            t0 = time.time()
            cb, stb = S.rollout_qb(ev.env.gc, ev.env.bc, legal, dets, cnet, 0, GAMMA, ROLL_CAP, False)
            t_batch += time.time() - t0
            for r1, r2 in zip(py, cb):
                for x, y in zip(r1, r2):
                    b_tot += 1
                    b_eq += int(x == y)
            b_pick += int(legal[int(np.argmax([np.mean(r) for r in cb]))] == legal[int(np.argmax([np.mean(r) for r in py]))])
            t0 = time.time()
            S.rollout_qb(ev.env.gc, ev.env.bc, legal, dets, None, 1, GAMMA, ROLL_CAP, False)
            t_rand += time.time() - t0
            states += 1
            for r1, r2 in zip(py, cc):
                for x, y in zip(r1, r2):
                    tot += 1
                    eq += int(x == y)
                    maxdiff = max(maxdiff, abs(x - y))
            bp = legal[int(np.argmax([np.mean(r) for r in py]))]
            bc_ = legal[int(np.argmax([np.mean(r) for r in cc]))]
            pick_n += 1
            pick_eq += int(bp == bc_)
        if ev.env.in_combat:
            a = ev.net.greedy(ev._obs1235(), ev._mask())
        else:
            a = ev.vare.act(ev._obs1200(), ev._mask())
        ev.env.step(int(a))
        ev.history.append(int(a))
print("obs1235 byte-identical python vs C++   : %d/%d" % (obs_eq, obs_n))
print("rollout scores identical (cand x K=%d)  : %d/%d   max|diff| %.4g" % (K, eq, tot, maxdiff))
print("chosen action identical                : %d/%d" % (pick_eq, pick_n))
print("batched+AVX2 (avx2=%s): scores identical %d/%d, chosen action identical %d/%d, time %.2fs (x%.1f vs python)" % (
    S.cpu_has_avx2(), b_eq, b_tot, b_pick, pick_n, t_batch, t_py / max(t_batch, 1e-9)))
print("time: python %.2fs  C++ net %.2fs  (x%.1f)  C++ random %.2fs  over %d decisions; C++ net steps/s %.0f" % (
    t_py, t_cpp, t_py / max(t_cpp, 1e-9), t_rand, states, steps_cpp / max(t_cpp, 1e-9)))
