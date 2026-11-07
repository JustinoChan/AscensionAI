"""Per-decision cost of combat engines on the same live combat states:
1-ply native rollouts (K samples x top-6 candidates) vs ensemble MCTS (K trees x S sims).
Also checks MCTS honesty plumbing (exact=True with one tree is the clairvoyant scum search)."""
import os
import time
os.environ.setdefault("HONEST", "1")
import numpy as np
import eval_fullrun_analytics_v2 as A
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
from honest import make_honest
import slaythespire as S

ev = make_honest(A.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
cn = ev._cnet()
states = []
for seed in [10007, 17926, 25845, 33764]:
    ev.seed = seed
    ev.env.reset(seed=seed)
    for step in range(3000):
        if ev.env.gc.outcome != S.GameOutcome.UNDECIDED:
            break
        if ev.env.in_combat and step % 4 == 0 and len(states) < 80:
            states.append((ev.env.gc.copy(), ev.env.bc.copy(), ev._mask().copy(), ev._obs1235().copy()))
        a = ev.net.greedy(ev._obs1235(), ev._mask()) if ev.env.in_combat else ev.vare.act(ev._obs1200(), ev._mask())
        ev.env.step(int(a))
print("states:", len(states))
dets = lambda K: [(0xABCDEF + 7919 * k) & ((1 << 64) - 1) for k in range(K)]
for K in (8, 16, 32):
    t0 = time.time()
    for gc, bc, mask, o in states:
        legal = [i for i in range(128) if mask[i]]
        if len(legal) > TOPK:
            pr = ev.net.logits(o, mask)
            legal = sorted(legal, key=lambda a: pr[a], reverse=True)[:TOPK]
        S.rollout_qb(gc, bc, legal, dets(K), cn, 0, GAMMA, ROLL_CAP, False)
    print("1ply native K=%-3d : %.1f ms/decision" % (K, 1000 * (time.time() - t0) / len(states)))
for K, sims in ((8, 50), (8, 200), (8, 1000), (16, 500)):
    t0 = time.time()
    for gc, bc, mask, o in states:
        S.mcts_root(bc, dets(K), sims, False)
    print("mcts K=%-3d sims/tree=%-5d: %.1f ms/decision" % (K, sims, 1000 * (time.time() - t0) / len(states)))
