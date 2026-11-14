"""Gate for the boss-fight combat RL net INSIDE honest K=64 search (held-out boss fights).

  boss_gate.py NEW_NET SEED0 N OUTDIR      (env: MACRO_MAPFEAT=1 MACRO_POLICY=<best out-of-combat policy>)
For each seed (fresh range, never used in boss-RL training or validation): play the run with the best
out-of-combat policy and honest K=64 search until the first decision of an Act 2/3 boss fight; then play that
fight to the end TWICE from the identical state with the identical search-sample stream (same _hctr):
  base : rollout/default policy = varN (current mainline)
  new  : rollout/default policy = NEW_NET (what HONEST_ROLLOUT_NET would use)
and record win + HP left. Writes OUTDIR/seed_<s>.json.
"""
import json, os, sys, time

os.environ.setdefault("HONEST", "1"); os.environ.setdefault("HONEST_CPP", "1"); os.environ.setdefault("HONEST_K", "64")
import numpy as np
import eval_fullrun_analytics_v2 as E
from honest import make_honest
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
import slaythespire as S

_LOSS = S.GameOutcome.PLAYER_LOSS
_UND = S.GameOutcome.UNDECIDED


class _Done(Exception):
    pass


def cnet(path, maxima):
    z = np.load(path)
    return S.CombatNet([z["pbody.%d.weight" % i] for i in (0, 2, 4)] + [z["pi.weight"]],
                       [z["pbody.%d.bias" % i] for i in (0, 2, 4)] + [z["pi.bias"]], maxima.astype("float32"))


def run(seed, new_path):
    ev = make_honest(E.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
    base = ev._search_action
    nets = {"base": cnet("policy_varN.npz", ev.maxima), "new": cnet(new_path, ev.maxima)}
    res = {}

    def hooked():
        gc = ev.env.gc
        if not getattr(ev, "_in_rollout", False) and str(gc.cur_room).split(".")[-1] == "BOSS" and int(gc.act) >= 2:
            snap = ev._hsnap(); ctr = ev.__dict__.get("_hctr", 0)
            res.update(seed=seed, act=int(gc.act), boss=str(gc.encounter).split(".")[-1], hp_in=int(ev.env.bc.player.cur_hp),
                       max_hp=int(gc.max_hp))
            for mode in ("base", "new"):
                ev._hload(snap); ev._hctr = ctr; ev._cnet_obj = nets[mode]
                t0 = time.time(); steps = 0
                while ev.env.in_combat and ev.env.gc.outcome == _UND and steps < 2000:
                    a = base()
                    if a is None:
                        break
                    ev.env.step(int(a)); steps += 1
                g = ev.env.gc
                won = g.outcome != _LOSS and not ev.env.in_combat
                res[mode] = {"won": int(won), "hp_end": int(g.cur_hp) if won else 0, "steps": steps, "secs": round(time.time() - t0, 1)}
            raise _Done()
        return base()

    ev._search_action = hooked
    try:
        ev.play_v2(seed)
    except _Done:
        pass
    return res


if __name__ == "__main__":
    new_path, seed0, n, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    step = int(os.environ.get("GATE_SEEDSTEP", "31"))
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        s = seed0 + i * step
        p = os.path.join(outdir, "seed_%d.json" % s)
        if os.path.exists(p):
            continue
        r = run(s, new_path)
        json.dump(r or {"seed": s, "none": 1}, open(p, "w"))
        print(json.dumps(r or {"seed": s, "none": 1}), flush=True)
