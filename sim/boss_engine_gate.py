"""Held-out boss-fight gate for combat ENGINES (honest). Usage:
  boss_engine_gate.py SEED0 N OUTDIR        (env: MACRO_MAPFEAT=1 MACRO_POLICY=<best out-of-combat policy>, GATE_SEEDSTEP)
For each seed: play the run with the best policy and honest 1-ply K=64 until the first decision of an Act 2/3 boss
fight, then play that fight to the end once per engine variant, each from the identical state and sample counter:
  k64      : honest 1-ply, K=64 (mainline)
  mcts16   : honest ensemble MCTS, 16 determinized trees x 1000 sims
  mcts32   : honest ensemble MCTS, 32 determinized trees x 2000 sims
Writes OUTDIR/seed_<s>.json with win / HP left / seconds per variant.
"""
import json, os, sys, time

os.environ.setdefault("HONEST", "1"); os.environ.setdefault("HONEST_CPP", "1"); os.environ.setdefault("HONEST_K", "64")
import eval_fullrun_analytics_v2 as E
import honest as H
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
import slaythespire as S

_LOSS = S.GameOutcome.PLAYER_LOSS
_UND = S.GameOutcome.UNDECIDED
VARIANTS = [("k64", "", 0, 0), ("mcts16", "mcts", 16, 1000), ("mcts32", "mcts", 32, 2000), ("mcts64", "mcts", 64, 4000)]


class _Done(Exception):
    pass


def run(seed):
    ev = H.make_honest(E.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
    base = ev._search_action
    res = {}

    def hooked():
        gc = ev.env.gc
        if (not getattr(ev, "_in_rollout", False) and not H.HONEST_ENGINE_BOSS
                and str(gc.cur_room).split(".")[-1] == "BOSS" and int(gc.act) >= 2):
            snap = ev._hsnap(); ctr = ev.__dict__.get("_hctr", 0)
            res.update(seed=seed, act=int(gc.act), boss=str(gc.encounter).split(".")[-1], hp_in=int(ev.env.bc.player.cur_hp),
                       max_hp=int(gc.max_hp))
            for name, eng, trees, sims in VARIANTS:
                ev._hload(snap); ev._hctr = ctr
                H.HONEST_ENGINE_BOSS, H.MCTS_TREES_BOSS = eng, trees
                if sims:
                    H.MCTS_SIMS_BY_ROOM["BOSS"] = sims
                t0 = time.time(); steps = 0
                try:
                    while ev.env.in_combat and ev.env.gc.outcome == _UND and steps < 2000:
                        a = base()
                        if a is None:
                            break
                        ev.env.step(int(a)); steps += 1
                finally:
                    H.HONEST_ENGINE_BOSS, H.MCTS_TREES_BOSS = "", 0
                g = ev.env.gc
                won = g.outcome != _LOSS and not ev.env.in_combat
                res[name] = {"won": int(won), "hp_end": int(g.cur_hp) if won else 0, "steps": steps, "secs": round(time.time() - t0, 1)}
            raise _Done()
        return base()

    ev._search_action = hooked
    try:
        ev.play_v2(seed)
    except _Done:
        pass
    return res


if __name__ == "__main__":
    seed0, n, outdir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    step = int(os.environ.get("GATE_SEEDSTEP", "31"))
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        s = seed0 + i * step
        p = os.path.join(outdir, "seed_%d.json" % s)
        if os.path.exists(p):
            continue
        r = run(s)
        json.dump(r or {"seed": s, "none": 1}, open(p, "w"))
        print(json.dumps(r or {"seed": s, "none": 1}), flush=True)
