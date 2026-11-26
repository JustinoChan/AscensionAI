"""Combat expert iteration, data collection (honest). Usage:
  exi_collect.py ROLLOUT_NPZ SEED0 N OUTDIR      (env: MACRO_* mainline policy flags, EXI_K label samples, EXI_EPS
                                                  episodes per fight, EXI_RUN_K search K for the run itself, EXI_SEEDSTEP)
For each seed: play the run with the mainline out-of-combat policy and honest 1-ply search (K=EXI_RUN_K). At the first
decision of every Act 2/3 ELITE or BOSS fight: snapshot; play EXI_EPS episodes of that fight, each in a freshly
DETERMINIZED world (hidden draw order / RNG resampled), choosing actions GREEDILY with the current play-out net (the same
rule as the C++ rollouts, 10% random legal action for coverage), i.e. on the states the play-outs actually visit.
Label EVERY visited state with honest search: mean over EXI_K sampled worlds of the rollout Q of every legal action
(rollout_qb with the current play-out net -- exactly what 1-ply search computes). Then restore the live state and the
sample counter and continue the run. Writes OUTDIR/seed_<s>.npz: obs [N,1235] f16, q [N,128] f32 (nan = illegal),
n_legal, act, room, boss/elite name per state.
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("HONEST", "1"); os.environ.setdefault("HONEST_CPP", "1")
os.environ.setdefault("HONEST_K", os.environ.get("EXI_RUN_K", "16"))
import eval_fullrun_analytics_v2 as E
import honest as H
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
import slaythespire as S

EXI_DOUBLE = os.environ.get("EXI_DOUBLE", "0") == "1"
EXI_AFTER = os.environ.get("EXI_AFTER", "0") == "1"
EXI_ACT = os.environ.get("EXI_ACT", "net")   # "net": greedy play-out net (default) | "search": the root search's choice
K = int(os.environ.get("EXI_K", "64")); EPS = int(os.environ.get("EXI_EPS", "2")); EPSILON = 0.1
_UND = S.GameOutcome.UNDECIDED


def run(seed, rollout_npz):
    ev = H.make_honest(E.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
    z = np.load(rollout_npz)
    cn = S.CombatNet([z["pbody.%d.weight" % i] for i in (0, 2, 4)] + [z["pi.weight"]],
                     [z["pbody.%d.bias" % i] for i in (0, 2, 4)] + [z["pi.bias"]], ev.maxima.astype("float32"))
    L = [(z["pbody.%d.weight" % i], z["pbody.%d.bias" % i]) for i in (0, 2, 4)] + [(z["pi.weight"], z["pi.bias"])]
    base = ev._search_action
    rng = np.random.default_rng(seed * 7919 + 13)
    D = {"obs": [], "q": [], "nl": [], "act": [], "room": [], "enc": []}
    st = {"seen": set()}

    def logits(o):
        h = o
        for li, (W, b) in enumerate(L):
            x = h @ W.T + b
            h = x if li == len(L) - 1 else 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
        return h

    def label():
        e = ev.env
        n = min(len(e.bc.get_possible_actions()), 128)
        legal = list(range(n))
        dets = [ev._hdet(k) for k in range(K)]
        last = getattr(e, "_last", None)
        sc, _ = S.rollout_qb(e.gc, e.bc, legal, dets, cn, 0, GAMMA, ROLL_CAP, False, [] if last is None else [int(v) for v in last])
        q = np.full(128, np.nan, np.float32)
        q[:n] = [float(np.mean(r)) for r in sc]
        return q, n

    def hooked():
        gc = ev.env.gc
        room = str(gc.cur_room).split(".")[-1]
        key = (int(gc.act), int(gc.floor_num))
        if (not getattr(ev, "_in_rollout", False) and ev.env.in_combat and room in ("ELITE", "BOSS") and int(gc.act) >= int(os.environ.get("EXI_MIN_ACT", "2"))
                and key not in st["seen"]):
            st["seen"].add(key)
            snap = ev._hsnap(); ctr = ev.__dict__.get("_hctr", 0)
            enc = str(gc.encounter).split(".")[-1]
            for ep in range(EPS):
                ev._hload(snap, int(rng.integers(1, 2 ** 62)))
                steps = 0
                while ev.env.in_combat and ev.env.gc.outcome == _UND and steps < 400:
                    n = min(len(ev.env.bc.get_possible_actions()), 128)
                    if n <= 0:
                        break
                    obs = ev._obs1235().astype(np.float32)
                    q, _ = label()
                    if EXI_DOUBLE:                 # label-noise diagnostic: a second, independent set of sampled worlds
                        q2, _ = label(); D.setdefault("q2", []).append(q2)
                    D["obs"].append(obs.astype(np.float16)); D["q"].append(q); D["nl"].append(n)
                    if EXI_AFTER:                  # afterstates of every legal action in one sampled world (honest)
                        import exi_as
                        D.setdefault("aobs", []).append(exi_as.afterstates(ev, n, int(rng.integers(1, 2 ** 62))).astype(np.float16))
                    D["act"].append(int(gc.act)); D["room"].append(room); D["enc"].append(enc)
                    if EXI_ACT == "search":       # follow the live search's own choice (root-state distribution of the mainline)
                        sc = np.nan_to_num(q[:n], nan=-1e9)
                        if H.HONEST_ROOT_NET and H.HONEST_ROOT_C:
                            sc = sc + H.HONEST_ROOT_C * ev._root_logpi()[:n]
                        a = int(rng.integers(n)) if rng.random() < 0.05 else int(np.argmax(sc))
                    else:
                        a = int(rng.integers(n)) if rng.random() < EPSILON else int(np.argmax(logits(obs)[:n]))
                    ev.env.step(a); steps += 1
            ev._hload(snap); ev._hctr = ctr
        return base()

    ev._search_action = hooked
    t0 = time.time()
    try:
        out = ev.play_v2(seed)
    except Exception as ex:
        out = {"error": str(ex)[:200]}
    return D, out, time.time() - t0


if __name__ == "__main__":
    net, seed0, n, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    step = int(os.environ.get("EXI_SEEDSTEP", "31"))
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        s = seed0 + i * step
        p = os.path.join(outdir, "seed_%d.npz" % s)
        if os.path.exists(p) or os.path.exists(p + ".none"):
            continue
        D, out, secs = run(s, net)
        if D["obs"]:
            np.savez_compressed(p + ".tmp.npz", obs=np.array(D["obs"]), q=np.array(D["q"]), n_legal=np.array(D["nl"], np.int16),
                                act=np.array(D["act"], np.int8), room=np.array(D["room"], "U5"), enc=np.array(D["enc"], "U24"),
                                **({"q2": np.array(D["q2"])} if D.get("q2") else {}),
                                **({"aobs": np.concatenate(D["aobs"])} if D.get("aobs") else {}))
            os.replace(p + ".tmp.npz", p)
        else:
            open(p + ".none", "w").write(json.dumps(out)[:300])
        print(json.dumps({"seed": s, "states": len(D["obs"]), "floor": out.get("floor"), "secs": round(secs, 1)}), flush=True)
