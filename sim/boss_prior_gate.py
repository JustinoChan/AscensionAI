"""Held-out boss-fight gate for the root policy prior HONEST_ROOT_NET / HONEST_ROOT_C (honest; from boss_expert_gate.py). Usage:
  boss_expert_gate.py SEED0 N OUTDIR    (env: MACRO_* policy flags, POTION_VALUE, GATE_ACT=2|3, GATE_SEEDSTEP)
For each seed: play the run with the given policy + honest 1-ply K=64 until the first decision of the Act GATE_ACT boss
fight; then play that fight to the end once per variant from the identical state and sample counter.
  base     : mainline (POTION_VALUE as set)
  pvdump   : EXPERT_BOSS_PV 0.5,0   (cheap potions at the Act 2 boss, free at the final boss)
  pvzero   : EXPERT_BOSS_PV 0,0
  risk     : EXPERT_BOSS_RISK 0.5   (survival over HP left)
  both     : pvdump + risk
  k128     : base with K=128 in the boss fight (compute control)
Writes OUTDIR/seed_<s>.json with win / HP left / potions left / seconds per variant.
"""
import json, os, sys, time

os.environ.setdefault("HONEST", "1"); os.environ.setdefault("HONEST_CPP", "1"); os.environ.setdefault("HONEST_K", "64")
import eval_fullrun_analytics_v2 as E
import honest as H
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
import slaythespire as S

_LOSS = S.GameOutcome.PLAYER_LOSS
_UND = S.GameOutcome.UNDECIDED
GATE_ACT = int(os.environ.get("GATE_ACT", "2"))
GATE_ROOM = os.environ.get("GATE_ROOM", "BOSS")   # BOSS (default) or ELITE: first fight of that room type in the act
NET = os.environ.get("PRIOR_NET", "nets/exi1/rollout.npz")
VARIANTS = [("base", {})] + [("c%g" % c, {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": c}) for c in (2.0, 5.0, 10.0, 20.0)] +            [("c5roll", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET})]
if os.environ.get("PRIOR_CSWEEP"):   # c sweep around c=5, all with NET as root prior + play-out policy
    VARIANTS = [("base", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET})] +                [("c%groll" % c, {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": c, "_ROLL": NET}) for c in (3.0, 10.0)]
if os.environ.get("PRIOR_HORIZON_ELITE"):   # elite-only horizon with the mainline prior: gamma 0.97 vs 0.99 vs 0.995 + cap 400
    VARIANTS = [("base", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET}),
                ("g99", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET, "HONEST_GAMMA_ELITE": 0.99}),
                ("g995", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET, "HONEST_GAMMA_ELITE": 0.995, "HONEST_ROLL_CAP_ELITE": 400})]
elif os.environ.get("PRIOR_HORIZON"):   # boss-only horizon with the mainline prior: gamma 0.97 (base) vs 0.99 vs 0.995 + cap 400
    VARIANTS = [("base", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET}),
                ("g99", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET, "HONEST_GAMMA_BOSS": 0.99}),
                ("g995", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET, "HONEST_GAMMA_BOSS": 0.995, "HONEST_ROLL_CAP_BOSS": 400})]
elif os.environ.get("PRIOR_KBOSS"):   # boss-fight search budget with the mainline prior: K=64 vs 128 vs 256 (c=5, NET prior + play-outs)
    _kb = lambda k: dict(H.HONEST_K_BY_ROOM, BOSS=k)
    VARIANTS = [("base", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET}),
                ("k128", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET, "HONEST_K_BY_ROOM": _kb(128)}),
                ("k256", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": NET, "HONEST_K_BY_ROOM": _kb(256)})]
elif os.environ.get("PRIOR_AS"):   # afterstate prior (HONEST_ROOT_AS) vs exi1 obs-only prior; both c=5 with exi1 play-outs
    _B = os.environ.get("PRIOR_BASE_NET", NET)
    VARIANTS = [("base", {"HONEST_ROOT_NET": _B, "HONEST_ROOT_AS": "", "HONEST_ROOT_C": 5.0, "_ROLL": _B}),
                ("c5roll", {"HONEST_ROOT_NET": _B, "HONEST_ROOT_AS": os.environ["PRIOR_AS"], "HONEST_ROOT_C": 5.0, "_ROLL": _B})] +                ([("asK", {"HONEST_ROOT_NET": _B, "HONEST_ROOT_AS": os.environ["PRIOR_AS"], "HONEST_ROOT_C": 5.0, "_ROLL": _B,
                              "HONEST_ROOT_AS_K": int(os.environ["PRIOR_AS_K"])})] if os.environ.get("PRIOR_AS_K") else [])
elif os.environ.get("PRIOR_MIX_AS"):   # mixed prior: 0.5 * afterstate + 0.5 * exi1 log-probs vs exi1 alone (c=5, exi1 play-outs)
    _B = os.environ.get("PRIOR_BASE_NET", NET)
    VARIANTS = [("base", {"HONEST_ROOT_NET": _B, "HONEST_ROOT_AS": "", "HONEST_ROOT_C": 5.0, "_ROLL": _B}),
                ("c5roll", {"HONEST_ROOT_NET": _B, "HONEST_ROOT_AS": os.environ["PRIOR_MIX_AS"], "HONEST_ROOT_MIX": 0.5, "HONEST_ROOT_C": 5.0, "_ROLL": _B})]
elif os.environ.get("PRIOR_BASE_NET"):   # pair mode: previous net vs new net, both as c=5 root prior + play-out policy
    _B = os.environ["PRIOR_BASE_NET"]
    VARIANTS = [("base", {"HONEST_ROOT_NET": _B, "HONEST_ROOT_C": 5.0, "_ROLL": _B}),
                ("c5roll", {"HONEST_ROOT_NET": NET, "HONEST_ROOT_C": 5.0, "_ROLL": os.environ.get("PRIOR_ROLL_NET", NET)})]


def _cnet(p, ev):
    import numpy as np
    z = np.load(p)
    return S.CombatNet([z["pbody.%d.weight" % i] for i in (0, 2, 4)] + [z["pi.weight"]],
                       [z["pbody.%d.bias" % i] for i in (0, 2, 4)] + [z["pi.bias"]], ev.maxima.astype("float32"))


class _Done(Exception):
    pass


def run(seed):
    ev = H.make_honest(E.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
    base = ev._search_action
    res = {}

    def hooked():
        gc = ev.env.gc
        if (not getattr(ev, "_in_rollout", False) and str(gc.cur_room).split(".")[-1] == GATE_ROOM and int(gc.act) == GATE_ACT
                and ev.env.in_combat):
            snap = ev._hsnap(); ctr = ev.__dict__.get("_hctr", 0)
            res.update(seed=seed, act=int(gc.act), boss=str(gc.encounter).split(".")[-1], hp_in=int(ev.env.bc.player.cur_hp),
                       max_hp=int(gc.max_hp), pots_in=ev._potion_slots()[0])
            for name, over in VARIANTS:
                ev._hload(snap); ev._hctr = ctr
                ov = {k: v for k, v in over.items() if not k.startswith("_")}
                saved = {k: getattr(H, k) for k in ov}
                for k, v in ov.items():
                    setattr(H, k, v)
                ev._cnet_obj = _cnet(over["_ROLL"], ev) if "_ROLL" in over else None
                t0 = time.time(); steps = 0
                try:
                    while ev.env.in_combat and ev.env.gc.outcome == _UND and steps < 2000:
                        a = base()
                        if a is None:
                            break
                        ev.env.step(int(a)); steps += 1
                finally:
                    for k, v in saved.items():
                        setattr(H, k, v)
                g = ev.env.gc
                won = g.outcome != _LOSS and not ev.env.in_combat
                res[name] = {"won": int(won), "hp_end": int(g.cur_hp) if won else 0, "pots_end": ev._potion_slots()[0] if won else 0,
                             "steps": steps, "secs": round(time.time() - t0, 1)}
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
        r = r or {"seed": s, "none": 1, "reached_act": None}
        json.dump(r, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        print(json.dumps(r), flush=True)
