"""Phase 3 step 3.6: combat headroom check (diagnostic).

  headroom.py SEED [SEED ...]   (env: HONEST=1 HONEST_CPP=1 HONEST_K=64, same as eval500_K64)

Replays an eval500 seed with the mainline (deterministic, so it follows the recorded run) and, at the
first decision of every Act 2/3 ELITE or BOSS fight, plays that fight to its end twice from the
identical state:
  honest  : the mainline's own honest 1-ply K=64 search (sampled worlds)
  clair   : the same 1-ply search on the EXACT state (true draw order / RNG; diagnostic only)
then restores the live state AND the sample counter, so the replay continues exactly as recorded.
Output: sim_logs/headroom/seed_<s>.json  [{act, floor, room, enc, hp_in, max_hp, honest:{won,hp_end,steps}, clair:{...}}]
"""
import json, os, sys, time

os.environ.setdefault("HONEST", "1")
sys.argv = [sys.argv[0]] + sys.argv[1:]
import eval_fullrun_analytics_v2 as E
from honest import make_honest
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
import slaythespire as S

OUT = "sim_logs/headroom"
os.makedirs(OUT, exist_ok=True)
_LOSS = S.GameOutcome.PLAYER_LOSS


def run(seed):
    ev = make_honest(E.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
    base = ev._search_action
    st = {"busy": False, "seen": set(), "rows": []}

    def play_fight(clair):
        ev._clair = clair
        steps = 0
        try:
            while ev.env.in_combat and ev.env.gc.outcome == S.GameOutcome.UNDECIDED and steps < 2000:
                a = base()
                if a is None:
                    break
                ev.env.step(int(a))
                steps += 1
        finally:
            ev._clair = False
        g = ev.env.gc
        return {"won": int(g.outcome != _LOSS), "hp_end": 0 if g.outcome == _LOSS else int(g.cur_hp), "steps": steps}

    def hooked():
        e = ev.env
        gc = e.gc
        room = str(gc.cur_room).split(".")[-1]
        key = int(gc.floor_num)
        if (not st["busy"] and not getattr(ev, "_in_rollout", False) and int(gc.act) >= 2 and room in ("ELITE", "BOSS")
                and key not in st["seen"]):
            st["seen"].add(key)
            st["busy"] = True
            live = ev._hsnap()
            ctr = ev.__dict__.get("_hctr", 0)
            hist = list(ev.history)
            row = {"seed": seed, "act": int(gc.act), "floor": key, "room": room, "enc": str(gc.encounter).split(".")[-1],
                   "hp_in": int(e.bc.player.cur_hp), "max_hp": int(gc.max_hp)}
            t0 = time.time()
            for mode in ("honest", "clair"):
                ev._hload(live)
                ev._hctr = ctr
                row[mode] = play_fight(mode == "clair")
            row["secs"] = round(time.time() - t0, 1)
            ev._hload(live)
            ev._hctr = ctr                 # replay continues exactly as recorded
            ev.history = hist
            st["rows"].append(row)
            st["busy"] = False
        return base()

    ev._search_action = hooked
    out = ev.play_v2(seed)
    json.dump({"seed": seed, "floor": out["floor"], "win": out["win"], "fights": st["rows"]},
              open(os.path.join(OUT, "seed_%d.json" % seed), "w"))
    return out


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(int(s))
