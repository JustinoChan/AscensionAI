"""Diagnose the value-bootstrap teacher collapse: play eval bosses with the
value-teacher search and report fight length + outcome category. Distinguishes
'fights time out / reward-farming' (steps->500, still in combat) from 'fast
loss'."""
import warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import slaythespire
from sim_worker_search import Searcher
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
s = Searcher("bank_train.jsonl", "fight_config.json", "policy_live.npz", 0)
print("zmean=%.3f zstd=%.3f" % (s.net.zmean, s.net.zstd))
bosses = [i for i, e in enumerate(s.eval_bank) if e.get("room") == "BOSS"][:6]
for j in bosses:
    s.entry, s.live = s.eval_bank[j], []
    if not s._restore():
        print(j, "restore-fail")
        continue
    steps, bq = 0, 0.0
    while s.env.in_combat and steps < 500:
        mask = s.env._get_info()["action_mask"]
        legal = [i for i in range(128) if mask[i]]
        if len(legal) > 6:
            prior = s.net.logits(s._obs(), mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:6]
        best, bq = legal[0], -1e18
        for a in legal:
            qa = s._q_value(a)
            if qa is not None and qa > bq:
                bq, best = qa, a
        if not s._restore():
            break
        s.env.step(best)
        s.live.append(best)
        steps += 1
    if s.env.in_combat:
        cat = "TIMEOUT(farming)"
    elif s.env.gc.outcome == _LOSS:
        cat = "DIED"
    else:
        cat = "WON"
    print("boss[%d] steps=%3d in_combat=%s -> %-16s bestQ=%.2f"
          % (j, steps, s.env.in_combat, cat, bq))
