"""Greedy (no-search) win-rate of a policy npz over the eval bank, by tier.
Documents the 'before' numbers on the expanded eval set."""
import sys, warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sim_worker_search import Searcher
npz = sys.argv[1] if len(sys.argv) > 1 else "policy_varL.npz"
s = Searcher("bank_train.jsonl", "fight_config.json", npz, 0)
by = {}
for i, e in enumerate(s.eval_bank):
    by.setdefault(e.get("room", "?"), []).append(i)
print("== %s ==" % npz)
for room in ("MONSTER", "ELITE", "BOSS"):
    idxs = by.get(room, [])
    if not idxs:
        continue
    wins = sum(s._play(i, False) for i in idxs)
    print("  greedy %-7s: %2d/%2d = %.1f%%" % (room, wins, len(idxs),
                                               100.0 * wins / len(idxs)))
