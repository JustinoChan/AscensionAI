"""Over the eval bank, the SEARCH policy must win more than the BASE greedy
policy -- otherwise search adds no signal and the approach is mis-specified.
Crash-tolerant via a seq-retry wrapper (sampled rollouts can segfault the fork)."""
import sys
import json
sys.path.insert(0, ".")
import numpy as np
from sim_worker_search import Searcher

s = Searcher("bank_train.jsonl", "fight_config.json", "policy_live.npz", 0)
n = len(s.eval_bank)
base = [s._play(i, False) for i in range(n)]
srch = [s._play(i, True) for i in range(n)]
bt = {}
st = {}
for i, e in enumerate(s.eval_bank):
    bt.setdefault(e["room"], []).append(base[i])
    st.setdefault(e["room"], []).append(srch[i])


def wr(d):
    return {k: round(100 * float(np.mean(v)), 0) for k, v in d.items()}


print("BASE  greedy:", wr(bt), "all", round(100 * np.mean(base)))
print("SEARCH      :", wr(st), "all", round(100 * np.mean(srch)))
print("VERDICT:", "PASS" if np.mean(srch) > np.mean(base) + 0.03 else "INVESTIGATE")
