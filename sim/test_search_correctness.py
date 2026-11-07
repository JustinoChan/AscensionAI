"""In-process: the Searcher's pi_target is a valid distribution over legal
actions, z is finite, and a search decision runs end to end on real bank fights."""
import sys
sys.path.insert(0, ".")
import numpy as np
from sim_worker_search import Searcher

s = Searcher("bank_train.jsonl", "fight_config.json", "policy_live.npz", 0)
ok = 0
for _ in range(30):
    rec = s.gen()
    if rec is None:
        break
    obs, pi, z, mask, done, outcome = rec
    assert obs.shape == (1235,) and pi.shape == (128,) and mask.shape == (128,)
    assert np.isfinite(z)
    legal = mask > 0
    assert abs(pi[legal].sum() - 1.0) < 1e-4, "pi not normalized over legal"
    assert pi[~legal].sum() == 0.0, "mass on illegal actions"
    ok += 1
print("PASS: %d search decisions valid" % ok)
assert ok >= 25
