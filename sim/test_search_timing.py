"""Realistic per-decision timing across the sampled bank distribution
(loss-oversampled -> includes deep boss fights), single worker, threads=1."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
from sim_worker_search import Searcher

s = Searcher("bank_train.jsonl", "fight_config.json", "policy_live.npz", 7)
# warm up one fight
s.gen()
N = 40
t = time.time()
done = 0
for _ in range(N):
    rec = s.gen()
    if rec is not None:
        done += 1
dt = (time.time() - t) / max(1, done) * 1000
print("avg search decision: %.0f ms (%d decisions) -> ~%.0f dec/s on 12 workers"
      % (dt, done, 12 * 1000.0 / dt))
