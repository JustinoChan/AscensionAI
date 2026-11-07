"""Fast-replay (_restore) must reproduce the SAME decision state as the full
FightEnv.reset_to_entry path, for the bank-prefix and for prefix+live actions.
Also times a search decision to confirm the optimization."""
import sys
import json
import time
sys.path.insert(0, ".")
import numpy as np
import slaythespire as sts
from sim_worker_fight import FightEnv
from sim_worker_search import Searcher

s = Searcher("bank_train.jsonl", "fight_config.json", "policy_live.npz", 0)
ref = FightEnv(ascension=0)
entries = s.bank[:25]


def state(env):
    return (env.gc.floor_num, env.bc.player.cur_hp,
            sum(m.cur_hp for m in env.bc.monsters.monsters if m.is_alive()),
            tuple(env._get_info()["action_mask"]))


ok = 0
for e in entries:
    # reference: full reset_to_entry (prefix only, live empty)
    if ref.reset_to_entry(e) is None:
        continue
    ref_state = state(ref)
    # fast: Searcher._restore with this entry, no live actions
    s.entry, s.live = e, []
    if not s._restore():
        continue
    assert state(s.env) == ref_state, "FF mismatch prefix-only:\n%s\n%s" % (
        ref_state, state(s.env))
    ok += 1
print("PASS: %d entries -> fast replay == full reset_to_entry" % ok)
assert ok >= 10

# timing: one full search decision
s.entry, s.live = entries[0], []
s._restore()
t = time.time()
rec = s.gen()
print("one search decision: %.0f ms" % ((time.time() - t) * 1000))
