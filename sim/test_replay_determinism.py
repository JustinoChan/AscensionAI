"""Load-bearing: replaying (bank entry + live actions) must reproduce the exact
mid-fight state. Uses FightEnv from sim_worker_fight (reset_to_entry replays the
bank prefix; we then replay extra in-combat actions and compare)."""
import sys
import json
import random
sys.path.insert(0, ".")
import slaythespire as sts
from sim_worker_fight import FightEnv

entries = [json.loads(l) for l in open("bank_train.jsonl") if l.strip()][:20]
env = FightEnv(ascension=0)
rng = random.Random(0)
ok = 0
for k, e in enumerate(entries):
    if env.reset_to_entry(e) is None:
        continue
    live = []
    for _ in range(rng.randint(1, 4)):
        if not env.in_combat or env.bc.outcome != sts.BattleOutcome.UNDECIDED:
            break
        acts = env._get_info()["action_mask"]
        legal = [i for i in range(128) if acts[i]]
        a = rng.choice(legal)
        live.append(a)
        env.step(a)
    if not env.in_combat:
        continue
    snap = (env.gc.floor_num, env.bc.player.cur_hp,
            sum(m.cur_hp for m in env.bc.monsters.monsters if m.is_alive()),
            tuple(env._get_info()["action_mask"]))
    assert env.reset_to_entry(e) is not None
    for a in live:
        env.step(a)
    snap2 = (env.gc.floor_num, env.bc.player.cur_hp,
             sum(m.cur_hp for m in env.bc.monsters.monsters if m.is_alive()),
             tuple(env._get_info()["action_mask"]))
    assert snap == snap2, "entry %d: replay mismatch\n%s\n%s" % (k, snap, snap2)
    ok += 1
print("PASS: %d fights replayed deterministically" % ok)
assert ok >= 10
