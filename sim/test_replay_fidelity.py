"""Replay 20 bank entries through FightEnv.reset_to_entry and assert the
rebuilt state matches the harvest record. Also sanity-checks the obs deck
region (the encoding-corruption canary)."""
import json
import sys

sys.path.insert(0, ".")
from sim_worker_fight import FightEnv  # noqa: E402  (torch-free import)

N_CHECK = 20
entries = [json.loads(ln) for ln in open("bank_train.jsonl") if ln.strip()]
entries = entries[:N_CHECK]
env = FightEnv(ascension=0)
for k, e in enumerate(entries):
    out = env.reset_to_entry(e)
    assert out is not None, "entry %d: replay failed" % k
    assert env.in_combat, "entry %d: not in combat after replay" % k
    assert env.gc.floor_num == e["floor"], \
        "entry %d: floor %d != %d" % (k, env.gc.floor_num, e["floor"])
    hp_frac = env.gc.cur_hp / max(1, env.gc.max_hp)
    assert abs(hp_frac - e["hp_frac"]) < 0.01, \
        "entry %d: hp_frac %.3f != %.3f" % (k, hp_frac, e["hp_frac"])
    room = str(env.gc.cur_room).split(".")[-1]
    assert room == e["room"], "entry %d: room %s != %s" % (k, room, e["room"])
    obs = out[0]
    deck = obs[19:239]
    assert (deck > 0).sum() >= 2, \
        "entry %d: deck region collapsed (%d nonzero)" % (k, (deck > 0).sum())
print("PASS %d/%d entries replayed exactly, deck region sane" % (
    len(entries), N_CHECK))
