"""Shadow-check the fight reward on random in-process fights: every step's
reward must equal the player HP delta exactly (legal actions only, so no
invalid-action penalty), and end-of-fight classification must be coherent."""
import json
import random
import sys

sys.path.insert(0, ".")
from sim_worker_fight import FightEnv, MAX_FIGHT_STEPS  # noqa: E402
import slaythespire  # noqa: E402

_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
entries = [json.loads(ln) for ln in open("bank_train.jsonl") if ln.strip()]
rng = random.Random(7)
env = FightEnv(ascension=0)
fights = checked = 0
while fights < 30:
    e = rng.choice(entries)
    if env.reset_to_entry(e) is None:
        continue
    steps = 0
    while steps < MAX_FIGHT_STEPS:
        hp_before = env.bc.player.cur_hp if env.in_combat else env.gc.cur_hp
        acts = (env.bc.get_possible_actions() if env.in_combat
                else env.gc.get_possible_actions())
        a = rng.randrange(max(1, len(acts)))
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        hp_after = env.bc.player.cur_hp if (env.in_combat and env.bc) \
            else env.gc.cur_hp
        expect = float(hp_after - hp_before)
        assert abs(r - expect) < 1e-6, \
            "step reward %.3f != hp delta %.3f" % (r, expect)
        checked += 1
        if term or not env.in_combat:
            if env.gc.outcome != _LOSS:
                assert not env.in_combat, "won but still in combat?"
            fights += 1
            break
print("PASS: %d fights, %d steps reward-exact" % (fights, checked))
