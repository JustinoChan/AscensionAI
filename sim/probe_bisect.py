"""Bisect which new native access corrupts the extension. Baseline (no extra
reads) = control_no_reward.py, known clean (3/3 x 200 games). Each mode adds
ONE class of native reads on top of the clean baseline.

argv: [1] mode in {deck, room, both}  [2] attempt (seed variation, default 0)
"""
import random
import sys
import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv

OBS = 1200
mode = sys.argv[1]
attempt = int(sys.argv[2]) if len(sys.argv) > 2 else 0


class GuardOnlyEnv(StsEnv):
    _last = None

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last = o
            return o
        except Exception:
            return self._last if self._last is not None else np.zeros(OBS, np.int32)


def deck_phi(deck):
    nonstarter = 0.0
    phi = 0.0
    for c in deck:
        if "CURSE" in str(c.type):
            phi -= 1.5
            continue
        if c.is_starter_strike_or_defend:
            phi -= 0.3
        else:
            nonstarter += 1.0
        phi += 0.5 * int(c.upgrade_count)
    return phi + min(nonstarter, 20.0)


env = GuardOnlyEnv(ascension=0)
rng = random.Random(7 + attempt)
obs, info = env.reset(seed=4242 + 1000 * attempt)
games = steps = 0
sink = 0.0
while games < 200:
    mask = info["action_mask"]
    legal = [i for i in range(len(mask)) if mask[i]]
    obs, r, term, trunc, info = env.step(rng.choice(legal))
    steps += 1
    if mode in ("deck", "both") and not env.in_combat:
        sink += deck_phi(env.gc.deck)
    if mode in ("room", "both"):
        sink += len(str(env.gc.cur_room)) + int(env.gc.act)
    if term or trunc:
        games += 1
        if games % 20 == 0:
            print("PROGRESS games=%d steps=%d" % (games, steps), flush=True)
        obs, info = env.reset()
print("MODE %s CLEAN: %d games %d steps (sink %.1f)" % (mode, games, steps, sink))
