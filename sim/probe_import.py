"""Isolate the import context: byte-for-byte the shadow test's import block
(pulling in sim_worker's module-level side effects), but the game loop uses a
local guard-only env with NO reward override and deck reads AFTER step()
(the bisect probe's known-clean pattern).

Crash here => the crash trigger is sim_worker's import-time side effects /
heap layout, not the reward logic. Clean here => the trigger is the reward
running inside step().

argv: [1] attempt (seed variation, default 0)
"""
import random
import sys
from sim_worker import SafeStsEnv, _deck_phi, PHI_W, GAMMA  # noqa: F401
import slaythespire

import numpy as np
from gym_sts_lightspeed.envs.sts_env import StsEnv

OBS = 1200
attempt = int(sys.argv[1]) if len(sys.argv) > 1 else 0


class GuardOnlyEnv(StsEnv):
    _last = None

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last = o
            return o
        except Exception:
            return self._last if self._last is not None else np.zeros(OBS, np.int32)


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
    if not env.in_combat:
        sink += _deck_phi(env.gc.deck)
    if term or trunc:
        games += 1
        if games % 20 == 0:
            print("PROGRESS games=%d steps=%d" % (games, steps), flush=True)
        obs, info = env.reset()
print("IMPORT-CONTEXT CLEAN: %d games %d steps (sink %.1f)" % (games, steps, sink))
