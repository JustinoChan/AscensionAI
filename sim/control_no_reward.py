"""Control for the Variant C shadow test: identical trajectory (same env seed,
same action rng), obs carry-forward guard only -- NO reward override. If this
also segfaults at the same game index, the crash is the fork's native
getObservation failure, not the Variant C reward code.

argv: [1] attempt index (varies seeds; default 0 = same seeds as the shadow test)
"""
import random
import sys
import numpy as np
import gym_sts_lightspeed  # noqa: F401
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
while games < 200:
    mask = info["action_mask"]
    legal = [i for i in range(len(mask)) if mask[i]]
    obs, r, term, trunc, info = env.step(rng.choice(legal))
    steps += 1
    if term or trunc:
        games += 1
        if games % 10 == 0:
            print("PROGRESS games=%d steps=%d" % (games, steps), flush=True)
        obs, info = env.reset()
print("CONTROL CLEAN: %d games %d steps" % (games, steps))
