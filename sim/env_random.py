"""Isolate the segfault: drive the fork's StsEnv with a plain random masked agent
(no torch). If THIS segfaults -> it's the env/obs/sim path. If it survives ->
the crash is torch-related. Prints progress so we see where it dies.
"""
import sys
import numpy as np
from gym_sts_lightspeed.envs.sts_env import StsEnv

OBS = 1200


class SafeStsEnv(StsEnv):
    obs_fail = 0
    _last_obs = None

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last_obs = o
            return o
        except Exception:
            SafeStsEnv.obs_fail += 1
            return self._last_obs if self._last_obs is not None else np.zeros(OBS, np.int32)


N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
env = SafeStsEnv(ascension=0)
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
games = 0
for i in range(N):
    m = np.asarray(info["action_mask"])
    legal = np.flatnonzero(m)
    a = int(rng.choice(legal)) if legal.size else 0
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        games += 1
        obs, info = env.reset()
    if i % 2000 == 0:
        print("step %6d games %4d floor %s obs_fail %d" % (
            i, games, info.get("floor"), SafeStsEnv.obs_fail), flush=True)
print("SURVIVED %d steps, %d games, obs_fail %d" % (N, games, SafeStsEnv.obs_fail))
