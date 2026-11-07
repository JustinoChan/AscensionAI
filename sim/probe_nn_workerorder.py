"""Pattern D: the production worker's EXACT call sequence, in-process:
import gym first, then slaythespire, then getNNInterface()+maximums at module
level (call #1), then StsEnv.__init__ (call #2). If the deck region collapses
here, the corruption is the call sequence, not the pipe I/O context.
"""
import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire

maxima = np.maximum(
    np.asarray(slaythespire.getNNInterface().getObservationMaximums(),
               dtype=np.float32), 1.0)

env = StsEnv(ascension=0)
obs, info = env.reset(seed=4242)
c = np.asarray(obs[19:239])
nz = np.nonzero(c)[0]
print("D (worker order) deck slots:", [(int(i), int(c[i])) for i in nz])
