"""Pattern A': the fix candidate. Env constructed FIRST (its __init__ makes the
only getNNInterface call), maxima taken from env.nn afterwards. Expect correct
deck slots {6:1, 38:4, 132:5} and maxima 7 for deck slots."""
import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv

env = StsEnv(ascension=0)
maxima = np.maximum(
    np.asarray(env.nn.getObservationMaximums(), dtype=np.float32), 1.0)
obs, info = env.reset(seed=4242)
c = np.asarray(obs[19:239])
nz = np.nonzero(c)[0]
print("A' deck slots:", [(int(i), int(c[i])) for i in nz],
      "| maxima[19:21]:", maxima[19], maxima[20])
