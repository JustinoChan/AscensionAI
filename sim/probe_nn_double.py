"""Pattern B: getNNInterface() called at module level FIRST (the worker
pattern, used by ALL sim workers since the baseline run), then StsEnv.__init__
calls it again. Print the deck region encoding at reset."""
import numpy as np
import slaythespire

nn0 = slaythespire.getNNInterface()
mx = np.asarray(nn0.getObservationMaximums())

import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv

env = StsEnv(ascension=0)
obs, info = env.reset(seed=4242)
c = np.asarray(obs[19:239])
nz = np.nonzero(c)[0]
print("B (double call) deck slots:", [(int(i), int(c[i])) for i in nz])

# also: does encoding via the FIRST instance work?
obs2 = np.asarray(nn0.getObservation(env.gc, None))
c2 = obs2[19:239]
nz2 = np.nonzero(c2)[0]
print("B (first instance directly):", [(int(i), int(c2[i])) for i in nz2])
