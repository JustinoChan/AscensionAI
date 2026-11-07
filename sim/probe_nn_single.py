"""Pattern A: getNNInterface() called ONCE (inside StsEnv.__init__ only),
like the import-context tests. Print the deck region encoding at reset."""
import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv

env = StsEnv(ascension=0)
obs, info = env.reset(seed=4242)
c = np.asarray(obs[19:239])
nz = np.nonzero(c)[0]
print("A (single call) deck slots:", [(int(i), int(c[i])) for i in nz])
