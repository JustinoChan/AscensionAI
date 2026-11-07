import sys
def log(m): sys.stderr.write(m+"\n"); sys.stderr.flush()
log("start")
import numpy as np; log("numpy")
import gym_sts_lightspeed
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire
log("imports ok")
env = StsEnv(ascension=0); log("env created")
o,i = env.reset(seed=0); log("reset ok %s" % (np.asarray(o).shape,))
log("DONE")
