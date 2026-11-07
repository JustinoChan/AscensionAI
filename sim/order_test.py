"""Test the ABI/load-order hypothesis: import the slaythespire C++ extension and
the env BEFORE torch. If reset()+steps+torch ops now survive, the segfault was
import order and the trainer just needs the extension imported first.
"""
import numpy as np
import gym_sts_lightspeed                       # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire                              # noqa: F401  (ext loaded first)
import torch                                     # AFTER the extension

print("imports ok; torch", torch.__version__)

env = StsEnv(ascension=0)
obs, info = env.reset(seed=0)
print("reset ok, obs", np.asarray(obs).shape)

rng = np.random.default_rng(0)
net = torch.nn.Sequential(torch.nn.Linear(1200, 256), torch.nn.Tanh(), torch.nn.Linear(256, 128))
for k in range(30000):
    m = np.asarray(info["action_mask"]); legal = np.flatnonzero(m)
    a = int(rng.choice(legal)) if legal.size else 0
    obs, r, t, tr, info = env.step(a)
    if t or tr:
        obs, info = env.reset()
    # stress torch alongside the extension
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    _ = net(x)
print("SURVIVED 30000 steps with extension-imported-before-torch")
