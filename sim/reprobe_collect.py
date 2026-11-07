"""Drive a frozen varE policy (acts on the OLD 1200-d obs) and record, per
step, BOTH the 1200-d and 1235-d normalized observations plus the gamma-0.97
Monte-Carlo return. Single process, torch-free. The A/B test downstream asks:
does adding the 35 combat features make the return learnable?"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire
from combat_features import combat_block

GAMMA = 0.97
RS = 0.1
TARGET_EPISODES = 240
OUT = "reprobe_dumps"
os.makedirs(OUT, exist_ok=True)


class SafeStsEnv(StsEnv):
    _last = None

    def _get_observation(self):
        try:
            o = super()._get_observation(); self._last = o; return o
        except Exception:
            return self._last if self._last is not None else np.zeros(1200, np.int32)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpPolicy:
    def __init__(self, p):
        w = np.load(p)
        self.layers = [(w["body.%d.weight" % i], w["body.%d.bias" % i])
                       for i in (0, 2, 4)]
        self.pi = (w["pi.weight"], w["pi.bias"])

    def act(self, x, mask):
        h = x
        for W, b in self.layers:
            h = gelu(h @ W.T + b)
        logits = h @ self.pi[0].T + self.pi[1]
        return int(np.argmax(np.where(mask > 0, logits, -1e9)))


pol = NpPolicy("policy_varE.npz")
env = SafeStsEnv(ascension=0)
maxima = np.maximum(np.asarray(env.nn.getObservationMaximums(), np.float32), 1.0)

done_eps = fidx = 0
acc1200, acc1235, accr = [], [], []
ep1200, ep1235, epr = [], [], []
seed_i = 777001
obs, info = env.reset(seed=seed_i)
while done_eps < TARGET_EPISODES:
    o1200 = np.asarray(obs, np.float32) / maxima
    o1235 = np.concatenate([o1200, combat_block(env)])
    a = pol.act(o1200, np.asarray(info["action_mask"]))
    ep1200.append(o1200.astype(np.float16))
    ep1235.append(o1235.astype(np.float16))
    obs, r, term, trunc, info = env.step(a)
    epr.append(float(r) * RS)
    if term or trunc:
        G = 0.0
        gs = [0.0] * len(epr)
        for t in reversed(range(len(epr))):
            G = epr[t] + GAMMA * G
            gs[t] = G
        acc1200.append(np.asarray(ep1200, np.float16)[::3])
        acc1235.append(np.asarray(ep1235, np.float16)[::3])
        accr.append(np.asarray(gs, np.float32)[::3])
        ep1200, ep1235, epr = [], [], []
        done_eps += 1
        if len(acc1200) >= 12:
            fidx += 1
            np.savez_compressed(os.path.join(OUT, "b%04d.npz" % fidx),
                                obs1200=np.concatenate(acc1200),
                                obs1235=np.concatenate(acc1235),
                                ret=np.concatenate(accr))
            acc1200, acc1235, accr = [], [], []
        seed_i += 1
        obs, info = env.reset(seed=seed_i)
print("collected %d episodes, %d files" % (done_eps, fidx))
