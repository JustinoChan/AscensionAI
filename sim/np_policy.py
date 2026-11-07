"""Torch-free NumPy copy of the separate-trunk ActorCritic, for in-worker
rollouts/eval. Reads the keys saved by train_sim_mp.save_ckpt-style npz:
pbody.{0,2,4}.{weight,bias}, vbody.{0,2}.{weight,bias}, pi.{weight,bias},
v.{weight,bias}. Reloads when the npz mtime changes (atomic writes from trainer)."""
import os
import numpy as np

NEG = -1e9


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpNet:
    def __init__(self, npz_path):
        self.path = npz_path
        self._mtime = None
        self.reload(force=True)

    def reload(self, force=False):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return False
        if not force and m == self._mtime:
            return False
        w = np.load(self.path)
        self.pbody = [(w["pbody.%d.weight" % i], w["pbody.%d.bias" % i])
                      for i in (0, 2, 4)]
        self.vbody = [(w["vbody.%d.weight" % i], w["vbody.%d.bias" % i])
                      for i in (0, 2)]
        self.pi = (w["pi.weight"], w["pi.bias"])
        self.vh = (w["v.weight"], w["v.bias"])
        self.zmean = float(w["zmean"]) if "zmean" in w.files else 0.0
        self.zstd = float(w["zstd"]) if "zstd" in w.files else 1.0
        self._mtime = m
        return True

    def logits(self, obs, mask):
        h = obs
        for W, b in self.pbody:
            h = _gelu(h @ W.T + b)
        lg = h @ self.pi[0].T + self.pi[1]
        return np.where(np.asarray(mask) > 0, lg, NEG)

    def greedy(self, obs, mask):
        return int(np.argmax(self.logits(obs, mask)))

    def value(self, obs):
        h = obs
        for W, b in self.vbody:
            h = _gelu(h @ W.T + b)
        return float(h @ self.vh[0].T + self.vh[1])

    def value_denorm(self, obs):
        return self.value(obs) * self.zstd + self.zmean
