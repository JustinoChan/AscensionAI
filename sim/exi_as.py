"""Afterstate features for the combat root prior (honest). afterstates(ev, n, det): for each of the first n legal actions,
play it once in the SAME freshly determinized copy of the current state (hidden draw order / RNG resampled with det, never
the true future) and return the visible obs1235 after it, [n, 1235]. The live state is restored exactly.
ASNet: score(a) = MLP([s, a_s - s]) over candidates; numpy forward for use at the root (HONEST_ROOT_AS)."""
import numpy as np


def afterstates(ev, n, det):
    snap = ev._hsnap()
    out = np.zeros((n, 1235), np.float32)
    for i in range(n):
        ev._hload(snap, det)
        try:
            ev.env.step(int(i))
            out[i] = ev._obs1235()
        except Exception:
            pass
    ev._hload(snap)
    return out


class ASNet:
    def __init__(self, path):
        z = np.load(path)
        self.L = [(z["l%d.weight" % i], z["l%d.bias" % i]) for i in range(3)]
        self.mu, self.sd = z["mu"], z["sd"]

    def scores(self, s, A):
        h = (np.concatenate([np.broadcast_to(s, A.shape), A - s], 1) - self.mu) / self.sd
        for li, (W, b) in enumerate(self.L):
            x = h @ W.T + b
            h = x if li == len(self.L) - 1 else 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
        return h[:, 0].astype(np.float64)
