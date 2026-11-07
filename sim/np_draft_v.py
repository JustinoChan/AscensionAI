import numpy as np


class NpDraftV:
    """Torch-free forward for the drafting value model. value() returns the
    denormalized predicted run value (floor + 1000*win scale)."""

    def __init__(self, path):
        d = np.load(path)
        n = int(d["nlayers"])
        self.W = [d["W%d" % i].astype(np.float32) for i in range(n)]
        self.b = [d["b%d" % i].astype(np.float32) for i in range(n)]
        self.vmean = float(d["vmean"])
        self.vstd = float(d["vstd"])

    def value(self, x):
        h = np.asarray(x, np.float32)
        for i in range(len(self.W) - 1):
            h = np.maximum(0.0, h @ self.W[i] + self.b[i])   # ReLU
        y = h @ self.W[-1] + self.b[-1]
        return float(y.reshape(-1)[0]) * self.vstd + self.vmean
