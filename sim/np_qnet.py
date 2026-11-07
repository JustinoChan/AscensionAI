import numpy as np


class NpQNet:
    """Torch-free forward for the combat student: obs(1235) -> Q(128).
    Scores are only used for argmax over candidate actions."""

    def __init__(self, path):
        d = np.load(path)
        n = int(d["nlayers"])
        self.W = [d["W%d" % i].astype(np.float32) for i in range(n)]
        self.b = [d["b%d" % i].astype(np.float32) for i in range(n)]

    def q(self, x):
        h = np.asarray(x, np.float32)
        for i in range(len(self.W) - 1):
            h = np.maximum(0.0, h @ self.W[i] + self.b[i])
        return h @ self.W[-1] + self.b[-1]
