"""Supervised learnability probe: can a FRESH net regress obs -> return?
Discriminates 'PPO training dynamics are broken' (this learns fine) from
'the target is unpredictable from a single obs' (this also fails).
argv: [1] dumps dir (default diag_mc). Val split = last 20% of files.
"""
import glob
import sys
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
DIR = sys.argv[1] if len(sys.argv) > 1 else "diag_mc"
files = sorted(glob.glob(DIR + "/b*.npz"))
assert len(files) >= 5, "need >=5 dump files, got %d" % len(files)
cut = int(len(files) * 0.8)


def load(fs):
    xs, ys = [], []
    for f in fs:
        z = np.load(f)
        xs.append(z["obs"].astype(np.float32))
        ys.append(z["ret"])
    return (torch.as_tensor(np.concatenate(xs)),
            torch.as_tensor(np.concatenate(ys)))


Xtr, Ytr = load(files[:cut])
Xva, Yva = load(files[cut:])
print("train %d  val %d | target sd train %.3f val %.3f" % (
    len(Xtr), len(Xva), float(Ytr.std()), float(Yva.std())), flush=True)

net = nn.Sequential(nn.Linear(1200, 512), nn.GELU(),
                    nn.Linear(512, 256), nn.GELU(),
                    nn.Linear(256, 256), nn.GELU(),
                    nn.Linear(256, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)


def r2(x, y):
    with torch.no_grad():
        p = net(x).squeeze(-1)
        return float(1 - (y - p).var() / (y.var() + 1e-8))


n = len(Xtr)
for ep in range(1, 201):
    perm = torch.randperm(n)
    for s in range(0, n, 512):
        j = perm[s:s + 512]
        loss = ((net(Xtr[j]).squeeze(-1) - Ytr[j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 20 == 0 or ep == 1:
        print("epoch %3d | R2 train %.3f | R2 val %.3f" % (
            ep, r2(Xtr, Ytr), r2(Xva, Yva)), flush=True)
print("SUPERVISED PROBE DONE")
