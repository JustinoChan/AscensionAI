"""Fit a fresh head to the sampling-policy pure-fight dumps; report best val R2
on both the 1235-d (enriched) and 1200-d obs (combat features are the last 35)."""
import glob
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
files = sorted(glob.glob("samp_dumps/b*.npz"))
assert len(files) >= 5
X = np.concatenate([np.load(f)["obs"].astype(np.float32) for f in files])
Y = np.concatenate([np.load(f)["ret"] for f in files])
print("samples %d | return sd %.3f" % (len(Y), Y.std()), flush=True)
n = len(Y); cut = int(n * 0.8)
perm = np.random.RandomState(0).permutation(n)
tr, va = perm[:cut], perm[cut:]


def best_r2(cols, dim):
    Xt = torch.as_tensor(X[tr][:, cols]); Yt = torch.as_tensor(Y[tr])
    Xv = torch.as_tensor(X[va][:, cols]); Yv = torch.as_tensor(Y[va])
    net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(),
                        nn.Linear(512, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    m = len(Xt); best = -9.0
    for ep in range(60):
        pp = torch.randperm(m)
        for s in range(0, m, 512):
            j = pp[s:s + 512]
            loss = ((net(Xt[j]).squeeze(-1) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            best = max(best, float(1 - (Yv - net(Xv).squeeze(-1)).var() / (Yv.var() + 1e-8)))
    return best


r1235 = best_r2(np.arange(1235), 1235)
r1200 = best_r2(np.arange(1200), 1200)
print("SAMPLING pure-fight | val R2 1200-d %.3f | 1235-d %.3f" % (r1200, r1235))
print("(greedy baseline was 0.984)")
