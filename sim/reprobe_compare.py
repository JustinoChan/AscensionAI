"""A/B learnability: train a fresh MLP to predict the gamma-0.97 return from
(a) the 1200-d obs and (b) the 1235-d enriched obs, on the SAME trajectories.
Reports best held-out R2 for each. Pass = 1235 clearly positive and well above
1200 (which reproduces the ~-0.5 baseline)."""
import glob
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
files = sorted(glob.glob("reprobe_dumps/b*.npz"))
assert len(files) >= 5
cut = int(len(files) * 0.8)


def load(fs, key):
    xs, ys = [], []
    for f in fs:
        z = np.load(f)
        xs.append(z[key].astype(np.float32)); ys.append(z["ret"])
    return np.concatenate(xs), np.concatenate(ys)


def best_val_r2(key, dim):
    Xtr, Ytr = load(files[:cut], key)
    Xva, Yva = load(files[cut:], key)
    Xt, Yt = torch.as_tensor(Xtr), torch.as_tensor(Ytr)
    Xv, Yv = torch.as_tensor(Xva), torch.as_tensor(Yva)
    net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(),
                        nn.Linear(512, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xt); best = -9.0
    for ep in range(1, 81):
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            j = perm[s:s + 512]
            loss = ((net(Xt[j]).squeeze(-1) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = net(Xv).squeeze(-1)
            best = max(best, float(1 - (Yv - p).var() / (Yv.var() + 1e-8)))
    return best


r1200 = best_val_r2("obs1200", 1200)
r1235 = best_val_r2("obs1235", 1235)
print("val R2 | 1200-d obs %.3f | 1235-d enriched %.3f | delta %.3f"
      % (r1200, r1235, r1235 - r1200))
print("VERDICT:", "PASS" if (r1235 >= 0.3 and r1235 - r1200 >= 0.2)
      else "INVESTIGATE")
