"""Horizon sweep: for each gamma target, report (a) ridge-regression val R2
(overfit-proof linear floor) and (b) MLP best-val R2 with early-stop tracking.
Val split = last 20% of files (different episodes, same frozen policy)."""
import glob
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
files = sorted(glob.glob("diag_mc2/b*.npz"))
assert len(files) >= 5
cut = int(len(files) * 0.8)
KEYS = ["ret99", "ret97", "ret95", "ret90", "ret80"]


def load(fs, key):
    xs, ys = [], []
    for f in fs:
        z = np.load(f)
        xs.append(z["obs"].astype(np.float32))
        ys.append(z[key])
    return np.concatenate(xs), np.concatenate(ys)


def ridge_val_r2(Xtr, Ytr, Xva, Yva, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = (Xtr - mu) / sd
    Xv = (Xva - mu) / sd
    A = Xt.T @ Xt + lam * np.eye(Xt.shape[1], dtype=np.float32)
    b = Xt.T @ (Ytr - Ytr.mean())
    w = np.linalg.solve(A, b)
    pred = Xv @ w + Ytr.mean()
    return float(1 - np.var(Yva - pred) / (np.var(Yva) + 1e-8))


def mlp_best_val_r2(Xtr, Ytr, Xva, Yva, epochs=60):
    Xt, Yt = torch.as_tensor(Xtr), torch.as_tensor(Ytr)
    Xv, Yv = torch.as_tensor(Xva), torch.as_tensor(Yva)
    net = nn.Sequential(nn.Linear(1200, 512), nn.GELU(),
                        nn.Linear(512, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(),
                        nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xt)
    best = -9.0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            j = perm[s:s + 512]
            loss = ((net(Xt[j]).squeeze(-1) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 2 == 0:
            with torch.no_grad():
                p = net(Xv).squeeze(-1)
                r2 = float(1 - (Yv - p).var() / (Yv.var() + 1e-8))
            best = max(best, r2)
    return best


for key in KEYS:
    Xtr, Ytr = load(files[:cut], key)
    Xva, Yva = load(files[cut:], key)
    rr = ridge_val_r2(Xtr, Ytr, Xva, Yva)
    mm = mlp_best_val_r2(Xtr, Ytr, Xva, Yva)
    print("%s | target sd %6.3f | ridge val R2 %6.3f | MLP best-val R2 %6.3f"
          % (key, float(Ytr.std()), rr, mm), flush=True)
print("SWEEP DONE")
