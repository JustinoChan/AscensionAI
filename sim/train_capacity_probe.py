"""Offline capacity probe: train policies of increasing size to imitate the
rollout teacher (CE to teacher argmax), measure held-out top-1 agreement overall
and per tier. Decisive: does BOSS-state agreement rise with capacity, or plateau?
Torch only (no extension)."""
import numpy as np
import torch
import torch.nn as nn

d = np.load("teacher_data.npz")
obs = torch.tensor(np.asarray(d["obs"], np.float32))
act = torch.tensor(np.asarray(d["act"], np.int64))
mask = torch.tensor(np.asarray(d["mask"], np.float32))
room = np.asarray(d["room"])
N = len(obs)
rng = np.random.default_rng(0)
idx = rng.permutation(N)
ntr = int(0.85 * N)
tr, va = idx[:ntr], idx[ntr:]
rv = room[va]
print("N=%d train=%d val=%d" % (N, len(tr), len(va)), flush=True)


def make(sizes):
    L, d0 = [], 1235
    for h in sizes:
        L += [nn.Linear(d0, h), nn.GELU()]
        d0 = h
    L += [nn.Linear(d0, 128)]
    return nn.Sequential(*L)


for sizes in [(512, 256, 256), (1024, 512, 512), (2048, 1024, 512)]:
    torch.manual_seed(0)
    net = make(sizes)
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    for ep in range(20):
        perm = rng.permutation(tr)
        for b in range(0, len(perm), 512):
            j = perm[b:b + 512]
            lg = net(obs[j]) + (mask[j] - 1.0) * 1e9
            loss = nn.functional.cross_entropy(lg, act[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lg = net(obs[va]) + (mask[va] - 1.0) * 1e9
        pred = lg.argmax(1)
        acc = (pred == act[va]).float().mean().item()
        params = sum(p.numel() for p in net.parameters()) / 1e6
        line = "sizes=%-18s params=%.2fM  overall=%.3f" % (str(sizes), params, acc)
        for t in ("MONSTER", "ELITE", "BOSS"):
            m = torch.tensor(rv == t)
            if int(m.sum()) > 0:
                at = (pred[m] == act[va][m]).float().mean().item()
                line += "  %s=%.3f(n%d)" % (t, at, int(m.sum()))
        print(line, flush=True)
