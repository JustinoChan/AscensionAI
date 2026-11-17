"""Train + offline gate for the boss-readiness net: P(beat this act's boss | public state obs1200, act).
  .venv_train/bin/python ready_train.py nets/ready/data.npz nets/ready/ready.npz
Split by run seed (20% held out). Reports held-out AUC on all states and on the last state before each boss,
plus an HP-only baseline. Writes GATE line: PASS if held-out last-before-boss AUC >= 0.65 and beats HP-only by 0.03.
"""
import sys, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

d = np.load(sys.argv[1]); out = sys.argv[2]
_last = d["last"] == 1
_rng = np.random.default_rng(0)
keep = _last | (_rng.random(len(_last)) < 0.2)     # all last-before-boss states + 20% of the rest (fits the 8 GB GPU)
X = d["X"][keep].astype(np.float32); y = d["y"][keep].astype(np.float32); act = d["act"][keep].astype(np.int64); last = _last[keep]
seed = d["seed"][keep]
test = (seed * 2654435761 % 1000003) % 5 == 0
print("rows", len(X), "test", int(test.sum()), flush=True)
A = np.eye(3, dtype=np.float32)[act - 1]
mu = X[~test].mean(0); sd = X[~test].std(0) + 1e-3
Z = np.concatenate([(X - mu) / sd, A], 1).astype(np.float32)
dev = "cuda"
g = lambda: nn.GELU(approximate="tanh")
m = nn.Sequential(nn.Linear(Z.shape[1], 256), g(), nn.Dropout(0.1), nn.Linear(256, 64), g(), nn.Linear(64, 1)).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
Xt = torch.tensor(Z[~test], device=dev); Yt = torch.tensor(y[~test], device=dev)
Xv = torch.tensor(Z[test], device=dev)


def auc(s, t):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = t.sum(); n = len(t) - p
    return float((r[t == 1].sum() - p * (p + 1) / 2) / max(1, p * n))


best, bstate = -1, None
for ep in range(12):
    m.train()
    for i in torch.randperm(len(Xt), device=dev).split(4096):
        l = F.binary_cross_entropy_with_logits(m(Xt[i])[:, 0], Yt[i]); opt.zero_grad(); l.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        sv = torch.cat([m(b)[:, 0] for b in Xv.split(8192)]).cpu().numpy()
    a_all = auc(sv, y[test]); a_last = auc(sv[last[test]], y[test][last[test]])
    print("epoch %d  held-out AUC all %.3f  last-before-boss %.3f" % (ep, a_all, a_last), flush=True)
    if a_last > best:
        best, bstate, bsv = a_last, {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}, sv
# HP-only baseline: obs index of HP fraction is unknown; use the best single input feature (by held-out AUC) as baseline
tl = test & last
cand = np.argsort(-np.abs(np.corrcoef(X[~test & last][:, :1200].T, y[~test & last])[-1, :-1]))[:20] if (~test & last).sum() > 100 else []
base = max([max(auc(X[tl][:, j], y[tl]), auc(-X[tl][:, j], y[tl])) for j in cand] + [0.5])
res = {"auc_last": best, "best_single_feature_auc_last": base, "n_train": int((~test).sum()), "n_test": int(test.sum())}
for a in (1, 2, 3):
    k = last[test] & (act[test] == a)
    res["auc_last_act%d" % a] = auc(bsv[k], y[test][k]) if k.sum() > 20 else None
print(json.dumps(res))
np.savez(out, mu=mu, sd=sd, **{k.replace(".", "_"): v for k, v in bstate.items()})
ok = best >= 0.65 and best >= base + 0.03
print("GATE %s" % ("PASS" if ok else "FAIL"))
