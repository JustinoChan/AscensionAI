"""Afterstate root-prior learner (GPU). Usage: .venv_train/bin/python exi_train_as.py OUT_NPZ DATADIR [DATADIR ...]
Data from exi_collect.py with EXI_AFTER=1 (obs, aobs per legal action, honest K=64 Q). Model: score(a) = MLP([s, a_s - s])
(2470 -> 512 -> 256 -> 1, standardized inputs), softmax over each state's legal actions. Loss: cross-entropy to
softmax(Q / EXI_TAU) + EXI_REGRET_W * expected regret. 10% of seeds held out; reports agreement / regret vs the labels and,
for reference, exi1's (obs-only) agreement / regret on the same held-out states."""
import glob, json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

out, dirs = sys.argv[1], sys.argv[2:]
TAU = float(os.environ.get("EXI_TAU", "1.0")); RW = float(os.environ.get("EXI_REGRET_W", "0.2"))
EPOCHS = int(os.environ.get("EXI_EPOCHS", "20")); LR = float(os.environ.get("EXI_LR", "3e-4")); dev = "cuda"
# memory-lean load: keep obs/afterstates as float16, build the standardized candidate matrix in place (float16)
S, A, Q, NL, VAL = [], [], [], [], []
for f in sorted(p for d in dirs for p in glob.glob(d + "/seed_*.npz") if ".tmp." not in p):
    z = np.load(f)
    if "aobs" not in z.files:
        continue
    seed = int(f.rsplit("_", 1)[1].split(".")[0]); off = np.concatenate([[0], np.cumsum(z["n_legal"])])
    ob, ao, qq, nl = z["obs"], z["aobs"], z["q"], z["n_legal"]
    for i in range(len(nl)):
        n = int(nl[i])
        if n < 2:
            continue
        S.append(ob[i]); A.append(ao[off[i]:off[i] + n]); Q.append(np.nan_to_num(qq[i][:n], nan=-1e9)); NL.append(n)
        VAL.append((seed * 2654435761 % 1000003) % 10 == 0)
N = len(S); VAL = np.array(VAL); NL = np.array(NL)
print("states", N, "held-out", int(VAL.sum()), "candidates", int(NL.sum()), flush=True)
S = np.array(S, np.float16)
rowstart = np.concatenate([[0], np.cumsum(NL)]); R = int(rowstart[-1])
sid = np.repeat(np.arange(N), NL); pos = np.concatenate([np.arange(n) for n in NL])
X = np.empty((R, 2470), np.float16)
for i in range(N):
    a = A[i].astype(np.float32); s0 = S[i].astype(np.float32)
    X[rowstart[i]:rowstart[i + 1], :1235] = s0; X[rowstart[i]:rowstart[i + 1], 1235:] = a - s0
del A
rs = np.random.default_rng(0); tr_rows = np.where(~VAL[sid])[0]
smp = X[rs.choice(tr_rows, min(len(tr_rows), 200000), replace=False)].astype(np.float32)
mu = smp.mean(0); sd = smp.std(0) + 1e-3; del smp
for k in range(0, R, 100000):
    X[k:k + 100000] = ((X[k:k + 100000].astype(np.float32) - mu) / sd).astype(np.float16)
S = S.astype(np.float32)
Qp = np.full((N, 128), -1e9, np.float32)
for i in range(N):
    Qp[i, :NL[i]] = Q[i]
g = lambda: nn.GELU(approximate="tanh")
m = nn.Sequential(nn.Linear(2470, 512), g(), nn.Linear(512, 256), g(), nn.Linear(256, 1)).to(dev)
opt = torch.optim.Adam(m.parameters(), lr=LR); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def batch(idx):
    rows = np.concatenate([np.arange(rowstart[i], rowstart[i + 1]) for i in idx])
    xs = torch.tensor(X[rows].astype(np.float32), device=dev)
    bi = torch.tensor(np.repeat(np.arange(len(idx)), NL[idx]), device=dev); bp = torch.tensor(pos[rows], device=dev)
    q = torch.tensor(Qp[idx], device=dev)
    return xs, bi, bp, q


def logits(xs, bi, bp, nb):
    lg = torch.full((nb, 128), -1e9, device=dev)
    lg[bi, bp] = m(xs)[:, 0]
    return lg


def evaluate():
    m.eval(); ag = rg = 0.0; idx_all = np.where(VAL)[0]
    with torch.no_grad():
        for k in range(0, len(idx_all), 2048):
            idx = idx_all[k:k + 2048]; xs, bi, bp, q = batch(idx); lg = logits(xs, bi, bp, len(idx)); a = lg.argmax(1)
            ag += float((a == q.argmax(1)).float().sum()); rg += float((q.max(1).values - q.gather(1, a[:, None])[:, 0]).sum())
    m.train(); n = max(1, len(idx_all))
    return ag / n, rg / n


tr_idx = np.where(~VAL)[0]; best = (1e9, None)
for ep in range(EPOCHS):
    np.random.shuffle(tr_idx)
    for k in range(0, len(tr_idx), 512):
        idx = tr_idx[k:k + 512]; xs, bi, bp, q = batch(idx); lg = logits(xs, bi, bp, len(idx))
        tgt = torch.softmax(q / TAU, 1); loss = -(tgt * F.log_softmax(lg, 1)).sum(1).mean()
        if RW:
            loss = loss + RW * (torch.softmax(lg, 1) * (q.max(1, keepdim=True).values - q).clamp(max=100.0) * (q > -1e8)).sum(1).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    sched.step(); a, r = evaluate()
    print(json.dumps({"epoch": ep, "loss": float(loss), "agree": a, "regret": r}), flush=True)
    if r < best[0]:
        best = (r, {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()})
sd_ = best[1]
np.savez(out + ".tmp.npz", **{"l0.weight": sd_["0.weight"], "l0.bias": sd_["0.bias"], "l1.weight": sd_["2.weight"], "l1.bias": sd_["2.bias"],
                              "l2.weight": sd_["4.weight"], "l2.bias": sd_["4.bias"], "mu": mu, "sd": sd})
os.replace(out + ".tmp.npz", out)
# reference: exi1 (obs-only) on the same held-out states
z = np.load("nets/exi1/rollout.npz"); L = [(z["pbody.%d.weight" % i], z["pbody.%d.bias" % i]) for i in (0, 2, 4)] + [(z["pi.weight"], z["pi.bias"])]
h = S[VAL]
for li, (W, b) in enumerate(L):
    x = h @ W.T + b; h = x if li == len(L) - 1 else 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
lg = np.where(np.arange(128)[None] < NL[VAL][:, None], h, -1e9); a = lg.argmax(1); qv = Qp[VAL]
print(json.dumps({"best_regret": best[0], "exi1_agree": float((a == qv.argmax(1)).mean()),
                  "exi1_regret": float((qv.max(1) - qv[np.arange(len(a)), a]).mean()), "out": out}))
