"""Combat expert iteration, learner (GPU). Usage:
  .venv_train/bin/python exi_train.py INIT_NPZ OUT_NPZ DATADIR [DATADIR ...]    (env: EXI_TAU, EXI_EPOCHS, EXI_LR)
Trains the play-out net (varN layout, warm-started from INIT_NPZ) on the honest-search labels: cross-entropy to the soft
target softmax(Q / EXI_TAU) over the legal actions of each visited state. Held out: 10% of seeds. Reports, for the init
net and the trained net, top-1 agreement with the search's best action and mean regret (Q_best - Q_chosen, in score
points) on held-out states. Writes OUT_NPZ (best held-out regret), varN layout, so HONEST_ROLLOUT_NET can use it.
"""
import glob, json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

init, out, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
RW = float(os.environ.get("EXI_REGRET_W", "0"))   # weight of the expected-regret term (0 = off, plain soft-target CE)
TAU = float(os.environ.get("EXI_TAU", "2.0")); EPOCHS = int(os.environ.get("EXI_EPOCHS", "12")); LR = float(os.environ.get("EXI_LR", "1e-4"))
dev = "cuda"
files = sorted(p for d in dirs for p in glob.glob(d + "/seed_*.npz") if ".tmp." not in p)
O, Q, NL, VAL = [], [], [], []
for f in files:
    z = np.load(f)
    keep = z["n_legal"] >= 2
    if not keep.any():
        continue
    s = int(f.rsplit("_", 1)[1].split(".")[0])
    O.append(z["obs"][keep]); Q.append(z["q"][keep]); NL.append(z["n_legal"][keep])
    VAL.append(np.full(int(keep.sum()), (s * 2654435761 % 1000003) % 10 == 0))
O = np.concatenate(O).astype(np.float32); Q = np.concatenate(Q); NL = np.concatenate(NL).astype(np.int64); VAL = np.concatenate(VAL)
print("states", len(O), "held-out", int(VAL.sum()), "files", len(files), flush=True)
legal = np.arange(128)[None, :] < NL[:, None]
Qf = np.where(legal, np.nan_to_num(Q, nan=-1e9), -1e9).astype(np.float32)


WID = [int(x) for x in os.environ.get("EXI_WIDTH", "512,256,256").split(",")]   # hidden widths (3 layers)


def net():
    g = lambda: nn.GELU(approximate="tanh")
    return nn.Sequential(nn.Linear(1235, WID[0]), g(), nn.Linear(WID[0], WID[1]), g(), nn.Linear(WID[1], WID[2]), g(), nn.Linear(WID[2], 128))


def load(m, p):
    z = np.load(p)
    for i in (0, 2, 4):
        m[i].weight.data = torch.tensor(z["pbody.%d.weight" % i]); m[i].bias.data = torch.tensor(z["pbody.%d.bias" % i])
    m[6].weight.data = torch.tensor(z["pi.weight"]); m[6].bias.data = torch.tensor(z["pi.bias"])
    return m


def save(m, p):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    z = np.load("policy_varN.npz")
    d = {"pbody.0.weight": sd["0.weight"], "pbody.0.bias": sd["0.bias"], "pbody.2.weight": sd["2.weight"], "pbody.2.bias": sd["2.bias"],
         "pbody.4.weight": sd["4.weight"], "pbody.4.bias": sd["4.bias"], "pi.weight": sd["6.weight"], "pi.bias": sd["6.bias"]}
    for k in ("vbody.0.weight", "vbody.0.bias", "vbody.2.weight", "vbody.2.bias", "v.weight", "v.bias", "zmean", "zstd"):
        d[k] = z[k]
    np.savez(p + ".tmp.npz", **d); os.replace(p + ".tmp.npz", p)


m = (load(net(), init) if WID == [512, 256, 256] and os.environ.get("EXI_FRESH", "0") != "1" else net()).to(dev)   # wider net: fresh init (INIT only for the baseline row)
Xt = torch.tensor(O, device=dev); Qt = torch.tensor(Qf, device=dev); Lt = torch.tensor(legal, device=dev)
tr = torch.tensor(np.where(~VAL)[0], device=dev); va = torch.tensor(np.where(VAL)[0], device=dev)
target = torch.softmax(Qt / TAU, 1)


def evaluate():
    m.eval()
    with torch.no_grad():
        ag, rg = [], []
        for b in va.split(8192):
            lg = m(Xt[b]).masked_fill(~Lt[b], -1e9); a = lg.argmax(1)
            best = Qt[b].max(1).values; ch = Qt[b].gather(1, a[:, None])[:, 0]
            ag.append((a == Qt[b].argmax(1)).float()); rg.append(best - ch)
    m.train()
    return float(torch.cat(ag).mean()), float(torch.cat(rg).mean())


a0, r0 = evaluate()
print(json.dumps({"epoch": "init", "agree": a0, "regret": r0}), flush=True)
opt = torch.optim.Adam(m.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
best = (r0, None)
for ep in range(EPOCHS):
    for b in tr[torch.randperm(len(tr), device=dev)].split(1024):
        lg = m(Xt[b]).masked_fill(~Lt[b], -1e9)
        loss = -(target[b] * F.log_softmax(lg, 1)).sum(1).mean()
        if RW:                                   # value-aware: + RW * expected regret of the net's policy (score points)
            reg = (Qt[b].max(1, keepdim=True).values - Qt[b]).clamp(max=100.0) * Lt[b]
            loss = loss + RW * (torch.softmax(lg, 1) * reg).sum(1).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    sched.step()
    a, r = evaluate()
    print(json.dumps({"epoch": ep, "loss": float(loss), "agree": a, "regret": r}), flush=True)
    if r < best[0]:
        best = (r, {k: v.clone() for k, v in m.state_dict().items()})
if best[1] is not None:
    m.load_state_dict(best[1])
save(m, out)
print(json.dumps({"init_regret": r0, "best_regret": best[0], "init_agree": a0, "out": out, "improved": best[1] is not None}))
