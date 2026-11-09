"""Phase 3 step 3.5: combat policy iteration, round 1 (offline).

Train a new ROLLOUT policy (the default policy that plays out every honest K-sample rollout) to pick
what honest K=64 search picks: target = softmax(Q/TAU) over the searched candidates of each harvested
root decision (sim_logs/harvest/h1; Q = mean of the 64 sampled-world returns). varN layout
(1235-512-256-256-128), warm-started from varN, low lr so non-candidate logits stay close to varN.
Offline gate on held-out seeds: top-1 agreement with argmax-Q and regret@1 must both beat varN.
Writes nets/rollout1/rollout.npz (+ report.json, GATE.json).
  .venv_train/bin/python train_rollout.py
"""
import glob, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.expanduser("~/sts"))
OUT = "nets/rollout1"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
TAU = float(os.environ.get("TAU", "0.5"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
torch.manual_seed(0)
files = sorted(glob.glob("sim_logs/harvest/h1/seed_*.npz"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
nval = len(files) // 10


def load(fs):
    O, L, Q = [], [], []
    for f in fs:
        z = np.load(f)
        if len(z["c_chosen"]) == 0:
            continue
        O.append(z["c_obs"]); L.append(z["c_legal"]); Q.append(np.nanmean(z["c_q"], 2))
    return (torch.tensor(np.concatenate(O)).to(dev), torch.tensor(np.concatenate(L).astype(np.int64)).to(dev),
            torch.tensor(np.nan_to_num(np.concatenate(Q), nan=-1e9).astype(np.float32)).to(dev))


tr, va = load(files[:-nval]), load(files[-nval:])
print("decisions train %d val %d" % (len(tr[0]), len(va[0])), flush=True)
z = np.load("policy_varN.npz")


def net():
    g = lambda: nn.GELU(approximate="tanh")
    m = nn.Sequential(nn.Linear(1235, 512), g(), nn.Linear(512, 256), g(), nn.Linear(256, 256), g(), nn.Linear(256, 128))
    for i in (0, 2, 4):
        m[i].weight.data = torch.tensor(z["pbody.%d.weight" % i]); m[i].bias.data = torch.tensor(z["pbody.%d.bias" % i])
    m[6].weight.data = torch.tensor(z["pi.weight"]); m[6].bias.data = torch.tensor(z["pi.bias"])
    return m.to(dev)


def cand(lg, legal):
    ok = legal >= 0
    return torch.gather(lg, 1, legal.clamp(min=0)).masked_fill(~ok, -1e9), ok


def metrics(m, d):
    with torch.no_grad():
        lg = torch.cat([m(d[0][i:i + 8192]) for i in range(0, len(d[0]), 8192)])
    g, ok = cand(lg, d[1])
    sel = ok.sum(1) >= 2
    q = d[2]
    pick = g.argmax(1)
    best = q.max(1).values
    got = torch.gather(q, 1, pick[:, None])[:, 0]
    return {"top1_agree": float((pick == q.argmax(1))[sel].float().mean()), "regret@1": float((best - got)[sel].mean()),
            "n": int(sel.sum())}


base = net().eval()
rep = {"varN": metrics(base, va), "tau": TAU, "epochs": []}
print("varN", rep["varN"], flush=True)
m = net()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
best, best_state = None, None
for ep in range(EPOCHS):
    m.train()
    idx = torch.randperm(len(tr[0]), device=dev)
    for i in range(0, len(idx), 1024):
        b = idx[i:i + 1024]
        g, ok = cand(m(tr[0][b]), tr[1][b])
        tgt = F.softmax(tr[2][b] / TAU, 1) * ok
        tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-9)
        loss = -(tgt * F.log_softmax(g, 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    r = metrics(m, va); r["epoch"] = ep; r["loss"] = loss.item()
    rep["epochs"].append(r)
    print(r, flush=True)
    if best is None or r["regret@1"] < best["regret@1"]:
        best, best_state = r, {k: v.clone() for k, v in m.state_dict().items()}
m.load_state_dict(best_state)
sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
out = {"pbody.0.weight": sd["0.weight"], "pbody.0.bias": sd["0.bias"], "pbody.2.weight": sd["2.weight"], "pbody.2.bias": sd["2.bias"],
       "pbody.4.weight": sd["4.weight"], "pbody.4.bias": sd["4.bias"], "pi.weight": sd["6.weight"], "pi.bias": sd["6.bias"]}
for k in ("vbody.0.weight", "vbody.0.bias", "vbody.2.weight", "vbody.2.bias", "v.weight", "v.bias", "zmean", "zstd"):
    out[k] = z[k]
np.savez(OUT + "/rollout.npz", **out)
rep["best"] = best
gate = {"pass": best["top1_agree"] > rep["varN"]["top1_agree"] + 0.02 and best["regret@1"] < rep["varN"]["regret@1"],
        "why": "top1 %.3f vs varN %.3f (need +0.02); regret@1 %.3f vs varN %.3f (need lower)" % (
            best["top1_agree"], rep["varN"]["top1_agree"], best["regret@1"], rep["varN"]["regret@1"])}
rep["gate"] = gate
json.dump(rep, open(OUT + "/report.json", "w"), indent=1)
json.dump(gate, open(OUT + "/GATE.json", "w"), indent=1)
print("GATE", gate)
