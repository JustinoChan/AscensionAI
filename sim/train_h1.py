"""Phase 3 step 1b: train the candidate prior and the combat value net on the
honest harvest (HARVEST=<dir> npz files from mainline_h1, K=64).

  prior  P(s)[a]: same MLP as policy_varN (1235-512-256-256-128, GELU-tanh),
         warm-started from varN, trained with a within-decision listwise ranking
         loss: KL( softmax(Qmean/tau) over the searched candidates || softmax(logits) ).
  value  V(s): same trunk, 3-output head:
         [0] greedy-rollout return from s (target = Qmean of the varN-greedy
             candidate, i.e. exactly what a K=64 rollout of the default policy
             scores, /VSCALE) -> the leaf value for depth-cut rollouts
         [1] HP lost for the rest of the fight / max HP
         [2] fight-win logit
Split by seed (last VAL_FRAC of seeds = held-out fights). Writes
  <out>/prior_h1.npz, <out>/value_h1.npz  (NpNet / C++ CombatNet layout)
  <out>/train_report.json

Run with the training venv (torch must never load in the sim process):
  .venv_train/bin/python train_h1.py sim_logs/harvest/h1 nets/h1
"""
import glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HV, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(os.environ.get("EPOCHS", "40"))
TAU = float(os.environ.get("TAU", "2.0"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.1"))
VSCALE = 50.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
os.makedirs(OUT, exist_ok=True)

# ---------------- data ----------------
files = sorted(glob.glob(HV + "/seed_*.npz"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
nval = max(1, int(len(files) * VAL_FRAC))
split = {"train": files[:-nval], "val": files[-nval:]}


def load(fs):
    O, L, Q, HPL, WIN, G = [], [], [], [], [], []
    for f in fs:
        z = np.load(f)
        if len(z["c_chosen"]) == 0:
            continue
        q = np.nanmean(z["c_q"], 2)                       # (N,T) mean over K samples
        fi = z["c_fight"]
        hp_out = z["f_hp_out"][fi].astype(np.float32)
        mhp = np.maximum(1, z["f_mhp"][fi]).astype(np.float32)
        O.append(z["c_obs"]); L.append(z["c_legal"]); Q.append(q)
        HPL.append(np.clip(z["c_hp"] - hp_out, 0, None) / mhp)
        WIN.append(z["f_won"][fi].astype(np.float32))
        G.append(np.full(len(fi), int(z["run"][0])))
    return dict(obs=np.concatenate(O), legal=np.concatenate(L).astype(np.int64), q=np.concatenate(Q).astype(np.float32),
                hpl=np.concatenate(HPL).astype(np.float32), win=np.concatenate(WIN), seed=np.concatenate(G))


data = {k: load(v) for k, v in split.items()}
for k, d in data.items():
    print(k, "seeds", len(split[k]), "decisions", len(d["obs"]), flush=True)


def mlp(n_out):
    return nn.Sequential(nn.Linear(1235, 512), nn.GELU(approximate="tanh"), nn.Linear(512, 256), nn.GELU(approximate="tanh"),
                         nn.Linear(256, 256), nn.GELU(approximate="tanh"), nn.Linear(256, n_out))


varN = np.load("policy_varN.npz")


def warm(m, head=True):
    for i, li in zip((0, 2, 4), (0, 2, 4)):
        m[li].weight.data = torch.tensor(varN["pbody.%d.weight" % i]); m[li].bias.data = torch.tensor(varN["pbody.%d.bias" % i])
    if head:
        m[6].weight.data = torch.tensor(varN["pi.weight"]); m[6].bias.data = torch.tensor(varN["pi.bias"])
    return m


def T(d):
    return {k: torch.tensor(v).to(dev) for k, v in d.items() if k != "seed"}


tr, va = T(data["train"]), T(data["val"])


def cand_logits(lg, legal):
    """gather logits at candidate slots; padded slots -> -inf"""
    ok = legal >= 0
    g = torch.gather(lg, 1, legal.clamp(min=0))
    return g.masked_fill(~ok, -1e9), ok


def greedy_slot(lg, legal):
    g, _ = cand_logits(lg, legal)
    return g.argmax(1)


# varN logits (fixed) -> greedy candidate per decision (= rollout default policy's choice)
base = warm(mlp(128)).to(dev).eval()
with torch.no_grad():
    for d in (tr, va):
        lg = torch.cat([base(d["obs"][i:i + 8192]) for i in range(0, len(d["obs"]), 8192)])
        d["base_lg"] = lg
        d["gslot"] = greedy_slot(lg, d["legal"])
        d["qn"] = torch.nan_to_num(d["q"], nan=-1e9)
        d["vy"] = torch.gather(d["qn"], 1, d["gslot"][:, None])[:, 0] / VSCALE


def rank_metrics(lg, d):
    g, ok = cand_logits(lg, d["legal"])
    qn = d["qn"]
    nc = ok.sum(1)
    sel = nc >= 2
    best = qn.max(1).values
    order = g.argsort(1, descending=True)
    qs = torch.gather(qn, 1, order)
    out = {"decisions_multi": int(sel.sum())}
    out["top1_agree"] = float((order[:, 0] == qn.argmax(1))[sel].float().mean())
    for m in (1, 2, 3, 4):
        reg = best - qs[:, :m].max(1).values
        out["regret@%d" % m] = float(reg[sel].mean())
        out["recall@%d" % m] = float((reg[sel] <= 1e-6).float().mean())
    return out


def batches(n, bs, shuf=True):
    idx = torch.randperm(n, device=dev) if shuf else torch.arange(n, device=dev)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


HIST = {"prior": [], "value": []}
report = {"files": {k: len(v) for k, v in split.items()}, "decisions": {k: len(data[k]["obs"]) for k in data},
          "tau": TAU, "epochs": EPOCHS}
with torch.no_grad():
    report["varN_val"] = rank_metrics(va["base_lg"], va)
print("varN baseline val", report["varN_val"], flush=True)

# ---------------- prior ----------------
prior = warm(mlp(128)).to(dev)
opt = torch.optim.AdamW(prior.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
best_state, best_reg = None, 1e9
for ep in range(EPOCHS):
    prior.train(); t0 = time.time()
    for b in batches(len(tr["obs"]), 1024):
        g, ok = cand_logits(prior(tr["obs"][b]), tr["legal"][b])
        tgt = F.softmax(tr["qn"][b] / TAU, 1) * ok
        tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-9)
        loss = -(tgt * F.log_softmax(g, 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    prior.eval()
    with torch.no_grad():
        m = rank_metrics(torch.cat([prior(va["obs"][i:i + 8192]) for i in range(0, len(va["obs"]), 8192)]), va)
    print("prior ep %d loss %.4f  val %s  (%.0fs)" % (ep, loss.item(), m, time.time() - t0), flush=True)
    HIST["prior"].append({"epoch": ep, "loss": loss.item(), **m})
    if m["regret@2"] < best_reg:
        best_reg, best_state, report["prior_val"], report["prior_epoch"] = m["regret@2"], {k: v.clone() for k, v in prior.state_dict().items()}, m, ep
prior.load_state_dict(best_state)

# ---------------- value ----------------
val = warm(mlp(3), head=False).to(dev)
opt = torch.optim.AdamW(val.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def vpred(d):
    return torch.cat([val(d["obs"][i:i + 8192]) for i in range(0, len(d["obs"]), 8192)])


def vmetrics(p, d):
    y = d["vy"] * VSCALE; yh = p[:, 0] * VSCALE
    r2 = 1 - ((y - yh) ** 2).mean() / ((y - y.mean()) ** 2).mean()
    pw = torch.sigmoid(p[:, 2]); w = d["win"]
    bins = []
    for lo in np.linspace(0, 1, 11)[:-1]:
        s = (pw >= lo) & (pw < lo + 0.1)
        if s.sum() > 0:
            bins.append([round(float(lo), 1), int(s.sum()), round(float(pw[s].mean()), 3), round(float(w[s].mean()), 3)])
    ece = sum(b[1] * abs(b[2] - b[3]) for b in bins) / max(1, len(w))
    return {"V_r2": float(r2), "V_mae": float((y - yh).abs().mean()), "V_mae_const": float((y - y.mean()).abs().mean()),
            "V_corr": float(torch.corrcoef(torch.stack([y, yh]))[0, 1]),
            "hpl_mae": float((p[:, 1] - d["hpl"]).abs().mean()), "hpl_mae_const": float((d["hpl"] - d["hpl"].mean()).abs().mean()),
            "win_brier": float(((pw - w) ** 2).mean()), "win_brier_const": float(((w.mean() - w) ** 2).mean()),
            "win_ece": float(ece), "win_bins[lo,n,pred,obs]": bins}


best_state, best_mae = None, 1e9
for ep in range(EPOCHS):
    val.train(); t0 = time.time()
    for b in batches(len(tr["obs"]), 1024):
        p = val(tr["obs"][b])
        loss = (F.smooth_l1_loss(p[:, 0], tr["vy"][b], beta=0.1) + F.smooth_l1_loss(p[:, 1], tr["hpl"][b], beta=0.05)
                + 0.2 * F.binary_cross_entropy_with_logits(p[:, 2], tr["win"][b]))
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    val.eval()
    with torch.no_grad():
        m = vmetrics(vpred(va), va)
    print("value ep %d loss %.4f  val V_r2 %.3f V_mae %.2f (const %.2f) hpl_mae %.3f brier %.4f (const %.4f) ece %.3f (%.0fs)" % (
        ep, loss.item(), m["V_r2"], m["V_mae"], m["V_mae_const"], m["hpl_mae"], m["win_brier"], m["win_brier_const"], m["win_ece"],
        time.time() - t0), flush=True)
    HIST["value"].append({"epoch": ep, "loss": loss.item(), **{k: v for k, v in m.items() if not k.startswith("win_bins")}})
    if m["V_mae"] < best_mae:
        best_mae, best_state, report["value_val"], report["value_epoch"] = m["V_mae"], {k: v.clone() for k, v in val.state_dict().items()}, m, ep
val.load_state_dict(best_state)


def save(m, path, extra_head=None):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    d = {"pbody.0.weight": sd["0.weight"], "pbody.0.bias": sd["0.bias"], "pbody.2.weight": sd["2.weight"], "pbody.2.bias": sd["2.bias"],
         "pbody.4.weight": sd["4.weight"], "pbody.4.bias": sd["4.bias"], "pi.weight": sd["6.weight"], "pi.bias": sd["6.bias"]}
    for k in ("vbody.0.weight", "vbody.0.bias", "vbody.2.weight", "vbody.2.bias", "v.weight", "v.bias", "zmean", "zstd"):
        d[k] = varN[k]
    np.savez(path + ".tmp.npz", **d)
    os.replace(path + ".tmp.npz", path)


save(prior, os.path.join(OUT, "prior_h1.npz"))
save(val, os.path.join(OUT, "value_h1.npz"))
report["vscale"] = VSCALE
json.dump(report, open(os.path.join(OUT, "train_report.json"), "w"), indent=1)
json.dump(HIST, open(os.path.join(OUT, "train_history.json"), "w"))
print(json.dumps({k: report[k] for k in ("varN_val", "prior_val", "value_val")}, indent=1))
