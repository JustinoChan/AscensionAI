"""Rank-trained drafting value model v2: seed-tagged data, leakage-free
SEED-level split, win-flip pair weighting, early stop on held-out
edge-captured. Pre-registered deploy gate: edge_captured >= 0.5.
argv: [1] jsonl (default draft_data_v2.jsonl)  [2] FLIP_W (default 5.0)"""
import json, sys, numpy as np, torch, torch.nn as nn
from run_state_features import FEAT_DIM
from np_draft_v import NpDraftV

PATH = sys.argv[1] if len(sys.argv) > 1 else "draft_data_v2.jsonl"
FLIP_W = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
EPOCHS, PATIENCE = 1500, 6

rows = [json.loads(l) for l in open(PATH)]
F = np.asarray([r["feat"] for r in rows], np.float32)
y = np.asarray([r["value"] for r in rows], np.float32)
seeds = np.asarray([r["seed"] for r in rows], np.int64)

# decision groups: consecutive rows, same seed, same scalar dims 11..15
groups, s0 = [], 0
for i in range(1, len(y) + 1):
    if (i == len(y) or seeds[i] != seeds[s0]
            or not np.array_equal(F[i, 11:], F[s0, 11:])):
        if i - s0 >= 2:
            groups.append((s0, i))
        s0 = i

useeds = np.unique(seeds)
rng = np.random.default_rng(0)
val_seeds = set(rng.permutation(useeds)[:int(0.15 * len(useeds))].tolist())
gtr = [g for g in groups if int(seeds[g[0]]) not in val_seeds]
gva = [g for g in groups if int(seeds[g[0]]) in val_seeds]
print("groups train %d  val %d  (val seeds %d/%d)"
      % (len(gtr), len(gva), len(val_seeds), len(useeds)))

ii, jj, ww = [], [], []
for a, b in gtr:
    for i in range(a, b):
        for j in range(a, b):
            if y[i] > y[j]:
                ii.append(i); jj.append(j)
                ww.append(FLIP_W if y[i] - y[j] >= 500 else 1.0)
pi, pj = torch.tensor(ii), torch.tensor(jj)
pw = torch.tensor(ww, dtype=torch.float32)
print("ordered pairs %d  (win-flip %d)" % (len(ii), int((pw > 1).sum())))

X = torch.tensor(F)
H = [FEAT_DIM, 128, 128, 1]
mods = []
for k in range(len(H) - 1):
    mods.append(nn.Linear(H[k], H[k + 1]))
    if k < len(H) - 2:
        mods.append(nn.ReLU())
net = nn.Sequential(*mods)
opt = torch.optim.Adam(net.parameters(), 1e-3)


def heldout():
    net.eval()
    with torch.no_grad():
        sc = net(X).squeeze(-1).numpy()
    agree = rand = vpick = tpick = mpick = 0.0
    for a, b in gva:
        lab = y[a:b]
        tb = np.flatnonzero(lab == lab.max())
        vb = int(np.argmax(sc[a:b]))
        agree += 1.0 if vb in tb else 0.0
        rand += len(tb) / (b - a)
        vpick += lab[vb]; tpick += lab.max(); mpick += lab.mean()
    n = len(gva)
    cap = (vpick - mpick) / max(1e-9, tpick - mpick)
    return agree / n, rand / n, cap


best_cap, best_state, since = -1.0, None, 0
for ep in range(EPOCHS):
    net.train(); opt.zero_grad()
    sc = net(X).squeeze(-1)
    loss = (pw * nn.functional.softplus(sc[pj] - sc[pi])).mean() / pw.mean()
    loss.backward(); opt.step()
    if ep % 25 == 0 or ep == EPOCHS - 1:
        ag, rd, cap = heldout()
        print("ep %4d  loss %.4f  val_top1 %.3f (rand %.3f)  edge_captured %.3f"
              % (ep, float(loss), ag, rd, cap))
        if cap > best_cap:
            best_cap, since = cap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                print("early stop at ep %d" % ep)
                break

net.load_state_dict(best_state)
ag, rd, cap = heldout()
print("BEST  val_top1 %.3f (rand %.3f)  edge_captured %.3f" % (ag, rd, cap))
print("GATE(>=0.5): %s" % ("PASS" if cap >= 0.5 else "FAIL"))

lins = [m for m in net if isinstance(m, nn.Linear)]
kw = {"nlayers": np.array(len(lins)), "vmean": np.array(0.0), "vstd": np.array(1.0)}
for k, m in enumerate(lins):
    kw["W%d" % k] = m.weight.detach().numpy().T.astype(np.float32)
    kw["b%d" % k] = m.bias.detach().numpy().astype(np.float32)
np.savez("draft_v2.npz", **kw)
npv = NpDraftV("draft_v2.npz")
net.eval()
with torch.no_grad():
    mx = max(abs(float(net(X[a:a + 1]).squeeze()) - npv.value(F[a]))
             for a, b in gva[:200])
print("PARITY max|torch-numpy| = %.2e" % mx)
assert mx < 1e-3
print("OK exported draft_v2.npz")
