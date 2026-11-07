"""Rank-trained drafting model: same data, same MLP, but pairwise logistic
loss WITHIN decision groups (candidates sharing scalar dims). MSE on absolute
value gave within-decision corr ~0.02; ranking trains exactly the margin the
deploy argmax uses. Exports draft_v_rank.npz (NpDraftV-compatible, vmean=0
vstd=1: scores are only used for argmax)."""
import json, sys, numpy as np, torch, torch.nn as nn
from run_state_features import FEAT_DIM
from np_draft_v import NpDraftV

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "draft_data.jsonl")]
F = np.asarray([r["feat"] for r in rows], np.float32)
y = np.asarray([r["value"] for r in rows], np.float32)

# decision groups = consecutive rows with identical scalar dims 11..15
groups, s0 = [], 0
for i in range(1, len(y) + 1):
    if i == len(y) or not np.array_equal(F[i, 11:], F[s0, 11:]):
        if i - s0 >= 2:
            groups.append((s0, i))
        s0 = i
rng = np.random.default_rng(0)
perm = rng.permutation(len(groups))
cut = int(0.85 * len(groups))
gtr = [groups[k] for k in perm[:cut]]
gva = [groups[k] for k in perm[cut:]]
print("groups train %d  val %d" % (len(gtr), len(gva)))

ii, jj = [], []
for a, b in gtr:
    for i in range(a, b):
        for j in range(a, b):
            if y[i] > y[j]:
                ii.append(i)
                jj.append(j)
pi, pj = torch.tensor(ii), torch.tensor(jj)
print("ordered pairs", len(ii))

X = torch.tensor(F)
H = [FEAT_DIM, 128, 128, 1]
mods = []
for i in range(len(H) - 1):
    mods.append(nn.Linear(H[i], H[i + 1]))
    if i < len(H) - 2:
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
        vb = a + int(np.argmax(sc[a:b]))
        agree += 1.0 if (vb - a) in tb else 0.0
        rand += len(tb) / (b - a)
        vpick += lab[vb - a]
        tpick += lab.max()
        mpick += lab.mean()
    n = len(gva)
    cap = (vpick - mpick) / max(1e-9, tpick - mpick)
    return agree / n, rand / n, cap


best_cap, best_state = -1.0, None
for ep in range(600):
    net.train()
    opt.zero_grad()
    sc = net(X).squeeze(-1)
    loss = nn.functional.softplus(sc[pj] - sc[pi]).mean()
    loss.backward()
    opt.step()
    if ep % 50 == 0 or ep == 599:
        ag, rd, cap = heldout()
        print("ep %3d  loss %.4f  val_top1 %.3f (rand %.3f)  edge_captured %.3f"
              % (ep, float(loss), ag, rd, cap))
        if cap > best_cap:
            best_cap = cap
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

net.load_state_dict(best_state)
ag, rd, cap = heldout()
print("BEST  val_top1 %.3f (rand %.3f)  edge_captured %.3f" % (ag, rd, cap))

lins = [m for m in net if isinstance(m, nn.Linear)]
kw = {"nlayers": np.array(len(lins)), "vmean": np.array(0.0), "vstd": np.array(1.0)}
for i, m in enumerate(lins):
    kw["W%d" % i] = m.weight.detach().numpy().T.astype(np.float32)
    kw["b%d" % i] = m.bias.detach().numpy().astype(np.float32)
np.savez("draft_v_rank.npz", **kw)

npv = NpDraftV("draft_v_rank.npz")
net.eval()
mx = 0.0
with torch.no_grad():
    for a, b in gva[:200]:
        t = float(net(X[a:a + 1]).squeeze())
        mx = max(mx, abs(t - npv.value(F[a])))
print("PARITY max|torch-numpy| = %.2e" % mx)
assert mx < 1e-3, "numpy/torch mismatch"
print("OK exported draft_v_rank.npz")
