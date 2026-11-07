import json, sys, numpy as np, torch, torch.nn as nn
from run_state_features import FEAT_DIM
from np_draft_v import NpDraftV

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "draft_data.jsonl")]
X = torch.tensor([r["feat"] for r in rows], dtype=torch.float32)
y = torch.tensor([r["value"] for r in rows], dtype=torch.float32)
vmean, vstd = float(y.mean()), float(y.std() + 1e-6)
yn = (y - vmean) / vstd

g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(rows), generator=g)
cut = int(0.85 * len(rows))
tr, va = idx[:cut], idx[cut:]

H = [FEAT_DIM, 128, 128, 1]
mods = []
for i in range(len(H) - 1):
    mods.append(nn.Linear(H[i], H[i + 1]))
    if i < len(H) - 2:
        mods.append(nn.ReLU())
net = nn.Sequential(*mods)
opt = torch.optim.Adam(net.parameters(), 1e-3)

for ep in range(400):
    net.train(); opt.zero_grad()
    loss = ((net(X[tr]).squeeze(-1) - yn[tr]) ** 2).mean()
    loss.backward(); opt.step()
    if ep % 50 == 0 or ep == 399:
        net.eval()
        with torch.no_grad():
            pv = net(X[va]).squeeze(-1)
            r2 = 1 - ((pv - yn[va]) ** 2).mean() / (yn[va].var() + 1e-9)
        print("ep %3d  val_r2 %.3f" % (ep, float(r2)))

# export to npz (Linear weight is [out,in]; store transposed for x @ W)
lins = [m for m in net if isinstance(m, nn.Linear)]
kw = {"nlayers": np.array(len(lins)), "vmean": np.array(vmean), "vstd": np.array(vstd)}
for i, m in enumerate(lins):
    kw["W%d" % i] = m.weight.detach().numpy().T.astype(np.float32)
    kw["b%d" % i] = m.bias.detach().numpy().astype(np.float32)
np.savez("draft_v.npz", **kw)

# parity: numpy forward must match torch on the val set
npv = NpDraftV("draft_v.npz")
net.eval()
mx = 0.0
with torch.no_grad():
    for j in va[:512].tolist():
        t = float(net(X[j:j + 1]).squeeze()) * vstd + vmean
        mx = max(mx, abs(t - npv.value(X[j].numpy())))
print("PARITY max|torch-numpy| = %.2e" % mx)
assert mx < 1e-3, "numpy/torch mismatch"
print("OK exported draft_v.npz  vmean=%.1f vstd=%.1f" % (vmean, vstd))
