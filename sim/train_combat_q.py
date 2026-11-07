"""Combat student: distill the 1-ply search teacher's Q-landscape with a
within-state ranking loss over the TOPK candidates. Held-out split is by
SEED. Reports top-1 agreement and teacher-Q regret (teacher units: dense
combat reward + terminal +/-50; mean regret << 50 is the scale to beat).
argv: [1] qland dir (default sim_logs/qland)  [2] epochs (default 20)"""
import glob, sys, numpy as np, torch, torch.nn as nn

QDIR = sys.argv[1] if len(sys.argv) > 1 else "sim_logs/qland"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BATCH = 2048

obs_l, cands_l, qs_l, sid_l = [], [], [], []
for f in sorted(glob.glob(QDIR + "/q_*.npz")):
    d = np.load(f)
    obs_l.append(d["obs"]); cands_l.append(d["cands"]); qs_l.append(d["qs"])
    sid_l.append(np.full(d["obs"].shape[0], int(f.split("_")[-1][:-4])))
OBS = np.concatenate(obs_l).astype(np.float32)
CAND = np.concatenate(cands_l).astype(np.int64)
QS = np.concatenate(qs_l).astype(np.float32)
SID = np.concatenate(sid_l)
print("states %d  seeds %d" % (OBS.shape[0], len(np.unique(SID))))

useeds = np.unique(SID)
rng = np.random.default_rng(0)
val_seeds = rng.permutation(useeds)[:max(1, int(0.10 * len(useeds)))]
va = np.isin(SID, val_seeds)
tr = ~va
print("train %d  val %d states" % (int(tr.sum()), int(va.sum())))

net = nn.Sequential(nn.Linear(1235, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, 128))
opt = torch.optim.Adam(net.parameters(), 1e-3)
Xtr = torch.tensor(OBS[tr]); Ctr = torch.tensor(CAND[tr]); Qtr = torch.tensor(QS[tr])
idx_tr = np.arange(Xtr.shape[0])


def state_rank_loss(scores, cands, qs):
    # scores [B,128]; cands [B,K] (-1 pad); qs [B,K] (NaN pad).
    # pair[b,i,j] = teacher says candidate i beats j; loss pushes s_i > s_j
    # via softplus(s_j - s_i) = softplus(s.unsqueeze(1) - s.unsqueeze(2))[b,i,j].
    s = torch.gather(scores, 1, cands.clamp(min=0))            # [B,K]
    valid = (cands >= 0) & torch.isfinite(qs)
    pair = ((qs.unsqueeze(2) > qs.unsqueeze(1))
            & valid.unsqueeze(2) & valid.unsqueeze(1))
    if not bool(pair.any()):
        return None
    return nn.functional.softplus(s.unsqueeze(1) - s.unsqueeze(2))[pair].mean()


def heldout():
    net.eval()
    Xv = torch.tensor(OBS[va]); Cv = CAND[va]; Qv = QS[va]
    with torch.no_grad():
        S = np.concatenate([net(Xv[k:k + 8192]).numpy()
                            for k in range(0, Xv.shape[0], 8192)])
    top1 = rand = 0.0; regs = []
    for i in range(Cv.shape[0]):
        m = (Cv[i] >= 0) & np.isfinite(Qv[i])
        if int(m.sum()) < 2:
            continue
        c, q = Cv[i][m], Qv[i][m]
        pick = int(np.argmax(S[i][c]))
        best = np.flatnonzero(q == q.max())
        top1 += 1.0 if pick in best else 0.0
        rand += len(best) / len(q)
        regs.append(float(q.max() - q[pick]))
    n = max(1, len(regs))
    return top1 / n, rand / n, float(np.mean(regs)), float(np.median(regs)), n


for ep in range(EPOCHS):
    net.train()
    rng.shuffle(idx_tr)
    tot = nb = 0.0
    for k in range(0, len(idx_tr), BATCH):
        b = torch.tensor(idx_tr[k:k + BATCH])
        opt.zero_grad()
        loss = state_rank_loss(net(Xtr[b]), Ctr[b], Qtr[b])
        if loss is None:
            continue
        loss.backward(); opt.step()
        tot += float(loss); nb += 1
    t1, rd, rmean, rmed, n = heldout()
    print("ep %2d  loss %.4f  val_top1 %.3f (rand %.3f)  regret mean %.2f med %.3f (n=%d)"
          % (ep, tot / max(1, nb), t1, rd, rmean, rmed, n))

lins = [m for m in net if isinstance(m, nn.Linear)]
kw = {"nlayers": np.array(len(lins))}
for i, m in enumerate(lins):
    kw["W%d" % i] = m.weight.detach().numpy().T.astype(np.float32)
    kw["b%d" % i] = m.bias.detach().numpy().astype(np.float32)
np.savez("combat_q_student.npz", **kw)
from np_qnet import NpQNet
npq = NpQNet("combat_q_student.npz")
net.eval()
with torch.no_grad():
    t = net(torch.tensor(OBS[:64])).numpy()
mx = float(np.abs(t - np.stack([npq.q(o) for o in OBS[:64]])).max())
print("PARITY max|torch-numpy| = %.2e" % mx)
assert mx < 1e-3
print("OK exported combat_q_student.npz")
