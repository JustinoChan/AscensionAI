"""Discriminator: can the value head FIT the batch it trains on, vs GENERALIZE
to the next rollout? For each update, log:
  ev_pre   = ev on this batch using collection-time V (generalization from
             prior updates)
  ev_post  = ev on this SAME batch after the 4 value epochs (in-sample fit)
If ev_post is high but ev_pre ~0 -> value CAN fit, the failure is
non-stationarity/too-little-data (fix: value normalization / more data / more
value optimization). If ev_post is also ~0 -> the value head cannot fit even
its own training batch (fix: value lr/epochs/target scale).
"""
import sys
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagVF", "fightmix"]
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import train_sim_mp as tm

N, T = tm.N, tm.T
GAMMA, LAM, CLIP, VF = tm.GAMMA, tm.LAM, tm.CLIP, tm.VF
EPOCHS, MB = tm.EPOCHS, tm.MB
RS = tm.REWARD_SCALE
UPD = 80

vec = tm.VecEnv(N)
net = tm.ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
ent_coef = tm.ENT_INIT


def ev(ret, v):
    return float(1.0 - (ret - v).var() / (ret.var() + 1e-8))


for upd in range(1, UPD + 1):
    O, A, LP, V, R, D, M = [], [], [], [], [], [], []
    for _ in range(T):
        obs_t = torch.as_tensor(vec.obs)
        mask_t = torch.as_tensor(vec.mask, dtype=torch.float32)
        with torch.no_grad():
            logits, val = net(obs_t)
            dist = Categorical(logits=logits + (mask_t - 1.0) * 1e9)
            a = dist.sample(); lp = dist.log_prob(a)
        O.append(obs_t); A.append(a); LP.append(lp); V.append(val); M.append(mask_t)
        _, r, d, _, epfl, ndone = vec.step(a.numpy())
        R.append(torch.as_tensor(r) * RS); D.append(torch.as_tensor(d))
    with torch.no_grad():
        _, last_v = net(torch.as_tensor(vec.obs))
    Vs = torch.stack(V + [last_v]); Rs = torch.stack(R); Ds = torch.stack(D)
    adv = torch.zeros(T, N); gae = torch.zeros(N)
    for t in reversed(range(T)):
        nonterm = 1.0 - Ds[t]
        delta = Rs[t] + GAMMA * Vs[t + 1] * nonterm - Vs[t]
        gae = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    ret = (adv + Vs[:-1]).reshape(-1)
    vflat_pre = Vs[:-1].reshape(-1)
    ev_pre = ev(ret, vflat_pre)
    Of = torch.cat(O); Af = torch.cat(A); LPf = torch.cat(LP); Mf = torch.cat(M)
    advf = adv.reshape(-1); advf = (advf - advf.mean()) / (advf.std() + 1e-8)
    idx = np.arange(T * N)
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, T * N, MB):
            j = idx[s:s + MB]
            logits, val = net(Of[j])
            dist = Categorical(logits=logits + (Mf[j] - 1.0) * 1e9)
            lp = dist.log_prob(Af[j]); ratio = torch.exp(lp - LPf[j]); aj = advf[j]
            pg = -torch.min(ratio * aj, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * aj).mean()
            vl = ((val - ret[j]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + VF * vl - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    with torch.no_grad():
        _, vpost = net(Of)
    ev_post = ev(ret, vpost)
    if upd % 10 == 0 or upd == 1:
        print("upd %3d | ev_pre %5.2f | ev_post(in-sample fit) %5.2f | "
              "V sd %.3f ret sd %.3f" % (
                  upd, ev_pre, ev_post, float(vpost.std()), float(ret.std())),
              flush=True)
vec.close()
print("DONE")
