"""Instrumented short PPO run (bare mode, 12 workers, ~150 updates) to
diagnose the dead value head. Replicates train_sim_mp's loop exactly (same
GAE, same normalization, same clip) and logs per update:
  - std/mean of V and ret, corr(V,ret), ev
  - value-loss magnitude and SPLIT gradient norms (policy loss vs value loss)
  - return percentiles (heavy-tail check)
Dumps (obs, ret) pairs every 5th update for the offline supervised probe.
"""
import sys
import os
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagV", "bare"]
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import train_sim_mp as tm

MAX_UPD = 150
DUMP_EVERY = 5
N, T = tm.N, tm.T
GAMMA, LAM, CLIP, VF = tm.GAMMA, tm.LAM, tm.CLIP, tm.VF
EPOCHS, MB = tm.EPOCHS, tm.MB
RS = tm.REWARD_SCALE

os.makedirs("diag_dumps", exist_ok=True)
print("launching %d bare workers..." % N, flush=True)
vec = tm.VecEnv(N)
net = tm.ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
ent_coef = tm.ENT_INIT


def gnorm(params):
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.pow(2).sum())
    return s ** 0.5


for upd in range(1, MAX_UPD + 1):
    O, A, LP, V, R, D, M = [], [], [], [], [], [], []
    for _ in range(T):
        obs_t = torch.as_tensor(vec.obs)
        mask_t = torch.as_tensor(vec.mask, dtype=torch.float32)
        with torch.no_grad():
            logits, val = net(obs_t)
            dist = Categorical(logits=logits + (mask_t - 1.0) * 1e9)
            a = dist.sample()
            lp = dist.log_prob(a)
        O.append(obs_t); A.append(a); LP.append(lp); V.append(val); M.append(mask_t)
        _, r, d, _, epfl, ndone = vec.step(a.numpy())
        R.append(torch.as_tensor(r) * RS)
        D.append(torch.as_tensor(d))
    with torch.no_grad():
        _, last_v = net(torch.as_tensor(vec.obs))
    Vs = torch.stack(V + [last_v])
    Rs = torch.stack(R)
    Ds = torch.stack(D)
    adv = torch.zeros(T, N)
    gae = torch.zeros(N)
    for t in reversed(range(T)):
        nonterm = 1.0 - Ds[t]
        delta = Rs[t] + GAMMA * Vs[t + 1] * nonterm - Vs[t]
        gae = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    ret = (adv + Vs[:-1]).reshape(-1)
    Of = torch.cat(O); Af = torch.cat(A); LPf = torch.cat(LP); Mf = torch.cat(M)
    advf = adv.reshape(-1)
    vflat = Vs[:-1].reshape(-1)

    # ---- diagnostics (pre-update) ----
    ev = float(1.0 - (ret - vflat).var() / (ret.var() + 1e-8))
    corr = float(np.corrcoef(vflat.numpy(), ret.numpy())[0, 1])
    q = np.percentile(ret.numpy(), [0, 1, 50, 99, 100])
    advn_all = (advf - advf.mean()) / (advf.std() + 1e-8)
    j = np.arange(MB)
    logits, val = net(Of[j])
    dist = Categorical(logits=logits + (Mf[j] - 1.0) * 1e9)
    lp = dist.log_prob(Af[j])
    ratio = torch.exp(lp - LPf[j])
    aj = advn_all[j]
    pg = -torch.min(ratio * aj, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * aj).mean()
    vl = ((val - ret[j]) ** 2).mean()
    opt.zero_grad(); pg.backward(retain_graph=True)
    g_pg_all = gnorm(net.parameters())
    opt.zero_grad(); (VF * vl).backward(retain_graph=True)
    g_vl_all = gnorm(net.parameters())
    g_vl_vhead = gnorm(net.v.parameters())
    opt.zero_grad()

    if upd % 5 == 0 or upd == 1:
        print("upd %3d | V mu %7.3f sd %6.3f | ret mu %7.3f sd %6.3f | "
              "corr %5.2f | ev %5.2f | vl %9.2f | g_pg %7.3f g_vl %8.3f "
              "(vhead %7.3f) | ret[min,p1,p50,p99,max]=[%.1f,%.1f,%.1f,%.1f,%.1f]" % (
                  upd, float(vflat.mean()), float(vflat.std()),
                  float(ret.mean()), float(ret.std()), corr, ev, float(vl),
                  g_pg_all, g_vl_all, g_vl_vhead, q[0], q[1], q[2], q[3], q[4]),
              flush=True)
    if upd % DUMP_EVERY == 0:
        np.savez_compressed("diag_dumps/b%04d.npz",
                            obs=Of.numpy()[::2].astype(np.float16),
                            ret=ret.numpy()[::2].astype(np.float32))

    # ---- the actual update, exactly as the trainer does ----
    advn = advn_all
    idx = np.arange(T * N)
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, T * N, MB):
            jj = idx[s:s + MB]
            logits, val = net(Of[jj])
            dist = Categorical(logits=logits + (Mf[jj] - 1.0) * 1e9)
            lp = dist.log_prob(Af[jj])
            ratio = torch.exp(lp - LPf[jj])
            aj = advn[jj]
            pg = -torch.min(ratio * aj,
                            torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * aj).mean()
            vl = ((val - torch.as_tensor(ret.numpy()[jj])) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + VF * vl - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

vec.close()
print("DIAG RUN DONE")
