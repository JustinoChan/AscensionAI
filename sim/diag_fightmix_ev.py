"""Live-PPO diagnostic with the NEW trainer (separate value trunk, OBS 1235,
gamma 0.97) in fightmix. Runs ~120 updates and prints explained variance split
by worker type: fight-worker steps (target ~0.98 learnable) vs run-worker steps
(~0.22). Answers: does the separate trunk let the value head capture the
learnable fight signal that varF's shared trunk missed (ev~0)?"""
import sys
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagFE", "fightmix"]
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import train_sim_mp as tm

N, T = tm.N, tm.T
N_RUN = tm.N_RUN
GAMMA, LAM, CLIP, VF = tm.GAMMA, tm.LAM, tm.CLIP, tm.VF
EPOCHS, MB = tm.EPOCHS, tm.MB
RS = tm.REWARD_SCALE
UPD = 120

vec = tm.VecEnv(N)
net = tm.ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
ent_coef = tm.ENT_INIT
run_cols = list(range(N_RUN))
fight_cols = list(range(N_RUN, N))


def ev_of(ret2d, v2d, cols):
    r = ret2d[:, cols].reshape(-1)
    v = v2d[:, cols].reshape(-1)
    return float(1.0 - (r - v).var() / (r.var() + 1e-8))


for upd in range(1, UPD + 1):
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
    Vs = torch.stack(V + [last_v]); Rs = torch.stack(R); Ds = torch.stack(D)
    adv = torch.zeros(T, N); gae = torch.zeros(N)
    for t in reversed(range(T)):
        nonterm = 1.0 - Ds[t]
        delta = Rs[t] + GAMMA * Vs[t + 1] * nonterm - Vs[t]
        gae = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    ret2d = (adv + Vs[:-1])           # [T, N]
    v2d = Vs[:-1]
    if upd % 10 == 0 or upd == 1:
        print("upd %3d | ev all %5.2f | ev fight %5.2f | ev run %5.2f" % (
            upd, ev_of(ret2d, v2d, list(range(N))),
            ev_of(ret2d, v2d, fight_cols), ev_of(ret2d, v2d, run_cols)),
            flush=True)
    ret = ret2d.reshape(-1)
    Of = torch.cat(O); Af = torch.cat(A); LPf = torch.cat(LP); Mf = torch.cat(M)
    advf = adv.reshape(-1)
    advf = (advf - advf.mean()) / (advf.std() + 1e-8)
    idx = np.arange(T * N)
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, T * N, MB):
            j = idx[s:s + MB]
            logits, val = net(Of[j])
            dist = Categorical(logits=logits + (Mf[j] - 1.0) * 1e9)
            lp = dist.log_prob(Af[j])
            ratio = torch.exp(lp - LPf[j])
            aj = advf[j]
            pg = -torch.min(ratio * aj, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * aj).mean()
            vl = ((val - ret[j]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + VF * vl - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
vec.close()
print("DIAG FIGHTMIX EV DONE")
