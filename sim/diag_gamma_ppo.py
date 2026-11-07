"""Confirmatory test: does lowering gamma (shorter horizon -> more learnable
value target, per the offline sweep) raise explained variance in LIVE PPO?
Runs the trainer's exact inner loop for a few gammas, 50 updates each, fresh
net each time, same workers. Reports mean ev over the last 20 updates.
Also tests a SEPARATE value head (no shared-trunk noise injection) at gamma
0.95 as a second lever."""
import sys
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagG", "bare"]
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import train_sim_mp as tm

N, T = tm.N, tm.T
LAM, CLIP, VF = tm.LAM, tm.CLIP, tm.VF
EPOCHS, MB = tm.EPOCHS, tm.MB
RS = tm.REWARD_SCALE
UPD = 50
OBS, ACT = tm.OBS, tm.ACT


class SepAC(nn.Module):
    """Separate policy and value trunks (no shared features)."""
    def __init__(self):
        super().__init__()
        self.pbody = nn.Sequential(nn.Linear(OBS, 512), nn.GELU(),
                                   nn.Linear(512, 256), nn.GELU(),
                                   nn.Linear(256, 256), nn.GELU())
        self.vbody = nn.Sequential(nn.Linear(OBS, 256), nn.GELU(),
                                   nn.Linear(256, 256), nn.GELU())
        self.pi = nn.Linear(256, ACT)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        return self.pi(self.pbody(x)), self.v(self.vbody(x)).squeeze(-1)


def run(vec, gamma, net, tag):
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    ent_coef = tm.ENT_INIT
    evs = []
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
            delta = Rs[t] + gamma * Vs[t + 1] * nonterm - Vs[t]
            gae = delta + gamma * LAM * nonterm * gae
            adv[t] = gae
        ret = (adv + Vs[:-1]).reshape(-1)
        vflat = Vs[:-1].reshape(-1)
        ev = float(1.0 - (ret - vflat).var() / (ret.var() + 1e-8))
        if upd > UPD - 20:
            evs.append(ev)
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
    print("%-28s | mean ev (last 20 upd) = %6.3f" % (tag, float(np.mean(evs))),
          flush=True)


vec = tm.VecEnv(N)
torch.manual_seed(0); run(vec, 0.99, tm.ActorCritic(), "shared trunk, gamma 0.99")
torch.manual_seed(0); run(vec, 0.95, tm.ActorCritic(), "shared trunk, gamma 0.95")
torch.manual_seed(0); run(vec, 0.90, tm.ActorCritic(), "shared trunk, gamma 0.90")
torch.manual_seed(0); run(vec, 0.95, SepAC(), "SEPARATE value, gamma 0.95")
torch.manual_seed(0); run(vec, 0.90, SepAC(), "SEPARATE value, gamma 0.90")
vec.close()
print("GAMMA PPO TEST DONE")
