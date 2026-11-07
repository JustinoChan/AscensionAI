"""Confirm the value-optimization diagnosis. On collected fightmix batches,
compare the value head's in-sample ev under three regimes on the SAME batch:
  (a) standard 4-epoch joint update (raw targets)
  (b) + 60 extra value-only epochs, RAW targets
  (c) a fresh value head fit for 60 epochs on NORMALIZED targets (zero-mean,
      unit-std) -- the proposed production fix
If (b)/(c) reach high ev, the bottleneck is value optimization budget/scale,
and target normalization is the clean fix.
"""
import sys
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagVH", "fightpure"]
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import train_sim_mp as tm

N, T = tm.N, tm.T
GAMMA, LAM, CLIP, VF = tm.GAMMA, tm.LAM, tm.CLIP, tm.VF
EPOCHS, MB = tm.EPOCHS, tm.MB
RS = tm.REWARD_SCALE

vec = tm.VecEnv(N)
net = tm.ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
ent_coef = tm.ENT_INIT


def ev(ret, v):
    return float(1.0 - (ret - v).var() / (ret.var() + 1e-8))


def collect():
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
    return torch.cat(O), torch.cat(A), torch.cat(LP), torch.cat(M), adv.reshape(-1), ret


# warm up the policy a little so states are realistic
for _ in range(20):
    Of, Af, LPf, Mf, advf, ret = collect()
    advn = (advf - advf.mean()) / (advf.std() + 1e-8)
    idx = np.arange(T * N)
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, T * N, MB):
            j = idx[s:s + MB]
            logits, val = net(Of[j])
            dist = Categorical(logits=logits + (Mf[j] - 1.0) * 1e9)
            lp = dist.log_prob(Af[j]); ratio = torch.exp(lp - LPf[j]); aj = advn[j]
            pg = -torch.min(ratio * aj, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * aj).mean()
            vl = ((val - ret[j]) ** 2).mean()
            loss = pg + VF * vl - ent_coef * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()

# fresh batch for the experiment
Of, Af, LPf, Mf, advf, ret = collect()
with torch.no_grad():
    _, v_std = net(Of)
print("(a) standard 4-epoch joint update  | in-sample ev %.3f | V sd %.3f ret sd %.3f"
      % (ev(ret, v_std), float(v_std.std()), float(ret.std())), flush=True)

# (b) 60 extra value-only epochs, RAW targets, on the value trunk
vopt = torch.optim.Adam(list(net.vbody.parameters()) + list(net.v.parameters()), lr=1e-3)
idx = np.arange(T * N)
for _ in range(60):
    np.random.shuffle(idx)
    for s in range(0, T * N, MB):
        j = idx[s:s + MB]
        _, val = net(Of[j])
        vl = ((val - ret[j]) ** 2).mean()
        vopt.zero_grad(); vl.backward(); vopt.step()
with torch.no_grad():
    _, v_heavy = net(Of)
print("(b) +60 value-only epochs RAW       | in-sample ev %.3f | V sd %.3f"
      % (ev(ret, v_heavy), float(v_heavy.std())), flush=True)

# (c) fresh value head, NORMALIZED targets
mu, sd = float(ret.mean()), float(ret.std()) + 1e-6
retn = (ret - mu) / sd
vhead = nn.Sequential(nn.Linear(tm.OBS, 256), nn.GELU(),
                      nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
vopt2 = torch.optim.Adam(vhead.parameters(), lr=1e-3)
for _ in range(60):
    np.random.shuffle(idx)
    for s in range(0, T * N, MB):
        j = idx[s:s + MB]
        pred = vhead(Of[j]).squeeze(-1)
        vl = ((pred - retn[j]) ** 2).mean()
        vopt2.zero_grad(); vl.backward(); vopt2.step()
with torch.no_grad():
    vn = vhead(Of).squeeze(-1) * sd + mu
print("(c) fresh head, NORMALIZED targets  | in-sample ev %.3f | V sd %.3f"
      % (ev(ret, vn), float(vn.std())), flush=True)
vec.close()
print("DONE")
