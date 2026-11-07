"""Step-2/3 spike: PPO on STS-v0, end-to-end, to prove a learning signal + measure
real throughput with NN inference. Single env, compact PPO. Guards the fork's
getObservation (carry-forward) so the ~1.7% combat-obs crash can't abort training.

Success = mean episode floor rises above the random baseline (~4.15).
"""
import sys
import time
import numpy as np
# IMPORTANT: import the slaythespire C++ extension BEFORE torch. torch ships its
# own libstdc++; if it loads first, the extension's std::array/unordered_map ops
# bind to a mismatched C++ runtime and segfault on the first getObservation call.
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire
import torch
import torch.nn as nn
from torch.distributions import Categorical

OBS, ACT = 1200, 128
DEVICE = "cpu"
torch.manual_seed(0)
np.random.seed(0)

maxima = np.asarray(slaythespire.getNNInterface().getObservationMaximums(), dtype=np.float32)
maxima = np.maximum(maxima, 1.0)


class SafeStsEnv(StsEnv):
    """Carry-forward guard around the fork's brittle getObservation."""
    obs_fail = 0
    _last_obs = None

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last_obs = o
            return o
        except Exception:
            SafeStsEnv.obs_fail += 1
            if self._last_obs is None:
                return np.zeros(OBS, dtype=np.int32)
            return self._last_obs


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.pi = nn.Linear(256, ACT)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def norm(o):
    return torch.as_tensor(np.asarray(o, dtype=np.float32) / maxima, dtype=torch.float32)


def masked_logits(logits, mask):
    m = torch.as_tensor(mask, dtype=torch.float32)
    return logits + (m - 1.0) * 1e9  # illegal -> -1e9


net = ActorCritic().to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

GAMMA, LAM, CLIP, VF, ENT = 0.99, 0.95, 0.2, 0.5, 0.01
T = 4096          # steps per update
EPOCHS, MB = 4, 512
BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0

env = SafeStsEnv(ascension=0)
obs, info = env.reset(seed=0)
ep_ret, ep_floor = 0.0, 0
recent_floors, recent_rets = [], []

t0 = time.time()
total_steps = 0
total_games = 0
upd = 0
while time.time() - t0 < BUDGET_S:
    bobs, bact, blogp, bval, brew, bdone, bmask = [], [], [], [], [], [], []
    for _ in range(T):
        x = norm(obs)
        mask = info["action_mask"]
        with torch.no_grad():
            logits, value = net(x)
            dist = Categorical(logits=masked_logits(logits, mask))
            a = dist.sample()
            logp = dist.log_prob(a)
        nobs, r, term, trunc, ninfo = env.step(int(a.item()))
        bobs.append(x); bact.append(a); blogp.append(logp); bval.append(value)
        brew.append(float(r)); bdone.append(term or trunc); bmask.append(mask)
        ep_ret += float(r); ep_floor = info.get("floor", ep_floor)
        total_steps += 1
        obs, info = nobs, ninfo
        if term or trunc:
            total_games += 1
            recent_floors.append(env.gc.floor_num)
            recent_rets.append(ep_ret)
            recent_floors[:] = recent_floors[-200:]
            recent_rets[:] = recent_rets[-200:]
            ep_ret, ep_floor = 0.0, 0
            obs, info = env.reset()

    # bootstrap
    with torch.no_grad():
        _, last_v = net(norm(obs))
    vals = torch.stack(bval + [last_v]).detach()
    rew = torch.tensor(brew, dtype=torch.float32)
    done = torch.tensor(bdone, dtype=torch.float32)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + GAMMA * vals[t + 1] * nonterm - vals[t]
        gae = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    ret = adv + vals[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    Obs = torch.stack(bobs)
    Act = torch.stack(bact)
    Logp = torch.stack(blogp).detach()
    Mask = torch.tensor(np.array(bmask), dtype=torch.float32)

    idx = np.arange(T)
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, T, MB):
            j = idx[s:s + MB]
            logits, value = net(Obs[j])
            logits = logits + (Mask[j] - 1.0) * 1e9
            dist = Categorical(logits=logits)
            lp = dist.log_prob(Act[j])
            ratio = torch.exp(lp - Logp[j])
            a_j = adv[j]
            pg = -torch.min(ratio * a_j, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_j).mean()
            vloss = ((value - ret[j]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + VF * vloss - ENT * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

    upd += 1
    dt = time.time() - t0
    mf = np.mean(recent_floors) if recent_floors else 0
    mr = np.mean(recent_rets) if recent_rets else 0
    print("upd %2d | games %5d | steps/s %4d | mean_floor %.2f | mean_ret %.1f | "
          "ent %.2f | obs_fail %d" % (
              upd, total_games, round(total_steps / dt), mf, mr,
              float(ent), SafeStsEnv.obs_fail), flush=True)

print("\nDONE: %.0fs, %d updates, %d games, final mean_floor %.2f (random baseline ~4.15)" % (
    time.time() - t0, upd, total_games, np.mean(recent_floors) if recent_floors else 0))
