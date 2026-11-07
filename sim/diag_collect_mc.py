"""Collect (obs, Monte-Carlo discounted return) pairs under the FROZEN varE
policy (sampling, matching training rollouts). Bootstrap-free targets for the
supervised learnability probe: can discounted future reward be predicted from
a single obs at all in this env?"""
import sys
import os
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagMC", "bare"]
import numpy as np
import torch
from torch.distributions import Categorical
import train_sim_mp as tm

GAMMA = tm.GAMMA
RS = tm.REWARD_SCALE
TARGET_EPISODES = 240
OUT = "diag_mc"
os.makedirs(OUT, exist_ok=True)

net = tm.ActorCritic()
ck = torch.load("sim_models_varE/ppo_sim_final.pt", map_location="cpu")
net.load_state_dict(ck["model"])
vec = tm.VecEnv(tm.N)

buf = [{"obs": [], "r": []} for _ in range(tm.N)]
state = {"done_eps": 0, "fidx": 0, "samples": 0}
out_obs, out_g = [], []


def flush_episode(i):
    obs_l, r_l = buf[i]["obs"], buf[i]["r"]
    buf[i] = {"obs": [], "r": []}
    if len(obs_l) < 5:
        return
    G = 0.0
    gs = [0.0] * len(r_l)
    for t in reversed(range(len(r_l))):
        G = r_l[t] + GAMMA * G
        gs[t] = G
    out_obs.append(np.asarray(obs_l, np.float16)[::3])
    out_g.append(np.asarray(gs, np.float32)[::3])
    state["samples"] += len(out_g[-1])
    state["done_eps"] += 1
    if len(out_obs) >= 12:
        state["fidx"] += 1
        np.savez_compressed(os.path.join(OUT, "b%04d.npz" % state["fidx"]),
                            obs=np.concatenate(out_obs),
                            ret=np.concatenate(out_g))
        del out_obs[:], out_g[:]


with torch.no_grad():
    while state["done_eps"] < TARGET_EPISODES:
        obs_t = torch.as_tensor(vec.obs)
        mask_t = torch.as_tensor(vec.mask, dtype=torch.float32)
        logits, _ = net(obs_t)
        dist = Categorical(logits=logits + (mask_t - 1.0) * 1e9)
        a = dist.sample()
        pre_obs = vec.obs.copy()
        _, r, d, _, epfl, ndone = vec.step(a.numpy())
        for i in range(tm.N):
            buf[i]["obs"].append(pre_obs[i])
            buf[i]["r"].append(float(r[i]) * RS)
            if d[i] == 1.0:
                flush_episode(i)
vec.close()
print("collected %d episodes, %d samples, %d files" % (
    state["done_eps"], state["samples"], state["fidx"]))
