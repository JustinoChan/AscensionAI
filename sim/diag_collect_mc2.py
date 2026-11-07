"""Collect (obs, MC return) pairs under the frozen varE policy with returns
computed at FIVE gammas in one pass -- finds the horizon at which the value
target becomes predictable from a single obs."""
import sys
import os
sys.argv = ["train_sim_mp.py", "999999", "12", "999999", "diagMC2", "bare"]
import numpy as np
import torch
from torch.distributions import Categorical
import train_sim_mp as tm

GAMMAS = [0.99, 0.97, 0.95, 0.90, 0.80]
KEYS = ["ret99", "ret97", "ret95", "ret90", "ret80"]
RS = tm.REWARD_SCALE
TARGET_EPISODES = 240
OUT = "diag_mc2"
os.makedirs(OUT, exist_ok=True)

net = tm.ActorCritic()
ck = torch.load("sim_models_varE/ppo_sim_final.pt", map_location="cpu")
net.load_state_dict(ck["model"])
vec = tm.VecEnv(tm.N)

buf = [{"obs": [], "r": []} for _ in range(tm.N)]
state = {"done_eps": 0, "fidx": 0, "samples": 0}
acc = {k: [] for k in KEYS}
acc["obs"] = []


def flush_episode(i):
    obs_l, r_l = buf[i]["obs"], buf[i]["r"]
    buf[i] = {"obs": [], "r": []}
    if len(obs_l) < 5:
        return
    L = len(r_l)
    for g, k in zip(GAMMAS, KEYS):
        G = 0.0
        gs = [0.0] * L
        for t in reversed(range(L)):
            G = r_l[t] + g * G
            gs[t] = G
        acc[k].append(np.asarray(gs, np.float32)[::3])
    acc["obs"].append(np.asarray(obs_l, np.float16)[::3])
    state["samples"] += len(acc["obs"][-1])
    state["done_eps"] += 1
    if len(acc["obs"]) >= 12:
        state["fidx"] += 1
        np.savez_compressed(
            os.path.join(OUT, "b%04d.npz" % state["fidx"]),
            **{k: np.concatenate(v) for k, v in acc.items()})
        for k in acc:
            del acc[k][:]


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
