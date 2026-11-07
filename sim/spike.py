"""Throwaway Step-0 spike: prove STS-v0 drives end-to-end + benchmark throughput.

Random masked agent. Measures steps/s and games/s, episode length, and reports
whatever info keys the env exposes (so we can see floor/outcome signals).
"""
import time
import numpy as np
import gymnasium as gym
import gym_sts_lightspeed  # noqa: F401  (registers STS-v0)

env = gym.make("STS-v0", ascension=0)
rng = np.random.default_rng(0)

obs, info = env.reset(seed=0)
print("RESET ok")
print("  obs.shape =", np.asarray(obs).shape, "dtype =", np.asarray(obs).dtype)
print("  obs min/max =", int(np.min(obs)), int(np.max(obs)))
print("  info keys  =", sorted(info.keys()))
m = info["action_mask"]
print("  mask.shape =", np.asarray(m).shape, "legal =", int(np.sum(m)))

BUDGET_S = 15.0
steps = 0
games = 0
ep_len = 0
ep_lens = []
floors = []
outcomes = []
t0 = time.time()
while time.time() - t0 < BUDGET_S:
    mask = np.asarray(info["action_mask"])
    legal = np.flatnonzero(mask)
    a = int(rng.choice(legal)) if legal.size else 0
    obs, r, term, trunc, info = env.step(a)
    steps += 1
    ep_len += 1
    if term or trunc:
        games += 1
        ep_lens.append(ep_len)
        ep_len = 0
        # capture any progress/outcome signal the env exposes
        for k in ("floor", "floor_num", "final_floor", "outcome", "victory"):
            if k in info:
                (floors if "floor" in k else outcomes).append(info[k])
        obs, info = env.reset()
dt = time.time() - t0

print("\nBENCHMARK (%.1fs, random masked agent)" % dt)
print("  steps        =", steps, "-> %d steps/s" % round(steps / dt))
print("  games        =", games, "-> %.1f games/s" % (games / dt))
if ep_lens:
    print("  ep_len mean  = %.1f  (min %d, max %d)" % (
        sum(ep_lens) / len(ep_lens), min(ep_lens), max(ep_lens)))
if floors:
    fl = [f for f in floors if isinstance(f, (int, float))]
    if fl:
        print("  floor mean/max =", round(sum(fl) / len(fl), 2), "/", max(fl))
if outcomes:
    print("  outcomes sample =", outcomes[:10])
print("  last info keys  =", sorted(info.keys()))
