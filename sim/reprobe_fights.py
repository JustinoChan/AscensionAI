"""Targeted A/B: does the enriched obs make the FIGHT-episode return learnable?
This is the distribution varG's fight workers train on (single fights, terminal
+/-50), where the full-run reprobe's long-horizon stochasticity is absent.

Replay bank fights, play with the frozen varE policy (acting on the 1200-d obs),
record (obs1200, obs1235, gamma-0.97 within-fight return), then compare best
held-out R2 with and without the 35 combat features."""
import sys
import json
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import slaythespire
from sim_worker_fight import FightEnv, MAX_FIGHT_STEPS
from combat_features import combat_block

GAMMA = 0.97
WIN_BONUS, LOSS_PENALTY = 50.0, -50.0
RS = 0.1
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
TARGET_FIGHTS = 1200


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpPolicy:
    def __init__(self, p):
        w = np.load(p)
        self.layers = [(w["body.%d.weight" % i], w["body.%d.bias" % i])
                       for i in (0, 2, 4)]
        self.pi = (w["pi.weight"], w["pi.bias"])

    def act(self, x, mask):
        h = x
        for W, b in self.layers:
            h = gelu(h @ W.T + b)
        logits = h @ self.pi[0].T + self.pi[1]
        return int(np.argmax(np.where(mask > 0, logits, -1e9)))


pol = NpPolicy("policy_varE.npz")
bank = [json.loads(l) for l in open("bank_train.jsonl") if l.strip()]
rng = np.random.default_rng(0)
env = FightEnv(ascension=0)
maxima = np.maximum(np.asarray(env.nn.getObservationMaximums(), np.float32), 1.0)

X1200, X1235, Y = [], [], []
fights = 0
while fights < TARGET_FIGHTS:
    e = bank[int(rng.integers(len(bank)))]
    if env.reset_to_entry(e) is None:
        continue
    o1200s, o1235s, rs = [], [], []
    steps = 0
    last_hp = env.bc.player.cur_hp
    while steps < MAX_FIGHT_STEPS:
        obs = env._get_observation()
        o1200 = np.asarray(obs, np.float32) / maxima
        o1235 = np.concatenate([o1200, combat_block(env)])
        a = pol.act(o1200, np.asarray(env._get_info()["action_mask"]))
        o1200s.append(o1200.astype(np.float16))
        o1235s.append(o1235.astype(np.float16))
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        cur_hp = env.bc.player.cur_hp if (env.in_combat and env.bc) else env.gc.cur_hp
        rew = (cur_hp - last_hp) * 1.0
        last_hp = cur_hp
        if term or not env.in_combat:
            if env.gc.outcome == _LOSS:
                rew += LOSS_PENALTY
            elif not env.in_combat:
                rew += WIN_BONUS
            rs.append(rew * RS)
            break
        rs.append(rew * RS)
        if steps >= MAX_FIGHT_STEPS:
            break
    G = 0.0
    gs = [0.0] * len(rs)
    for t in reversed(range(len(rs))):
        G = rs[t] + GAMMA * G
        gs[t] = G
    X1200.extend(o1200s); X1235.extend(o1235s); Y.extend(gs)
    fights += 1

X1200 = np.asarray(X1200, np.float32)
X1235 = np.asarray(X1235, np.float32)
Y = np.asarray(Y, np.float32)
print("fights %d | samples %d | return sd %.3f" % (fights, len(Y), Y.std()),
      flush=True)

# 80/20 split by sample
n = len(Y); cut = int(n * 0.8)
perm = np.random.RandomState(0).permutation(n)
tr, va = perm[:cut], perm[cut:]


def best_val_r2(X, dim):
    Xt = torch.as_tensor(X[tr]); Yt = torch.as_tensor(Y[tr])
    Xv = torch.as_tensor(X[va]); Yv = torch.as_tensor(Y[va])
    net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(),
                        nn.Linear(512, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    m = len(Xt); best = -9.0
    for ep in range(1, 81):
        pp = torch.randperm(m)
        for s in range(0, m, 512):
            j = pp[s:s + 512]
            loss = ((net(Xt[j]).squeeze(-1) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pr = net(Xv).squeeze(-1)
            best = max(best, float(1 - (Yv - pr).var() / (Yv.var() + 1e-8)))
    return best


r1200 = best_val_r2(X1200, 1200)
r1235 = best_val_r2(X1235, 1235)
print("FIGHT-return val R2 | 1200-d %.3f | 1235-d enriched %.3f | delta %.3f"
      % (r1200, r1235, r1235 - r1200))
print("VERDICT:", "PASS" if (r1235 >= 0.3 and r1235 - r1200 >= 0.1)
      else "INVESTIGATE")
