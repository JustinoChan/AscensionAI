"""Decisive fork: how learnable is the PURE-fight return under a SAMPLING
policy (what training uses) vs GREEDY (0.98 baseline)? Drives bank fights with
the varE policy, once with argmax (greedy) and once sampling from the masked
softmax (~on-policy), records within-fight gamma-0.97 returns, and fits a fresh
head to each. High sampling R2 -> value CAN work on concentrated fights ->
fight-focused training is the path. Low -> fights are low-SNR -> deck lever."""
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

    def logits(self, x, mask):
        h = x
        for W, b in self.layers:
            h = gelu(h @ W.T + b)
        lg = h @ self.pi[0].T + self.pi[1]
        return np.where(mask > 0, lg, -1e9)

    def act(self, x, mask, rng, sample):
        lg = self.logits(x, mask)
        if not sample:
            return int(np.argmax(lg))
        lg = lg - lg.max()
        p = np.exp(lg); p /= p.sum()
        return int(rng.choice(len(p), p=p))


def collect(sample, seed):
    pol = NpPolicy("policy_varE.npz")
    bank = [json.loads(l) for l in open("bank_train.jsonl") if l.strip()]
    rng = np.random.default_rng(seed)
    env = FightEnv(ascension=0)
    maxima = np.maximum(np.asarray(env.nn.getObservationMaximums(), np.float32), 1.0)
    X, Y = [], []
    fights = wins = 0
    while fights < TARGET_FIGHTS:
        e = bank[int(rng.integers(len(bank)))]
        if env.reset_to_entry(e) is None:
            continue
        xs, rs = [], []
        steps = 0
        last_hp = env.bc.player.cur_hp
        won = False
        while steps < MAX_FIGHT_STEPS:
            obs = env._get_observation()
            o1200 = np.asarray(obs, np.float32) / maxima
            o = np.concatenate([o1200, combat_block(env)])
            a = pol.act(o1200, np.asarray(env._get_info()["action_mask"]),
                        rng, sample)
            xs.append(o.astype(np.float16))
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            cur = env.bc.player.cur_hp if (env.in_combat and env.bc) else env.gc.cur_hp
            rew = (cur - last_hp) * 1.0
            last_hp = cur
            if term or not env.in_combat:
                if env.gc.outcome == _LOSS:
                    rew += LOSS_PENALTY
                elif not env.in_combat:
                    rew += WIN_BONUS; won = True
                rs.append(rew * RS); break
            rs.append(rew * RS)
        G = 0.0; gs = [0.0] * len(rs)
        for t in reversed(range(len(rs))):
            G = rs[t] + GAMMA * G; gs[t] = G
        X.extend(xs); Y.extend(gs)
        fights += 1; wins += won
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), wins / fights


def best_r2(X, Y):
    n = len(Y); cut = int(n * 0.8)
    perm = np.random.RandomState(0).permutation(n)
    tr, va = perm[:cut], perm[cut:]
    Xt, Yt = torch.as_tensor(X[tr]), torch.as_tensor(Y[tr])
    Xv, Yv = torch.as_tensor(X[va]), torch.as_tensor(Y[va])
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(),
                        nn.Linear(512, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    m = len(Xt); best = -9.0
    for ep in range(60):
        pp = torch.randperm(m)
        for s in range(0, m, 512):
            j = pp[s:s + 512]
            loss = ((net(Xt[j]).squeeze(-1) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            best = max(best, float(1 - (Yv - net(Xv).squeeze(-1)).var() / (Yv.var() + 1e-8)))
    return best


for sample, name in [(False, "GREEDY"), (True, "SAMPLING")]:
    X, Y, wr = collect(sample, 0)
    print("%-9s | fights win rate %.2f | return sd %.3f | best val R2 %.3f"
          % (name, wr, Y.std(), best_r2(X, Y)), flush=True)
print("DONE")
