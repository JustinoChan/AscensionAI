"""Crash-tolerant pure-fight collection under a SAMPLING policy. Dumps every
20 fights to <dir>/bNNNN.npz; a wrapper restarts on native segfault and this
resumes by counting existing files. argv: [dir] [target_fights]."""
import sys
import os
import glob
import json
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import slaythespire
from sim_worker_fight import FightEnv, MAX_FIGHT_STEPS
from combat_features import combat_block

OUT = sys.argv[1] if len(sys.argv) > 1 else "samp_dumps"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
os.makedirs(OUT, exist_ok=True)
GAMMA = 0.97
WIN_BONUS, LOSS_PENALTY = 50.0, -50.0
RS = 0.1
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpPolicy:
    def __init__(self, p):
        w = np.load(p)
        self.layers = [(w["body.%d.weight" % i], w["body.%d.bias" % i])
                       for i in (0, 2, 4)]
        self.pi = (w["pi.weight"], w["pi.bias"])

    def sample(self, x, mask, rng):
        h = x
        for W, b in self.layers:
            h = gelu(h @ W.T + b)
        lg = h @ self.pi[0].T + self.pi[1]
        lg = np.where(mask > 0, lg, -1e9)
        lg = lg - lg.max()
        p = np.exp(lg); p /= p.sum()
        return int(rng.choice(len(p), p=p))


existing = glob.glob(os.path.join(OUT, "b*.npz"))
fidx = len(existing)
done_fights = fidx * 20
if done_fights >= TARGET:
    print("already have %d fights" % done_fights)
    raise SystemExit(0)

pol = NpPolicy("policy_varE.npz")
bank = [json.loads(l) for l in open("bank_train.jsonl") if l.strip()]
# seed RNG off the resume point so restarts explore fresh fights
rng = np.random.default_rng(1234 + fidx)
env = FightEnv(ascension=0)
maxima = np.maximum(np.asarray(env.nn.getObservationMaximums(), np.float32), 1.0)

accX, accY = [], []
batch = 0
while done_fights < TARGET:
    e = bank[int(rng.integers(len(bank)))]
    if env.reset_to_entry(e) is None:
        continue
    xs, rs = [], []
    steps = 0
    last_hp = env.bc.player.cur_hp
    while steps < MAX_FIGHT_STEPS:
        obs = env._get_observation()
        o1200 = np.asarray(obs, np.float32) / maxima
        o = np.concatenate([o1200, combat_block(env)])
        a = pol.sample(o1200, np.asarray(env._get_info()["action_mask"]), rng)
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
                rew += WIN_BONUS
            rs.append(rew * RS); break
        rs.append(rew * RS)
    G = 0.0; gs = [0.0] * len(rs)
    for t in reversed(range(len(rs))):
        G = rs[t] + GAMMA * G; gs[t] = G
    accX.append(np.asarray(xs, np.float16)); accY.append(np.asarray(gs, np.float32))
    done_fights += 1; batch += 1
    if batch >= 20:
        fidx += 1
        np.savez_compressed(os.path.join(OUT, "b%04d.npz" % fidx),
                            obs=np.concatenate(accX), ret=np.concatenate(accY))
        accX, accY = [], []
        batch = 0
print("DONE %d fights" % done_fights)
