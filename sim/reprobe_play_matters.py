"""Decisive deck-bound test: on the 64-entry eval bank, compare fight win rate
under RANDOM play vs varE GREEDY play. If random ~= greedy -> fights are
deck/state-determined, play barely matters -> no value/policy fix will move win
rates -> the lever is DECK QUALITY. If greedy >> random -> play matters -> the
value-learning problem is worth solving. No state cloning needed."""
import sys
import json
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import slaythespire
from sim_worker_fight import FightEnv, MAX_FIGHT_STEPS

_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
evalbank = [json.loads(l) for l in open("bank_eval.jsonl") if l.strip()]


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
        lg = h @ self.pi[0].T + self.pi[1]
        return int(np.argmax(np.where(mask > 0, lg, -1e9)))


pol = NpPolicy("policy_varE.npz")
env = FightEnv(ascension=0)
maxima = np.maximum(np.asarray(env.nn.getObservationMaximums(), np.float32), 1.0)


def play(entry, mode, rng):
    if env.reset_to_entry(entry) is None:
        return None
    steps = 0
    while steps < MAX_FIGHT_STEPS:
        info = env._get_info()
        mask = np.asarray(info["action_mask"])
        legal = np.flatnonzero(mask)
        if mode == "random":
            a = int(rng.choice(legal))
        else:
            obs = env._get_observation()
            a = pol.act(np.asarray(obs, np.float32) / maxima, mask)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or not env.in_combat:
            return 0 if env.gc.outcome == _LOSS else 1
    return 0   # capped = not won


def winrates(mode, seeds):
    by = {"MONSTER": [], "ELITE": [], "BOSS": []}
    allw = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for e in evalbank:
            w = play(e, mode, rng)
            if w is None:
                continue
            by[e["room"]].append(w); allw.append(w)
    def m(x):
        return 100 * float(np.mean(x)) if x else 0.0
    return m(allw), m(by["MONSTER"]), m(by["ELITE"]), m(by["BOSS"])


g = winrates("greedy", [0])
r = winrates("random", [1, 2, 3, 4, 5])
print("GREEDY  | all %.0f%% monster %.0f%% elite %.0f%% boss %.0f%%" % g)
print("RANDOM  | all %.0f%% monster %.0f%% elite %.0f%% boss %.0f%% (avg of 5 seeds)" % r)
print("DELTA   | all %+.0f elite %+.0f boss %+.0f  (greedy - random)"
      % (g[0] - r[0], g[2] - r[2], g[3] - r[3]))
print("READ: large positive delta => PLAY MATTERS; ~0 => DECK-BOUND")
