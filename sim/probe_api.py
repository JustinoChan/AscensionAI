"""Introspect the non-combat decision API: ScreenState members, GameContext
attributes, action-object and card-object fields, and which screens occur during
a varE greedy run. Output drives the v2 decision-logger design."""
import warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gym_sts_lightspeed  # noqa
import slaythespire
from sim_worker_fight import FightEnv


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class Pol:
    def __init__(self, p):
        w = np.load(p)
        self.L = [(w["body.%d.weight" % i], w["body.%d.bias" % i]) for i in (0, 2, 4)]
        self.pi = (w["pi.weight"], w["pi.bias"])

    def act(self, o, m):
        h = o
        for W, b in self.L:
            h = gelu(h @ W.T + b)
        lg = h @ self.pi[0].T + self.pi[1]
        return int(np.argmax(np.where(m > 0, lg, -1e9)))


print("=== ScreenState ===", [s for s in dir(slaythespire.ScreenState) if not s.startswith("_")])
print("=== slaythespire top ===", [x for x in dir(slaythespire) if not x.startswith("_")])

env = FightEnv(ascension=0)
maxima = np.maximum(np.asarray(env.nn.getObservationMaximums(), np.float32), 1.0)
pol = Pol("policy_varE.npz")
env.reset(seed=10007)
print("=== gc attrs ===", [a for a in dir(env.gc) if not a.startswith("_")])

seen = {}
dumped_action = False
dumped_card = False
for step in range(800):
    info = env._get_info()
    if not env.in_combat:
        ss = str(env.gc.screen_state).split(".")[-1]
        if ss not in seen:
            seen[ss] = 0
            acts = env.gc.get_possible_actions()
            print("\n--- SCREEN %s (floor %d room %s gold %d deck %d hp %d/%d) ---" % (
                ss, env.gc.floor_num, str(env.gc.cur_room).split(".")[-1],
                env.gc.gold, len(env.gc.deck), env.gc.cur_hp, env.gc.max_hp))
            for i, a in enumerate(acts[:14]):
                print("   act[%d]: %r" % (i, a))
            if not dumped_action and acts:
                print("   ACTION dir:", [x for x in dir(acts[0]) if not x.startswith("_")])
                dumped_action = True
            if not dumped_card and len(env.gc.deck) > 0:
                try:
                    c = env.gc.deck[0]
                    print("   CARD dir:", [x for x in dir(c) if not x.startswith("_")])
                    print("   CARD sample repr:", repr(c))
                    dumped_card = True
                except Exception as ex:
                    print("   card read fail:", ex)
        seen[ss] += 1
    obs = np.asarray(env._get_observation(), np.float32) / maxima
    a = pol.act(obs, np.asarray(info["action_mask"]))
    _, r, term, trunc, _ = env.step(a)
    if term or trunc:
        break

# relics / potions API
for attr in ("relics", "potions", "shop", "deck"):
    try:
        v = getattr(env.gc, attr)
        print("\n=== gc.%s ===" % attr, "len=%d" % len(v) if hasattr(v, "__len__") else type(v))
        if hasattr(v, "__len__") and len(v) > 0:
            print("   [0] dir:", [x for x in dir(v[0]) if not x.startswith("_")])
            print("   [0] repr:", repr(v[0]))
    except Exception as ex:
        print("gc.%s -> %s" % (attr, ex))

print("\n=== screens seen ===", seen)
