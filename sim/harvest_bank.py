"""Harvest fight-entry states from greedy full runs of a NumPy-exported
policy. Single process, NO torch (avoids the in-process torch<->extension
conflict). One JSONL entry per fight, flushed immediately; .harvest_progress
records the seed index being attempted so the wrapper can skip crash seeds.

argv: [1] policy .npz  [2] start idx  [3] end idx (excl)  [4] out path
Seeds are 1_000_000 + idx -- disjoint from EVAL_SEEDS (all < 400k)."""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire

OBS = 1200
MAX_GAME_STEPS = 3000
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS


class SafeStsEnv(StsEnv):
    _last = None

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last = o
            return o
        except Exception:
            return self._last if self._last is not None else np.zeros(OBS, np.int32)


def gelu(x):
    # tanh approximation; the tiny deviation vs torch's exact GELU is fine --
    # harvested states only need to come from a competent policy, and replay
    # uses the recorded actions, not this network.
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpPolicy:
    def __init__(self, npz_path):
        w = np.load(npz_path)
        self.layers = [(w["body.%d.weight" % i], w["body.%d.bias" % i])
                       for i in (0, 2, 4)]
        self.pi = (w["pi.weight"], w["pi.bias"])

    def act(self, obs_norm, mask):
        h = obs_norm
        for W, b in self.layers:
            h = gelu(h @ W.T + b)
        logits = h @ self.pi[0].T + self.pi[1]
        logits = np.where(mask > 0, logits, -1e9)
        return int(np.argmax(logits))


def main():
    npz, lo, hi, out_path = (sys.argv[1], int(sys.argv[2]),
                             int(sys.argv[3]), sys.argv[4])
    pol = NpPolicy(npz)
    env = SafeStsEnv(ascension=0)
    maxima = np.maximum(
        np.asarray(env.nn.getObservationMaximums(), dtype=np.float32), 1.0)
    out = open(out_path, "a")
    n_entries = 0
    for idx in range(lo, hi):
        seed = 1_000_000 + idx
        with open(".harvest_progress", "w") as f:
            f.write(str(idx))
        obs, info = env.reset(seed=seed)
        actions = []
        was_in_combat = False
        entry = None
        for _ in range(MAX_GAME_STEPS):
            a = pol.act(np.asarray(obs, np.float32) / maxima,
                        np.asarray(info["action_mask"]))
            obs, r, term, trunc, info = env.step(a)
            actions.append(a)
            if env.in_combat and not was_in_combat:
                entry = {
                    "seed": seed,
                    "prefix": list(actions),
                    "floor": int(env.gc.floor_num),
                    "act": int(env.gc.act),
                    "hp_frac": round(env.gc.cur_hp / max(1, env.gc.max_hp), 4),
                    "room": str(env.gc.cur_room).split(".")[-1],
                    "entry_hp": int(env.gc.cur_hp),
                }
            elif was_in_combat and (not env.in_combat or term):
                if entry is not None:
                    died = bool(env.gc.outcome == _LOSS)
                    cur = 0 if died else int(env.gc.cur_hp)
                    entry["lost"] = died
                    entry["hp_lost"] = max(0, entry.pop("entry_hp") - cur)
                    out.write(json.dumps(entry) + "\n")
                    out.flush()
                    n_entries += 1
                    entry = None
            was_in_combat = bool(env.in_combat)
            if term or trunc:
                break
        print("seed %d done (%d entries total)" % (seed, n_entries), flush=True)
    out.close()
    print("HARVEST DONE %d entries" % n_entries)


if __name__ == "__main__":
    main()
