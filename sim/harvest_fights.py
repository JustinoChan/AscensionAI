"""Harvest fight entries from the mainline search-combat + deck-hygiene agent
(which reaches Act 2/3). At each battle start, dump {seed, prefix, act, room,
floor, hp_frac} -- replayable to reconstruct that exact fight for combat-net
retraining. Mainline config: GATE_ELITES=0, RELIC_BUY=0 (set before import so
the module constants pick them up). One process per seed.

argv: [1] seed"""
import sys, json, warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["GATE_ELITES"] = "0"
os.environ["RELIC_BUY"] = "0"
import numpy as np  # noqa
from eval_fullrun_search import MAX_GAME_STEPS
from eval_fullrun_analytics_v2 import V2


class Harvester(V2):
    def harvest(self, seed):
        self.seed = seed
        self.history = []
        self._avoided = 0
        self._shop_removed_floor = -1
        self.env.reset(seed=int(seed))
        prev_ss = str(self.env.gc.screen_state).split(".")[-1]
        out = []
        for _ in range(MAX_GAME_STEPS):
            cur_ss = str(self.env.gc.screen_state).split(".")[-1]
            if cur_ss != prev_ss:
                if cur_ss == "BATTLE":
                    gc = self.env.gc
                    out.append({"seed": int(seed), "prefix": list(self.history),
                                "act": int(gc.act),
                                "room": str(gc.cur_room).split(".")[-1],
                                "floor": int(gc.floor_num),
                                "hp_frac": round(gc.cur_hp / max(1, gc.max_hp), 3)})
                prev_ss = cur_ss
            if self.env.in_combat:
                a = self._search_action()
                if a is None:
                    break
            else:
                a = self._macro_action()
            _, r, term, trunc, _ = self.env.step(int(a))
            self.history.append(int(a))
            if term or trunc:
                break
        return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    h = Harvester("policy_varL.npz", "policy_varE.npz", "macro")
    for e in h.harvest(seed):
        print(json.dumps(e), flush=True)
