"""AUDIT (read-only): what does the current mainline_v2 smith actually upgrade?
Subclasses V2; at every smith upgrade screen it logs the card upg[0] picks (the
deployed behavior, unchanged) plus all available upgrade targets. Identifies the
chosen card by diffing gc.deck across a scratch-step (guarded; falls back to
skipping the log on read failure). Behavior is identical to mainline_v2 -- the
override only observes, then delegates to super()._macro_action().

argv: [1] seed   (mode always macro; toggles via env, same as mainline_v2)"""
import sys
import json
import os
import warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: F401,E402
import slaythespire  # noqa: F401,E402
from eval_fullrun_analytics_v2 import V2, SMITH  # noqa: E402
from card_roles import classify  # noqa: E402

NONUP = ("CURSE", "STATUS")


def _deck_snap(gc):
    try:
        return [(str(c.id).split(".")[-1], str(c.type).split(".")[-1],
                 bool(c.upgraded)) for c in gc.deck]
    except Exception:
        return None


class V2Audit(V2):
    def _identify(self, i, bmult):
        """Card (name,type) that action i upgrades, via gc.deck diff. None on fail."""
        if not self._restore():
            return None
        try:
            self._ff_step(i)
        except Exception:
            return None
        after = _deck_snap(self.env.gc)
        if not after:
            return None
        amult = {}
        for nm, tp, up in after:
            if not up:
                amult[(nm, tp)] = amult.get((nm, tp), 0) + 1
        card = next((k for k in bmult if amult.get(k, 0) < bmult[k]), None)
        return list(card) if card else None

    def _maybe_log_smith(self):
        gc = self.env.gc
        n = len(gc.get_possible_actions())
        if n <= 1:
            return
        before = _deck_snap(gc)
        if before is None:
            return
        bmult = {}
        for nm, tp, up in before:
            if not up:
                bmult[(nm, tp)] = bmult.get((nm, tp), 0) + 1
        chosen = None                                # upg[0] = first ud>=1 action
        for i in range(min(n, 12)):
            c = self._identify(i, bmult)
            if c is not None:
                chosen = c
                break
        if chosen is None:
            self._restore()
            return
        self._restore()                       # back to LIVE upgrade screen (prod path)
        scorer = self._identify(self._smith_best(), bmult)   # new scorer's pick
        self._restore()
        avail = [[nm, tp] for (nm, tp, up) in before
                 if not up and not any(x in tp.upper() for x in NONUP)]
        self._smith_log.append(dict(
            floor=int(gc.floor_num), act=int(gc.act),
            chosen=chosen, scorer=scorer, available=avail))

    def _macro_action(self):
        ss = str(self.env.gc.screen_state).split(".")[-1]
        if SMITH and ss == "CARD_SELECT":
            try:
                self._maybe_log_smith()
            except Exception:
                pass
        return super()._macro_action()

    def play_audit(self, seed):
        self._smith_log = []
        out = self.play_v2(seed)
        out["smith_log"] = self._smith_log
        return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    combat_npz = os.environ.get("COMBAT_NPZ", "policy_varN.npz")
    ev = V2Audit(combat_npz, "policy_varE.npz", "macro")
    print(json.dumps(ev.play_audit(seed)), flush=True)
