"""Debug: trace _smith_best's scan at the first smith upgrade screen."""
import sys, os, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa
import slaythespire  # noqa
from eval_fullrun_analytics_v2 import V2, SMITH


def _snap(gc):
    try:
        return [(str(c.id).split(".")[-1], str(c.type).split(".")[-1],
                 bool(c.upgraded)) for c in gc.deck]
    except Exception as e:
        return ("ERR", str(e))


class Dbg(V2):
    _did = 0

    def _macro_action(self):
        gc = self.env.gc
        ss = str(gc.screen_state).split(".")[-1]
        if SMITH and ss == "CARD_SELECT" and self._did < 2:
            acts = gc.get_possible_actions()
            n = len(acts)
            before = _snap(gc)
            if isinstance(before, list):
                bmult = {}
                for nm, tp, up in before:
                    if not up:
                        bmult[(nm, tp)] = bmult.get((nm, tp), 0) + 1
                # is this a smith (upgrade) screen? check action 0 raises upg
                bu = self._upg()
                print("=== smith screen: n=%d  floor=%d  act=%d ===" % (
                    n, int(gc.floor_num), int(gc.act)), flush=True)
                for i in range(min(n, 30)):
                    ro = self._restore()
                    if not ro:
                        print("  i=%2d RESTORE_FAILED" % i, flush=True)
                        continue
                    try:
                        self._ff_step(i)
                    except Exception as e:
                        print("  i=%2d ff_step EXC %s" % (i, e), flush=True)
                        continue
                    after = _snap(self.env.gc)
                    if not isinstance(after, list):
                        print("  i=%2d deck_snap %s" % (i, after), flush=True)
                        continue
                    amult = {}
                    for nm, tp, up in after:
                        if not up:
                            amult[(nm, tp)] = amult.get((nm, tp), 0) + 1
                    card = next((k for k in bmult if amult.get(k, 0) < bmult[k]), None)
                    print("  i=%2d -> %s  score=%s" % (
                        i, card, self._smith_score(*card) if card else "-"),
                        flush=True)
                self._restore()
                self._did += 1
                if self._did >= 2:
                    os._exit(0)
        return super()._macro_action()


if __name__ == "__main__":
    seed = int(sys.argv[1])
    ev = Dbg(os.environ.get("COMBAT_NPZ", "policy_varN.npz"), "policy_varE.npz", "macro")
    ev.play_v2(seed)
