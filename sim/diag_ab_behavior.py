"""Aggregate behavioral diff between the A/B arms' per-seed JSONs."""
import glob
import json
import sys

import numpy as np


def load(d):
    out = []
    for p in glob.glob(d + "/seed_*.json"):
        with open(p) as f:
            out.append(json.load(f))
    return out


def summ(name, rs):
    def m(k, default=0):
        return np.mean([r.get(k, default) or 0 for r in rs])
    deck = [r.get("final_deck") or {} for r in rs]

    def dm(k):
        return np.mean([d.get(k, 0) for d in deck])
    acts = {}
    for r in rs:
        a = r.get("death_act", 0) if not r.get("win") else "win"
        acts[a] = acts.get(a, 0) + 1
    print(f"== {name}  n={len(rs)}")
    print(f"  wins {sum(r.get('win',0) for r in rs)}  floor {m('floor'):.2f}  death_act {dict(sorted(acts.items(), key=str))}")
    print(f"  taken {m('cards_taken'):.2f}  skipped {m('cards_skipped'):.2f}  screens {m('card_screens'):.2f}  removes {m('removes'):.2f}  smiths {m('smiths'):.2f}")
    print(f"  deck: size {dm('size'):.1f} strikes {dm('strikes'):.1f} defends {dm('defends'):.1f} attacks {dm('attacks'):.1f} powers {dm('powers'):.1f} upgraded {dm('upgraded'):.1f}")


summ("A (mainline_v3)", load(sys.argv[1]))
summ("B (+V_DRAFT)", load(sys.argv[2]))
