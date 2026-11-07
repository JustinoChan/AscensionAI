"""Phase-0 probe report: student-vs-search full-run comparison + per-bucket
fight outcomes from the fb analytics field ([fights, deaths, hp_lost, ...])."""
import glob, json, sys
import numpy as np


def load(d):
    out = []
    for p in glob.glob(d + "/seed_*.json"):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def summ(name, rs):
    wins = sum(r.get("win", 0) for r in rs)
    fl = np.mean([r.get("floor", 0) for r in rs])
    print("== %s  n=%d  wins %d  avg_floor %.2f  timeouts %d"
          % (name, len(rs), wins, fl, sum(r.get("timeout", 0) for r in rs)))
    agg = {}
    for r in rs:
        for k, v in (r.get("fb") or {}).items():
            a = agg.setdefault(k, [0, 0, 0.0])
            a[0] += v[0]; a[1] += v[1]; a[2] += v[2]
    for k in sorted(agg):
        f, dth, hp = agg[k]
        print("   %-10s fights %4d deaths %3d  hp_lost/fight %5.1f  death_rate %.3f"
              % (k, f, dth, hp / max(1, f), dth / max(1, f)))


summ("search (teacher)", load(sys.argv[1]))
summ("student (no search)", load(sys.argv[2]))
