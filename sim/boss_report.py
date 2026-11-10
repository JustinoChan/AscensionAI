"""Per-boss / per-elite-encounter win rates for two traced eval500 arms (same seeds)."""
import collections, glob, json, sys
def fights(arm):
    d = collections.defaultdict(lambda: [0, 0, 0])   # met, won, hp lost
    for f in glob.glob("sim_logs/honest/%s/seed_*.json" % arm):
        for x in json.load(open(f)).get("trace", {}).get("fights", []):
            if x["room"] in ("BOSS", "ELITE"):
                k = ("A%d %s %s" % (x["act"], x["room"], x["enc"]))
                d[k][0] += 1; d[k][1] += int(not x["died"]); d[k][2] += x.get("hp_lost", 0)
    return d
a, b = sys.argv[1], sys.argv[2]
A, B = fights(a), fights(b)
print("\n%-38s %-24s %-24s" % ("fight", a + " won/met (HP)", b + " won/met (HP)"))
for k in sorted(set(A) | set(B), key=lambda k: (k.split()[0], k.split()[1] != "BOSS", k)):
    fa, fb = A.get(k, [0, 0, 0]), B.get(k, [0, 0, 0])
    f = lambda v: "%3d/%-3d %5.1f%% (%4.1f)" % (v[1], v[0], 100.0 * v[1] / v[0], v[2] / v[0]) if v[0] else "--"
    print("%-38s %-24s %-24s" % (k, f(fa), f(fb)))
