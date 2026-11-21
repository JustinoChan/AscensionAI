import json, glob, math, collections, sys
names = ["base", "c2", "c5", "c10", "c20", "c5roll"]
import sys as _s
names = [m for m in names if all(m in x for x in [json.load(open(f)) for d in _s.argv[1:] for f in glob.glob(d + "/seed_*.json")] if "base" in x)]
R = [r for f in sys.argv[1:] for r in (json.load(open(x)) for x in glob.glob(f + "/seed_*.json")) if "base" in r]
def mc(a, b):
    x = sum(1 for r in R if r[a]["won"] and not r[b]["won"]); y = sum(1 for r in R if r[b]["won"] and not r[a]["won"]); k = x + y
    return x, y, (min(1.0, 2 * sum(math.comb(k, i) for i in range(min(x, y) + 1)) / 2 ** k) if k else 1.0)
n = len(R)
print("held-out boss fights (%s), identical starts, honest K=64 root: n=%d" % (" + ".join(sys.argv[1:]), n))
best, bw = "base", sum(r["base"]["won"] for r in R)
for m in names:
    w = sum(r[m]["won"] for r in R); hp = sum(r[m]["hp_end"] / r["max_hp"] for r in R) / max(1, n)
    po = sum(r[m].get("pots_end", 0) for r in R) / max(1, n); t = sum(r[m]["secs"] for r in R) / max(1, n)
    line = "  %-7s won %3d (%.1f%%)  HP left %.3f  potions left %.2f  %.0f s/fight" % (m, w, 100.0 * w / max(1, n), hp, po, t)
    if m != "base":
        x, y, p = mc("base", m); line += "   vs base: flips %d/%d p=%.3f" % (x, y, p)
        if w > bw and p < 0.25:
            best, bw = m, w
    print(line)
by = collections.defaultdict(collections.Counter)
for r in R:
    k = "A%d %s" % (r["act"], r["boss"]); by[k]["n"] += 1
    for m in names:
        by[k][m] += r[m]["won"]
for k, c in sorted(by.items()):
    print("  %-22s n=%3d  " % (k, c["n"]) + "  ".join("%s %d" % (m, c[m]) for m in names))
print("BEST %s" % best)
