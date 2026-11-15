import json, glob, math, collections
R = [r for r in (json.load(open(f)) for f in glob.glob("sim_logs/boss_engine_gate/seed_*.json")) if "k64" in r]
n = len(R); names = ["k64", "mcts16", "mcts32", "mcts64"]
def mc(a, b):
    x = sum(1 for r in R if r[a]["won"] and not r[b]["won"]); y = sum(1 for r in R if r[b]["won"] and not r[a]["won"]); k = x + y
    return x, y, (min(1.0, 2 * sum(math.comb(k, i) for i in range(min(x, y) + 1)) / 2 ** k) if k else 1.0)
print("held-out Act 2/3 boss fights, identical starts, honest engines: n=%d" % n)
best, bw = "k64", sum(r["k64"]["won"] for r in R)
for m in names:
    w = sum(r[m]["won"] for r in R); hp = sum(r[m]["hp_end"] / r["max_hp"] for r in R) / max(1, n); t = sum(r[m]["secs"] for r in R) / max(1, n)
    line = "  %-7s won %3d (%.1f%%)  HP left %.3f  %.0f s/fight" % (m, w, 100.0 * w / max(1, n), hp, t)
    if m != "k64":
        x, y, p = mc("k64", m); line += "   vs k64: flips %d/%d p=%.3f" % (x, y, p)
        if w > bw and p < 0.25:
            best, bw = m, w
    print(line)
by = collections.defaultdict(lambda: collections.Counter())
for r in R:
    for m in names:
        by["A%d %s" % (r["act"], r["boss"])][m] += r[m]["won"]
    by["A%d %s" % (r["act"], r["boss"])]["n"] += 1
for k, c in sorted(by.items()):
    print("  %-22s n=%3d  " % (k, c["n"]) + "  ".join("%s %d" % (m, c[m]) for m in names))
print("BEST %s" % best)
