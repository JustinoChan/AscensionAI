import json, glob, collections, math
R = [json.load(open(f)) for f in glob.glob("sim_logs/boss_gate1/seed_*.json")]
R = [r for r in R if "base" in r]
n = len(R); bw = sum(r["base"]["won"] for r in R); nw = sum(r["new"]["won"] for r in R)
b_only = sum(1 for r in R if r["base"]["won"] and not r["new"]["won"]); n_only = sum(1 for r in R if r["new"]["won"] and not r["base"]["won"])
k = b_only + n_only
p = min(1.0, 2 * sum(math.comb(k, i) for i in range(min(b_only, n_only) + 1)) / 2 ** k) if k else 1.0
hp = lambda m: sum(r[m]["hp_end"] / r["max_hp"] for r in R) / max(1, n)
print("held-out boss fights inside honest K=64 search: n=%d" % n)
print("  base (varN rollouts): won %d (%.1f%%), mean HP left %.3f" % (bw, 100.0 * bw / max(1, n), hp("base")))
print("  new (boss-RL rollouts): won %d (%.1f%%), mean HP left %.3f" % (nw, 100.0 * nw / max(1, n), hp("new")))
print("  flips base-only %d new-only %d, exact McNemar p = %.3f" % (b_only, n_only, p))
by = collections.defaultdict(lambda: [0, 0, 0])
for r in R:
    k2 = "A%d %s" % (r["act"], r["boss"]); by[k2][0] += 1; by[k2][1] += r["base"]["won"]; by[k2][2] += r["new"]["won"]
for k2, v in sorted(by.items()):
    print("  %-22s n=%3d base %3d new %3d" % (k2, v[0], v[1], v[2]))
print("PASS" if (nw >= bw and hp("new") >= hp("base") - 0.01) else "FAIL")
