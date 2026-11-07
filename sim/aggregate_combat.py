import sys, json
path = sys.argv[1] if len(sys.argv) > 1 else "v2_diag.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
agg = {}
for r in rows:
    for k, v in r.get("fb", {}).items():
        a = agg.setdefault(k, [0, 0, 0.0, 0, 0.0])
        for j in range(5):
            a[j] += v[j]
print("==== combat by (act, room): %s (%d runs) ====" % (path, len(rows)))
print("%-10s %8s %5s %6s %8s %7s %10s" %
      ("bucket", "survived", "died", "win%", "hp_lost", "turns", "ehp_left%"))


def order(k):
    act, room = k.split("_", 1)
    return (int(act), {"MONSTER": 0, "ELITE": 1, "BOSS": 2}.get(room, 3))


for k in sorted(agg, key=order):
    s, d, hpl, t, ehp = agg[k]
    tot = s + d
    print("%-10s %8d %5d %5.0f%% %8.1f %7.1f %9.0f%%" % (
        k, s, d, (100.0 * s / tot if tot else 0),
        (hpl / s if s else 0), (t / s if s else 0),
        (100.0 * ehp / d if d else 0)))
