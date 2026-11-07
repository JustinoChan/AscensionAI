"""Potion audit report over sim_logs/honest/potion_audit_K8 (honest K=8)."""
import glob
import json
from collections import defaultdict

D = "sim_logs/honest/potion_audit_K8"
rows = [json.load(open(p)) for p in glob.glob(D + "/seed_*.json")]
ref = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob("sim_logs/honest/combat_K8/seed_*.json")}
same = sum(1 for r in rows if ref.get(r["seed"]) and {k: v for k, v in r.items() if k not in ("secs", "potion_audit")} ==
           {k: v for k, v in ref[r["seed"]].items() if k != "secs"})
print("runs %d  wins %d  avg floor %.2f  | decisions identical to combat_K8 on %d/%d seeds" % (
    len(rows), sum(r["win"] for r in rows), sum(r["floor"] for r in rows) / len(rows), same, len(rows)))
by = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # fights, held_start, used, fights_with_use, deaths, held_at_death
for r in rows:
    for f in r.get("potion_audit", []):
        k = f["room"]
        b = by[k]
        b[0] += 1
        b[1] += f["pots_start"]
        b[2] += f["used"]
        b[3] += int(f["used"] > 0)
        if f.get("died"):
            b[4] += 1
            b[5] += f["pots_end"]
print("\n%-9s %7s %13s %11s %14s %7s %18s %16s" % ("room", "fights", "held@start/f", "used/fight", "fights w/ use",
                                                   "deaths", "held@death/death", "deaths holding>=1"))
for k in ("MONSTER", "ELITE", "BOSS", "EVENT"):
    if k not in by:
        continue
    b = by[k]
    dh = sum(1 for r in rows for f in r.get("potion_audit", []) if f["room"] == k and f.get("died") and f["pots_end"] > 0)
    print("%-9s %7d %13.2f %11.2f %13.1f%% %7d %18.2f %16s" % (
        k, b[0], b[1] / b[0], b[2] / b[0], 100.0 * b[3] / b[0], b[4], (b[5] / b[4]) if b[4] else 0.0,
        "%d/%d" % (dh, b[4])))
tot_used = sum(b[2] for b in by.values())
tot_f = sum(b[0] for b in by.values())
deaths = [f for r in rows for f in r.get("potion_audit", []) if f.get("died")]
print("\nall fights %d, potions used %d (%.2f/fight); deaths %d, %d of them holding unused potions (%.2f held per death)" % (
    tot_f, tot_used, tot_used / tot_f, len(deaths), sum(1 for f in deaths if f["pots_end"] > 0),
    sum(f["pots_end"] for f in deaths) / max(1, len(deaths))))
full = sum(1 for r in rows for f in r.get("potion_audit", []) if f["pots_start"] >= 3)
print("fights started with all 3 slots full: %d (%.1f%%) -> potion rewards wasted" % (full, 100.0 * full / tot_f))
