import json, glob, sys

outdir = sys.argv[1]
rows, recs = [], []
for f in sorted(glob.glob(outdir + "/seed_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append(d)
    recs.extend(d.get("draft_records", []))
wins = sum(int(r.get("win", 0)) for r in rows)
fl = sum(r.get("floor", 0) for r in rows) / max(1, len(rows))
to = sum(int(r.get("timeout", 0)) for r in rows)
print("runs %d wins %d avg_floor %.2f timeouts %d" % (len(rows), wins, round(fl, 2), to))
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as o:
        for r in recs:
            o.write(json.dumps(r) + "\n")
    print("wrote %d records -> %s" % (len(recs), sys.argv[2]))
