"""Reduce per-seed teacher JSONs from one or more dirs into a seed-tagged
draft-decision jsonl: {"seed", "feat", "value"} per line. Seed tags enable
leakage-free seed-level train/val splits."""
import json, glob, sys

out = sys.argv[1]
dirs = sys.argv[2:]
n_rows = n_runs = wins = bad = 0
with open(out, "w") as o:
    for d in dirs:
        for f in sorted(glob.glob(d + "/seed_*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                bad += 1
                continue
            n_runs += 1
            wins += int(r.get("win", 0))
            for rec in r.get("draft_records", []):
                o.write(json.dumps({"seed": int(r["seed"]),
                                    "feat": rec["feat"],
                                    "value": rec["value"]}) + "\n")
                n_rows += 1
print("runs %d wins %d rows %d bad_files %d -> %s"
      % (n_runs, wins, n_rows, bad, out))
