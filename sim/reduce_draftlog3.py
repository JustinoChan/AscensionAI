"""Reduce per-seed teacher JSONs (raw-context draft records) into a jsonl:
{"seed", "roles", "scalars", "card", "value"} per line. Features are NOT
stored — trainers rebuild any feature set offline from the raw context."""
import json, glob, sys

out = sys.argv[1]
dirs = sys.argv[2:]
n_rows = n_runs = wins = bad = skipped_old = 0
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
                if "roles" not in rec:
                    skipped_old += 1               # pre-enrichment record
                    continue
                o.write(json.dumps({"seed": int(r["seed"]),
                                    "roles": rec["roles"],
                                    "scalars": rec["scalars"],
                                    "card": rec["card"],
                                    "value": rec["value"]}) + "\n")
                n_rows += 1
print("runs %d wins %d rows %d bad_files %d old_format_skipped %d -> %s"
      % (n_runs, wins, n_rows, bad, skipped_old, out))
