"""Combine the original Act-1 fight bank with the new all-act harvest into
bank_train.jsonl. Reads the backed-up original (bank_train.act1.bak) so it is
idempotent. New (harvested) fights get lost=False (outcome unknown; Act-2/3
emphasis comes from the worker's act-oversampling, not loss-oversampling)."""
import json
import collections
old = [l.rstrip("\n") for l in open("bank_train.act1.bak") if l.strip()]
new = []
for l in open("fights_deep.jsonl"):
    if not l.strip():
        continue
    e = json.loads(l)
    e["lost"] = False
    new.append(json.dumps(e))
with open("bank_train.jsonl", "w") as f:
    for l in old:
        f.write(l + "\n")
    for l in new:
        f.write(l + "\n")
rows = [json.loads(l) for l in old] + [json.loads(l) for l in new]
c = collections.Counter((r.get("act", 1), r.get("room", "?")) for r in rows)
print("combined bank: %d old + %d new = %d entries" % (len(old), len(new), len(rows)))
for k in sorted(c):
    print("   act %d %-8s %d" % (k[0], k[1], c[k]))
