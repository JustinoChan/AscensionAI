import sys, json
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else "v2_roles.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
a2 = [r["act2_deck"] for r in rows if r.get("act2_deck")]
n = len(a2)
print("==== Act-2-entry deck roles: %s (%d runs, %d reached Act 2) ====" % (path, len(rows), n))
if not a2:
    print("no Act-2 decks captured"); sys.exit()
def m(k):
    return np.mean([d[k] for d in a2])
print("avg deck size:      %.1f" % m("size"))
print("starters remaining: %.1f" % m("starters"))
print("upgraded:           %.1f" % m("upgraded"))
print("damage (attacks):   %.1f" % m("damage"))
print("block cards:        %.1f" % m("block"))
print("AoE cards:          %.2f" % m("aoe"))
print("scaling cards:      %.1f" % m("scaling"))
print("draw/energy cards:  %.2f" % m("draw_energy"))
print("powers:             %.1f" % m("powers"))
print("junk cards:         %.2f" % m("junk"))
print("avg card quality:   %.2f" % np.mean([d["qsum"] / max(1, d["size"]) for d in a2]))
print("damage:block ratio: %.2f" % (m("damage") / max(0.1, m("block"))))
print("-- %% of Act-2 decks with ZERO of a role --")
for k in ("aoe", "scaling", "draw_energy", "powers"):
    z = 100.0 * sum(1 for d in a2 if d[k] == 0) / n
    print("   no %-12s %.0f%%" % (k, z))
