"""Aggregate the PAIRED smith-target audit: on each smith screen, compare what
upg[0] (current mainline_v2) upgrades vs what the new _smith_best scorer picks.
Proves the scorer actually selects premium tempo cards before the full A/B.
argv: [1] audit_smith.jsonl"""
import sys
import json
from collections import Counter
from card_roles import classify, FRONTLOAD

# higher = better tempo target
RANK = {"0 premium attack": 6, "1 other attack": 5, "2 block": 4, "3 aoe": 3,
        "4 draw/energy": 2, "6 scaling/power": 1, "5 starter-filler": 0,
        "5 junk": 0, "7 other": 0}


def tier(name, ctype):
    if name in ("STRIKE_RED", "DEFEND_RED"):
        return "5 starter-filler"
    r = classify(name, ctype)
    if "junk" in r:
        return "5 junk"
    if name in FRONTLOAD:
        return "0 premium attack"
    if ctype == "ATTACK":
        return "1 other attack"
    if "block" in r:
        return "2 block"
    if "aoe" in r:
        return "3 aoe"
    if ("draw" in r) or ("energy" in r):
        return "4 draw/energy"
    if "scaling" in r or ctype == "POWER":
        return "6 scaling/power"
    return "7 other"


def is_tempo(name, ctype):
    return RANK.get(tier(name, ctype), 0) >= 4   # attack or strong block


rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
events = [e for r in rows for e in r.get("smith_log", [])]
print("runs: %d   smith events: %d" % (len(rows), len(events)))


def summarize(field):
    have = [e[field] for e in events if e.get(field)]
    tempo = sum(is_tempo(c[0], c[1]) for c in have)
    tiers = Counter(tier(c[0], c[1]) for c in have)
    names = Counter(c[0] for c in have)
    print("\n--- %s (n=%d) ---" % (field, len(have)))
    print("  IS premium tempo : %3.0f%% (%d/%d)"
          % (100 * tempo / max(1, len(have)), tempo, len(have)))
    for t, c in sorted(tiers.items()):
        print("    %-18s %3d (%2.0f%%)" % (t, c, 100 * c / max(1, len(have))))
    print("  top: " + ", ".join("%s x%d" % (n, c) for n, c in names.most_common(8)))


summarize("chosen")
summarize("scorer")

improved = worse = same = 0
for e in events:
    c, s = e.get("chosen"), e.get("scorer")
    if not c or not s:
        continue
    rc, rs = RANK.get(tier(*c), 0), RANK.get(tier(*s), 0)
    if rs > rc:
        improved += 1
    elif rs < rc:
        worse += 1
    else:
        same += 1
tot = max(1, improved + worse + same)
print("\nscorer vs upg[0] (paired, same screens):")
print("  improved tempo tier : %3.0f%% (%d)" % (100 * improved / tot, improved))
print("  same tier           : %3.0f%% (%d)" % (100 * same / tot, same))
print("  worse tier          : %3.0f%% (%d)" % (100 * worse / tot, worse))
