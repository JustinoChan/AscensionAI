"""Split Act-2 outcomes by starter count at Act-2 entry (from existing run data).
Answers: do decks with fewer un-removed starters actually perform better in
Act 2? (Correlational -- a signal, not proof of causation.)"""
import sys, json
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1
                                    else "v2_mainline_v2.jsonl") if l.strip()]
a2 = [r for r in rows if r.get("act2_deck")]


def bucket(s):
    if s <= 5:
        return "0-5"
    if s <= 7:
        return "6-7"
    if s <= 9:
        return "8-9"
    return "10+"


bk = {}
for r in a2:
    bk.setdefault(bucket(r["act2_deck"]["starters"]), []).append(r)
print("==== Act-2 outcomes by starter count @ Act-2 entry "
      "(%d of %d runs reached Act 2) ====" % (len(a2), len(rows)))
print("%-5s %5s %6s %7s %13s %14s %8s" %
      ("start", "runs", "wins", "floor", "a2mon_death%", "a2mon_ehp_left", "deck"))
for b in ("0-5", "6-7", "8-9", "10+"):
    grp = bk.get(b, [])
    if not grp:
        continue
    wins = sum(r["win"] for r in grp)
    floor = np.mean([r["floor"] for r in grp])
    deck = np.mean([r["act2_deck"]["size"] for r in grp])
    surv = died = ehp_n = 0
    ehp = 0.0
    for r in grp:
        m = r.get("fb", {}).get("2_MONSTER")
        if m:
            surv += m[0]; died += m[1]; ehp += m[4]; ehp_n += m[1]
    dr = 100.0 * died / max(1, surv + died)
    print("%-5s %5d %6d %7.1f %12.0f%% %14.2f %8.1f" %
          (b, len(grp), wins, floor, dr, ehp / max(1, ehp_n), deck))
