"""Pinpoint the Act-2 deck-throughput deficit from data already in diag_act2.jsonl.
Splits Act-2-ENTRY deck composition (act2_deck role counts) by outcome and
reports which deck property is most deficient in the losing (bucket-C) decks vs
decks that actually CLEARED Act 2 (reached Act 3 or won). Also correlates deck
damage/scaling with enemy-HP-left-at-death (the policy-independent throughput
proxy). argv: [1] diag_act2.jsonl"""
import sys
import json
import numpy as np

LOSS_Q, NEAR_WIN, LOW_HP = -10.0, 0.20, 0.40
FIELDS = ["size", "starters", "upgraded", "damage", "block", "aoe",
          "scaling", "draw_energy", "powers", "junk", "qsum"]

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]


def bucket(r):
    df = r.get("death_fight")
    if not df:
        return None
    ef = df[0]["php"] / max(1, df[0]["mhp"])
    enemy = float(r.get("enemy_hp_frac", 1.0))
    low, near = ef <= LOW_HP, enemy <= NEAR_WIN
    return ("A" if (low and near) else "B" if near else "C" if low else "D")


groups = {"WIN": [], "GOOD (cleared A2: act>=3 or win)": [],
          "ALL A2-monster deaths": [], "C bucket (low & crushed)": [],
          "D bucket (healthy & crushed)": []}
a2m_decks, a2m_enemy = [], []
for r in rows:
    d = r.get("act2_deck")
    if not d:
        continue
    if r["win"]:
        groups["WIN"].append(d)
    if r["win"] or r.get("act", 1) >= 3:
        groups["GOOD (cleared A2: act>=3 or win)"].append(d)
    b = bucket(r)
    if b:
        groups["ALL A2-monster deaths"].append(d)
        a2m_decks.append(d)
        a2m_enemy.append(float(r.get("enemy_hp_frac", 1.0)))
        if b == "C":
            groups["C bucket (low & crushed)"].append(d)
        if b == "D":
            groups["D bucket (healthy & crushed)"].append(d)


def mean(ds, f):
    v = [x.get(f, 0) for x in ds if x is not None]
    return float(np.mean(v)) if v else float("nan")


print("Act-2-ENTRY deck composition by outcome (mean):\n")
print("%-34s %3s " % ("group", "n")
      + " ".join("%6s" % f[:6] for f in FIELDS))
for g, ds in groups.items():
    print("%-34s %3d " % (g, len(ds))
          + " ".join("%6.2f" % mean(ds, f) for f in FIELDS))

good = groups["GOOD (cleared A2: act>=3 or win)"]
cb = groups["C bucket (low & crushed)"]
print("\nDEFICIT  (C bucket - GOOD)  negative => losing decks are weaker:")
deltas = []
for f in FIELDS:
    gm, cm = mean(good, f), mean(cb, f)
    deltas.append((f, gm, cm, cm - gm))
for f, gm, cm, dl in sorted(deltas, key=lambda x: x[3]):
    rel = (dl / gm * 100) if gm else 0.0
    print("  %-12s GOOD %6.2f   C %6.2f   delta %+6.2f  (%+5.0f%%)"
          % (f, gm, cm, dl, rel))

# throughput check: does more deck damage/scaling => less enemy HP left at death?
if len(a2m_decks) >= 5:
    enemy = np.asarray(a2m_enemy)
    print("\nThroughput correlation across the %d A2-monster deaths "
          "(corr with enemy-HP-left-at-death; negative = more of X => closer to"
          " a kill):" % len(a2m_decks))
    for f in ["damage", "scaling", "powers", "upgraded", "qsum"]:
        x = np.asarray([d.get(f, 0) for d in a2m_decks], float)
        c = np.corrcoef(x, enemy)[0, 1] if x.std() > 0 else float("nan")
        print("  %-12s corr=%+.2f" % (f, c))
