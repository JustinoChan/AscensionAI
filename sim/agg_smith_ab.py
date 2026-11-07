"""A/B aggregator for the smith-targeting lever. Reports the gate metrics side
by side. argv: [1] A jsonl (SMITH_TARGET=0)  [2] B jsonl (SMITH_TARGET=1)"""
import sys
import json
import numpy as np


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def stats(rows):
    n = len(rows)
    S = D = 0
    for r in rows:
        b = r.get("fb", {}).get("2_MONSTER")
        if b:
            S += b[0]
            D += b[1]
    bs = sum(r.get("boss_entry_hp_sum", 0) for r in rows)
    bn = sum(r.get("boss_entry_n", 0) for r in rows)
    decks = [r["act2_deck"] for r in rows if r.get("act2_deck")]
    return dict(
        n=n, wins=sum(r["win"] for r in rows),
        floor=np.mean([r["floor"] for r in rows]),
        a2m_deaths=sum(1 for r in rows if not r["win"]
                       and r.get("death_act") == 2
                       and "MONSTER" in str(r.get("death_room", ""))),
        a1_deaths=sum(1 for r in rows if not r["win"]
                      and r.get("death_act") == 1),
        a2m_wr=(S / (S + D) if (S + D) else 0.0),
        boss_hp=(bs / bn if bn else 0.0),
        upg=(np.mean([d["upgraded"] for d in decks]) if decks else 0.0),
        upgt=(np.mean([d.get("upg_tempo", 0) for d in decks]) if decks else 0.0),
        ndeck=len(decks))


A, B = stats(load(sys.argv[1])), stats(load(sys.argv[2]))
print("%-26s %12s %12s" % ("metric", "A target=0", "B target=1"))
print("-" * 52)
for label, k, fmt in [
        ("runs", "n", "%d"), ("wins", "wins", "%d"),
        ("avg floor", "floor", "%.2f"),
        ("Act-2 monster deaths", "a2m_deaths", "%d"),
        ("Act-2 monster win%", "a2m_wr", "%.3f"),
        ("Act-1 deaths", "a1_deaths", "%d"),
        ("boss-entry HP frac", "boss_hp", "%.3f"),
        ("upgraded @ Act-2", "upg", "%.2f"),
        ("upg_TEMPO @ Act-2", "upgt", "%.2f"),
        ("(reached Act 2)", "ndeck", "%d")]:
    print("%-26s %12s %12s" % (label, fmt % A[k], fmt % B[k]))
