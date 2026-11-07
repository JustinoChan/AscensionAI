"""Aggregate the combat/search v2 recon. The decisive question is WHERE in an
Act-2 monster fight the loss becomes inevitable:

  - entry Q (df[0] best rollout value): if already deeply negative, the rollout
    teacher predicts a loss from the FIRST move of the fight => the fight is lost
    on arrival (entry state: too little HP / too weak a deck) => deeper in-fight
    search cannot save it. If positive-at-entry but lost => in-fight play threw a
    winnable fight => search/teacher v2 could help.
  - entry HP frac: low => upstream attrition / no heal before the fight.
  - last-real-choice Q-gap (last decision with >=2 candidates): small => the
    rollout barely separated lines AT A POINT WHERE THERE WAS STILL A CHOICE.
  - death-turn lethal flag: net incoming >= HP at the final decision.
argv: [1] diag_act2.jsonl"""
import sys
import json
import numpy as np

LOSS_Q = -10.0   # bestq below this ~ rollout predicts a loss from this state

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
wins = sum(1 for r in rows if r["win"])
deaths = [r for r in rows if r.get("death_fight")]
a2m = [r for r in rows if not r["win"] and r.get("death_act") == 2
       and "MONSTER" in str(r.get("death_room", ""))]

print("runs: %d   wins: %d   floor avg: %.2f" % (
    len(rows), wins, np.mean([r["floor"] for r in rows])))
print("A2-monster deaths (by outcome): %d   (with decision log: %d)" % (
    len(a2m), len(deaths)))


def bestq(rec):
    t3 = rec.get("top3") or []
    return t3[0] if t3 else None


entry_fracs, entry_qs, choice_gaps, fight_len, enemy_fracs = [], [], [], [], []
lost_at_entry = death_lethal = 0
NEAR_WIN, LOW_HP = 0.20, 0.40   # enemy <=20% HP at death; entered <=40% HP
xtab = {"A_lowHP_nearwin": 0, "B_healthy_nearwin": 0,
        "C_lowHP_crushed": 0, "D_healthy_crushed": 0}
for r in deaths:
    df = r["death_fight"]
    entry, last = df[0], df[-1]
    ef = entry["php"] / max(1, entry["mhp"])
    entry_fracs.append(ef)
    eq = bestq(entry)
    if eq is not None:
        entry_qs.append(eq)
        lost_at_entry += int(eq < LOSS_Q)
    fight_len.append(len(df))
    death_lethal += int(bool(last.get("lethal")))
    # last decision that still had a real choice (>=2 candidates)
    cg = next((d["gap"] for d in reversed(df)
               if d.get("ncand", 1) >= 2 and d.get("gap") is not None), None)
    if cg is not None:
        choice_gaps.append(cg)
    # 2x2 lever-attribution cross-tab
    enemy_frac = float(r.get("enemy_hp_frac", 1.0))
    enemy_fracs.append(enemy_frac)
    low, near = ef <= LOW_HP, enemy_frac <= NEAR_WIN
    xtab[("A_lowHP_nearwin" if (low and near) else
          "B_healthy_nearwin" if near else
          "C_lowHP_crushed" if low else "D_healthy_crushed")] += 1


def pct(a):
    a = np.asarray(a, float)
    if not len(a):
        return "n=0"
    return ("n=%d  p10=%.2f p25=%.2f med=%.2f p75=%.2f p90=%.2f mean=%.2f"
            % (len(a), np.percentile(a, 10), np.percentile(a, 25),
               np.median(a), np.percentile(a, 75), np.percentile(a, 90),
               a.mean()))


nd = max(1, len(deaths))
print("\nentry HP frac       : %s" % pct(entry_fracs))
print("entry best-Q        : %s" % pct(entry_qs))
print("  -> lost-on-arrival : %.0f%% (%d/%d)  (entry best-Q < %.0f => fight lost"
      " from move 1)" % (100 * lost_at_entry / nd, lost_at_entry, len(deaths),
                         LOSS_Q))
print("enemy HP frac @death: %s  (low => we nearly won)" % pct(enemy_fracs))
print("last real-choice gap: %s  (where there was still a >=2-way choice)"
      % pct(choice_gaps))
print("fight length (#dec) : %s" % pct(fight_len))
print("death-turn lethal   : %.0f%% (%d/%d)" % (100 * death_lethal / nd,
                                                death_lethal, len(deaths)))
print("\nlever-attribution cross-tab (entered <=%.0f%% HP  x  enemy <=%.0f%% @death):"
      % (100 * LOW_HP, 100 * NEAR_WIN))
print("  A entered-low  & near-win  : %2d  -> HP-preservation upstream lever"
      % xtab["A_lowHP_nearwin"])
print("  B entered-OK   & near-win  : %2d  -> threw a winnable fight => search-v2"
      % xtab["B_healthy_nearwin"])
print("  C entered-low  & crushed   : %2d  -> arrived too low AND outmatched"
      % xtab["C_lowHP_crushed"])
print("  D entered-OK   & crushed   : %2d  -> outmatched at full HP => deck/combat"
      % xtab["D_healthy_crushed"])

print("\n--- sample death fights (entry + last 3 decisions) ---")
for r in deaths[:10]:
    df = r["death_fight"]
    print("seed %d  floor %d %s  (%d decisions, entry %d/%d HP):" % (
        r["seed"], df[-1]["floor"], df[-1]["room"], len(df),
        df[0]["php"], df[0]["mhp"]))
    show = [df[0]] + df[-3:] if len(df) > 4 else df
    for d in show:
        print("   php %3d/%3d blk %2d | inc %5.1f net %5.1f leth %d | nalive %d "
              "ehp %3d | ncand %d gap %s top3 %s" % (
                  d["php"], d["mhp"], d["pblk"], d["inc"], d["net"], d["lethal"],
                  d["nalive"], d["ehp"], d.get("ncand", -1), d["gap"], d["top3"]))
