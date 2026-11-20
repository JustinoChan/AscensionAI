#!/bin/bash
# Combat headroom re-check under the current mainline (5d + PV10): replay eval500_rl5d_pv10 seeds; at every Act 2/3
# elite/boss fight play it honest K=64 vs the same 1-ply search on the exact state (clairvoyant, diagnostic only).
# Seeds chosen until ~TARGET Act 2/3 BOSS fights (bosses are ~2/3 of losses). Previous check (11-08, baseline policy):
# +1.3 win pts overall, +8.4 at Act 2 bosses (n=83).
cd "$HOME/sts"
TARGET=${TARGET:-200}; PAR=${PAR:-14}; export HEADROOM_OUT=sim_logs/headroom2
SEEDS=$(python3 - $TARGET <<'PY'
import json, glob, sys
t = int(sys.argv[1]); n = 0; out = []
for f in sorted(glob.glob("sim_logs/honest/eval500_rl5d_pv10/seed_*.json"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0])):
    r = json.load(open(f))
    k = sum(1 for x in r.get("trace", {}).get("fights", []) if x["act"] >= 2 and x["room"] == "BOSS")
    if k:
        out.append(r["seed"]); n += k
    if n >= t:
        break
print(" ".join(map(str, out)))
PY
)
echo "$(echo $SEEDS | wc -w) seeds"; mkdir -p $HEADROOM_OUT
echo $SEEDS | tr ' ' '\n' | xargs -P $PAR -I{} bash -c '[ -s $HEADROOM_OUT/seed_{}.json ] && exit 0; HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10 nice -n 5 .venv/bin/python headroom.py {} > /dev/null 2>$HEADROOM_OUT/seed_{}.err'
python3 headroom_report.py | tee $HEADROOM_OUT/REPORT.txt
