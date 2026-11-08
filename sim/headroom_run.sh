#!/bin/bash
# Phase 3 step 3.6: pick eval500 seeds whose traces have Act 2/3 elite/boss fights until ~TARGET fights, replay them.
cd "$HOME/sts"
TARGET=${TARGET:-300}; PAR=${PAR:-14}
SEEDS=$(python3 - $TARGET <<'PY'
import json, glob, sys
t = int(sys.argv[1]); n = 0; out = []
for f in sorted(glob.glob("sim_logs/honest/eval500_K64/seed_*.json"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0])):
    r = json.load(open(f))
    k = sum(1 for x in r.get("trace", {}).get("fights", []) if x["act"] >= 2 and x["room"] in ("ELITE", "BOSS"))
    if k:
        out.append(r["seed"]); n += k
    if n >= t:
        break
print(" ".join(map(str, out)))
PY
)
echo "$(echo $SEEDS | wc -w) seeds"
mkdir -p sim_logs/headroom
echo $SEEDS | tr ' ' '\n' | xargs -P $PAR -I{} bash -c '[ -s sim_logs/headroom/seed_{}.json ] && exit 0; HONEST=1 HONEST_CPP=1 HONEST_K=64 nice -n 5 .venv/bin/python headroom.py {} > /dev/null 2>sim_logs/headroom/seed_{}.err; true'
python3 headroom_report.py | tee sim_logs/headroom/REPORT.txt
