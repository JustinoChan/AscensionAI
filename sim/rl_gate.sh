#!/bin/bash
# Gate for the RL out-of-combat policy: best validation checkpoint (final if as good) with honest K=64 combat
# on all 500 eval500 seeds, paired vs eval500_K64. Writes sim_logs/GATE_rl_eval500.txt.
cd "$HOME/sts"
PICK=$(python3 - <<'PY'
import json, os
h = json.load(open("nets/rl/train_history.json"))
V = [v for v in h["rl_val"] if os.path.exists("nets/rl/val_%s.npz" % v["tag"])]
best = max(V, key=lambda v: (v["wins"], v["floor"]))
last = V[-1]
pick = last if (last["wins"], last["floor"]) >= (best["wins"] - 1, best["floor"] - 0.5) else best
print(pick["tag"], pick["floor"], pick["wins"], best["tag"], last["tag"])
PY
)
set -- $PICK; TAG=$1
cp nets/rl/val_$TAG.npz nets/rl/pol_gate.npz
echo "gate policy: checkpoint $TAG (val floor $2, wins $3/100; best=$4, last=$5)" | tee sim_logs/GATE_rl_eval500.txt
grep -q "^n=500 " sim_logs/honest/eval500_rl_K64/SUMMARY.txt 2>/dev/null || \
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl_K64 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_POLICY=nets/rl/pol_gate.npz TRACE=1" "GATE: RL out-of-combat policy ($TAG) + honest K=64 combat, eval500"
python3 ledger.py ingest > /dev/null
{ echo; python3 ledger.py compare eval500_K64 eval500_rl_K64 | sed -n 1,20p; echo
  python3 - <<'PY'
import json, glob, collections
def load(a): return [json.load(open(f)) for f in glob.glob("sim_logs/honest/%s/seed_*.json" % a)]
for a in ("eval500_K64", "eval500_rl_K64"):
    rs = load(a)
    reach = {k: sum(1 for r in rs if r["floor"] >= f) for k, f in ((1, 0), (2, 17), (3, 34))}
    die = collections.Counter((r.get("death_act"), r.get("death_room")) for r in rs if not r["win"])
    print(a)
    for k in (1, 2, 3):
        d = sum(v for (da, rm), v in die.items() if da == k)
        print("  Act %d: reached %3d, died %3d (%.1f%% of reaching)  %s" % (k, reach[k], d, 100.0 * d / max(1, reach[k]),
              {rm: v for (da, rm), v in sorted(die.items(), key=lambda x: -x[1]) if da == k}))
PY
  echo; python3 ledger.py taxonomy eval500_rl_K64 | sed -n 1,20p; } | tee -a sim_logs/GATE_rl_eval500.txt
