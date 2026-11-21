#!/bin/bash
# Diagnostic: are the expert-iteration nets better STANDALONE fighters than varN? Held-out Act 2/3 boss starts (first one
# reached), 8 determinized episodes each, greedy play by each net only (no search); identical starts and worlds per net.
cd "$HOME/sts"
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"
for NET in policy_varN.npz nets/exi1/rollout.npz nets/exi1b/rollout.npz; do
  T=$(basename $NET .npz); [ "$(dirname $NET)" != . ] && T=$(basename $(dirname $NET))_$T; D=sim_logs/exi_ptest/$T
  for w in $(seq 0 13); do
    env $MAIN BOSS_ACTS=2,3 BOSS_SEEDSTEP=434 BOSS_GREEDY=1 nice -n 5 .venv/bin/python boss_rl.py worker $NET $((981000007 + w*31)) 8 8 $D >> sim_logs/exi_ptest.log 2>/dev/null &
  done; wait
done
.venv/bin/python - <<'PY' | tee sim_logs/GATE_exi_ptest.txt
import glob, numpy as np, os
R = {}
for d in sorted(glob.glob("sim_logs/exi_ptest/*")):
    R[os.path.basename(d)] = {os.path.basename(f): np.load(f) for f in glob.glob(d + "/seed_*.npz")}
common = set.intersection(*[set(v) for v in R.values()])
print("standalone greedy play, held-out Act 2/3 boss starts: %d starts x 8 determinized episodes" % len(common))
for k, v in R.items():
    r = np.concatenate([v[s]["reward"] for s in sorted(common)])
    print("  %-28s win %.3f   mean reward %.3f" % (k, float((r >= 1).mean()), float(r.mean())))
PY
