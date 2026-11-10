#!/bin/bash
# Nov 9-10 overnight: branch on RL round 3's eval500 gate vs iteration 99 (eval500_rl_K64: 28/500, floor 32.14).
cd "$HOME/sts"
python3 ledger.py ingest >/dev/null
DEC=$(python3 - <<'PY'
import sqlite3
c = sqlite3.connect("sim_logs/results.db")
def s(a):
    n, w, f = c.execute("select count(*), sum(win), avg(floor) from seeds where run=?", (a,)).fetchone()
    return n, w or 0, f or 0
n3, w3, f3 = s("eval500_rl3_K64"); n1, w1, f1 = s("eval500_rl_K64")
beats = n3 == 500 and (w3 > w1 or (w3 == w1 and f3 > f1))
print("A" if beats else "B", w3, round(f3, 2), w1, round(f1, 2))
PY
)
set -- $DEC
echo "[$(date '+%F %T')] round 3 eval500: $2 wins floor $3 vs it099 $4 wins floor $5 -> branch $1" | tee sim_logs/NIGHT_BRANCH.txt
export MACRO_MAPFEAT=1
if [ "$1" = A ]; then
  # round 3 is the new best: holdout confirmation, then continue its PPO from the gated checkpoint, then gate
  POL=nets/rl3/pol_gate.npz ARM=holdout_rl3_K64 EXTRA_ENV="MACRO_MAPFEAT=1" ./holdout_pol.sh
  mkdir -p nets/rl3c; [ -f nets/rl3c/pol_bc.npz ] || cp nets/rl3/pol_gate.npz nets/rl3c/pol_bc.npz
  RL_OUT=nets/rl3c RL_PROG=sim_logs/honest/rl_macro3c RL_TMP=sim_logs/rl3c RL_SEED0=660000011 RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0.3 RL_LR=5e-5 \
    nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-120} >> nets/rl3c/rl.log 2>&1
  NETDIR=nets/rl3c ARM=eval500_rl3c_K64 OUTF=sim_logs/GATE_rl3c_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1" ./rl_gate.sh
  python3 ledger.py compare eval500_rl_K64 eval500_rl3c_K64 | sed -n 1,20p | tee -a sim_logs/GATE_rl3c_eval500.txt
else
  # round 3 did not beat it099: single-variable ablation = map features WITHOUT the boss bonus (same BC start)
  mkdir -p nets/rl4; [ -f nets/rl4/pol_bc.npz ] || cp nets/rl3/pol_bc.npz nets/rl4/pol_bc.npz
  RL_OUT=nets/rl4 RL_PROG=sim_logs/honest/rl_macro4 RL_TMP=sim_logs/rl4 RL_SEED0=680000021 RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0 \
    nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-150} >> nets/rl4/rl.log 2>&1
  NETDIR=nets/rl4 ARM=eval500_rl4_K64 OUTF=sim_logs/GATE_rl4_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1" ./rl_gate.sh
  { python3 ledger.py compare eval500_rl_K64 eval500_rl4_K64 | sed -n 1,20p; python3 ledger.py compare eval500_rl3_K64 eval500_rl4_K64 | sed -n 4,6p; } | tee -a sim_logs/GATE_rl4_eval500.txt
fi
