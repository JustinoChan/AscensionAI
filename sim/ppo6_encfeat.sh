#!/bin/bash
# Round 6: + MACRO_ENCFEAT (public encounter / potion-economy features), PPO fine-tune from 5d with the first layer
# widened by zero columns (identical policy at start), critic on; then eval500 gate vs 5d.
cd "$HOME/sts"
export MACRO_MAPFEAT=1 MACRO_ENCFEAT=1
mkdir -p nets/rl6
[ -f nets/rl6/pol_bc.npz ] || .venv/bin/python - <<'PY'
import numpy as np
z = dict(np.load("nets/rl5d/pol_gate.npz")); w = z["l0.weight"]
z["l0.weight"] = np.concatenate([w, np.zeros((w.shape[0], 12), w.dtype)], 1)
np.savez("nets/rl6/pol_bc.npz", **z); print("widened", w.shape, "->", z["l0.weight"].shape)
PY
RL_CRITIC=1 RL_OUT=nets/rl6 RL_PROG=sim_logs/honest/rl_macro6 RL_TMP=sim_logs/rl6 RL_SEED0=900500041 RL_COMBAT_K=8 RL_WORKERS=14 \
  RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-80} >> nets/rl6/rl.log 2>&1
NETDIR=nets/rl6 ARM=eval500_rl6_K64 OUTF=sim_logs/GATE_rl6_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 MACRO_ENCFEAT=1" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5d_K64 eval500_rl6_K64 | sed -n 1,20p; python3 boss_report.py eval500_rl5d_K64 eval500_rl6_K64 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl6_eval500.txt
