#!/usr/bin/env bash
# Run one oracle arm over the 150 fixed eval seeds, one process per seed,
# parallel with a per-seed hard timeout. Usage: run_oracle_arm.sh <arm> [timeout_s]
set -u
ARM="${1:?usage: run_oracle_arm.sh <none|combat|path|deck|combined> [timeout_s]}"
export TMO="${2:-180}"
cd ~/sts_gym_spike || exit 1
export OUT="sim_logs/oracle_parts_$ARM"
rm -rf "$OUT"; mkdir -p "$OUT"
export GATE_ELITES=0 RELIC_BUY=0 ROLE_SCORE=1 SMITH=1 SMITH_TARGET=1 \
       COMBAT_NPZ=policy_varN.npz ORACLE="$ARM" P2_TOPK="${P2_TOPK:-3}"
.venv/bin/python -c "print('\n'.join(str(10007+i*7919) for i in range(150)))" | \
  xargs -P 12 -I{} bash -c \
  'timeout -s KILL "$TMO" .venv/bin/python eval_fullrun_analytics_v2.py "$1" macro > "$OUT/$1.json" 2>/dev/null' _ {}
cat "$OUT"/*.json > "sim_logs/oracle_$ARM.jsonl" 2>/dev/null
echo "wrote sim_logs/oracle_$ARM.jsonl rows=$(wc -l < sim_logs/oracle_$ARM.jsonl)"
