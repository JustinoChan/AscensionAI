#!/usr/bin/env bash
# Best-of-K sweep for the path/combined oracle: run K route variants per seed,
# keep the best outcome per seed. Usage: run_bestofk.sh <arm> <K> [timeout_s]
set -u
ARM="${1:?usage: run_bestofk.sh <path|combined> <K> [timeout_s]}"
K="${2:-8}"
export TMO="${3:-180}"
cd ~/sts_gym_spike || exit 1
export OUT="sim_logs/bok_parts_$ARM"
rm -rf "$OUT"; mkdir -p "$OUT"
export GATE_ELITES=0 RELIC_BUY=0 ROLE_SCORE=1 SMITH=1 SMITH_TARGET=1 \
       COMBAT_NPZ=policy_varN.npz ORACLE="$ARM" P2_TOPK="${P2_TOPK:-3}"
for v in $(seq 0 $((K - 1))); do
  export PATH_VARIANT=$v
  .venv/bin/python -c "print('\n'.join(str(10007+i*7919) for i in range(150)))" | \
    xargs -P 12 -I{} bash -c \
    'timeout -s KILL "$TMO" .venv/bin/python eval_fullrun_analytics_v2.py "$1" macro > "$OUT/${1}_v'"$v"'.json" 2>/dev/null' _ {}
  echo "variant $v done ($(ls $OUT/*_v$v.json 2>/dev/null | wc -l)/150)"
done
.venv/bin/python reduce_bestofk.py "$ARM"
