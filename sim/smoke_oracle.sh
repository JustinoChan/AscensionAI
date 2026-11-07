#!/usr/bin/env bash
set -u
cd ~/sts_gym_spike || exit 1
export GATE_ELITES=0 RELIC_BUY=0 ROLE_SCORE=1 SMITH=1 SMITH_TARGET=1 COMBAT_NPZ=policy_varN.npz
SEEDS="10007 17926 25845 33764 41683"
echo "=== baseline rows ==="
.venv/bin/python -c "import json; d={r['seed']:r for r in (json.loads(l) for l in open('v2_mainline_v3.jsonl'))}; [print('base ',s,'win',d[s]['win'],'floor',d[s]['floor']) for s in [$(echo $SEEDS|tr ' ' ',')]]"
echo "=== ORACLE=none (must reproduce baseline) ==="
for s in $SEEDS; do
  ORACLE=none timeout -s KILL 200 .venv/bin/python eval_fullrun_analytics_v2.py $s macro \
    | .venv/bin/python -c "import sys,json; r=json.load(sys.stdin); print('none ',r['seed'],'win',r['win'],'floor',r['floor'])"
done
echo "=== ORACLE=combat (2-ply) + timing ==="
for s in $SEEDS; do
  t0=$(date +%s)
  out=$(ORACLE=combat timeout -s KILL 600 .venv/bin/python eval_fullrun_analytics_v2.py $s macro)
  dt=$(( $(date +%s) - t0 ))
  echo "$out" | .venv/bin/python -c "import sys,json; r=json.load(sys.stdin); print('combat',r['seed'],'win',r['win'],'floor',r['floor'],'${dt}s')"
done
