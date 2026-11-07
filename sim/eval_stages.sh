#!/bin/bash
# Reproducible AscensionAI stage progression on the fixed eval seeds:
#   1 baseline       (varE everywhere, no search)
#   2 search-combat  (search drives combat, baseline deck/map policy)
#   3 deck-hygiene   (search-combat + macro deck hygiene, varL combat net)
#   4 mainline_v2    (+ varN combat + AoE/draw role scorer + conservative smith)
# Usage: bash eval_stages.sh [n_seeds]   (default 150; ~25 min/macro stage)
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
N=${1:-150}
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range($N)))")

run_macro () {  # $1 = out jsonl  (toggles preset via env)
  rm -f "$1"
  for s in $SEEDS; do
    for t in 1 2 3; do
      out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
      if [ -n "$out" ]; then echo "$out" >> "$1"; break; fi
    done
  done
}

echo "### 1 BASELINE (varE everywhere)"
python eval_fullrun_search.py baseline policy_varL.npz policy_varE.npz $N | tail -6
echo "### 2 SEARCH-COMBAT (varL combat, baseline deck)"
python eval_fullrun_search.py search policy_varL.npz policy_varE.npz $N | tail -6

export GATE_ELITES=0 RELIC_BUY=0
echo "### 3 DECK-HYGIENE (varL combat, no role/smith)"
ROLE_SCORE=0 SMITH=0 COMBAT_NPZ=policy_varL.npz run_macro stage_deckhygiene.jsonl
python aggregate_v2.py stage_deckhygiene.jsonl | head -2

echo "### 4 MAINLINE_v2 (varN + role scorer + conservative smith)"
ROLE_SCORE=1 SMITH=1 COMBAT_NPZ=policy_varN.npz run_macro stage_mainline_v2.jsonl
python aggregate_v2.py stage_mainline_v2.jsonl | head -2
echo STAGES_DONE
