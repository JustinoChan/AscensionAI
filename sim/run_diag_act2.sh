#!/bin/bash
# Diagnose-first recon for combat/search v2 (PARALLEL + per-seed timeout).
# Runs the mainline_v2 agent over the 150 fixed eval seeds with the instrumented
# search and dumps the Act-2 monster death fights. Parallel across cores; a
# per-seed timeout isolates the fork's occasional combat-state hangs so one bad
# seed can't stall the whole sweep (a hang on one seed wedged the sequential
# version overnight).
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
N=${1:-150}
P=${2:-12}
export GATE_ELITES=0 RELIC_BUY=0 ROLE_SCORE=1 SMITH=1 COMBAT_NPZ=policy_varN.npz
OUT=diag_act2.jsonl
rm -f "$OUT" diag_failed.txt
rm -rf diag_out; mkdir -p diag_out
python -c "print('\n'.join(str(10007+i*7919) for i in range($N)))" \
  | xargs -P "$P" -I {} bash -c '
      s="$1"
      for t in 1 2; do
        out=$(timeout -s KILL 180 python diag_act2.py "$s" 2>/dev/null)
        if [ -n "$out" ]; then echo "$out" > "diag_out/$s.json"; exit 0; fi
      done
      echo "$s" >> diag_failed.txt
    ' _ {}
cat diag_out/*.json > "$OUT" 2>/dev/null
echo "DIAG_DONE rows=$(wc -l < "$OUT")  failed=$(wc -l < diag_failed.txt 2>/dev/null || echo 0)"
python agg_diag_act2.py "$OUT"
