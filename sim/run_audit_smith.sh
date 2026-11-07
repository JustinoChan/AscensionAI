#!/bin/bash
# Smith-target audit over the 150 fixed eval seeds (parallel + per-seed timeout).
# Runs the mainline_v2 agent with the read-only smith-target logger; identical
# behavior, just records what upg[0] upgrades. ~3 min.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
N=${1:-150}
P=${2:-12}
export GATE_ELITES=0 RELIC_BUY=0 ROLE_SCORE=1 SMITH=1 COMBAT_NPZ=policy_varN.npz
OUT=audit_smith.jsonl
rm -f "$OUT" audit_failed.txt
rm -rf audit_out; mkdir -p audit_out
python -c "print('\n'.join(str(10007+i*7919) for i in range($N)))" \
  | xargs -P "$P" -I {} bash -c '
      s="$1"
      for t in 1 2; do
        out=$(timeout -s KILL 180 python audit_smith.py "$s" 2>/dev/null)
        if [ -n "$out" ]; then echo "$out" > "audit_out/$s.json"; exit 0; fi
      done
      echo "$s" >> audit_failed.txt
    ' _ {}
cat audit_out/*.json > "$OUT" 2>/dev/null
echo "AUDIT_DONE rows=$(wc -l < "$OUT")  failed=$(cat audit_failed.txt 2>/dev/null | wc -l)"
python agg_audit_smith.py "$OUT"
