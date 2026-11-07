#!/bin/bash
# A/B: smith targeting (SMITH_TARGET) vs mainline_v2. Both arms identical except
# the smith upgrade target. 150 fixed seeds each, parallel + per-seed timeout.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
N=${1:-150}
P=${2:-12}
export GATE_ELITES=0 RELIC_BUY=0 ROLE_SCORE=1 SMITH=1 COMBAT_NPZ=policy_varN.npz

run_arm () {  # $1 = SMITH_TARGET value, $2 = out jsonl
  local st=$1 out=$2
  rm -f "$out"
  rm -rf "ab_out_$st"; mkdir -p "ab_out_$st"
  export SMITH_TARGET="$st" ABDIR="ab_out_$st"
  python -c "print('\n'.join(str(10007+i*7919) for i in range($N)))" \
    | xargs -P "$P" -I {} bash -c '
        s="$1"
        for t in 1 2; do
          out=$(timeout -s KILL 180 python eval_fullrun_analytics_v2.py "$s" macro 2>/dev/null)
          if [ -n "$out" ]; then echo "$out" > "$ABDIR/$s.json"; exit 0; fi
        done
      ' _ {}
  cat "ab_out_$st"/*.json > "$out" 2>/dev/null
}

echo "### A arm (SMITH_TARGET=0 = mainline_v2)"
run_arm 0 ab_smithA.jsonl
echo "### B arm (SMITH_TARGET=1 = tempo targeting)"
run_arm 1 ab_smithB.jsonl
echo "AB_DONE A=$(wc -l < ab_smithA.jsonl) B=$(wc -l < ab_smithB.jsonl)"
python agg_smith_ab.py ab_smithA.jsonl ab_smithB.jsonl
