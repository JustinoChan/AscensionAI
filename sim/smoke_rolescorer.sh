#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0; export RELIC_BUY=0
PY='import sys,json
d=json.load(sys.stdin); a=d.get("act2_deck") or {}
print("  floor",d["floor"],"deck",d["final_deck"]["size"],"act2_aoe",a.get("aoe"),"act2_draw",a.get("draw_energy"),"act2_size",a.get("size"))'
for s in 144630 10007 41683; do
  echo "seed $s:"
  echo -n "  A(off):"; ROLE_SCORE=0 python eval_fullrun_analytics_v2.py $s macro 2>/dev/null | python -c "$PY"
  echo -n "  B(on): "; ROLE_SCORE=1 python eval_fullrun_analytics_v2.py $s macro 2>/dev/null | python -c "$PY"
done
