#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "=== checkpoint mainline_v1 ==="
cp eval_fullrun_analytics_v2.py eval_mainline_v1.py
cp v2_Br.jsonl v2_mainline_v1.jsonl 2>/dev/null
echo "saved eval_mainline_v1.py + v2_mainline_v1.jsonl ($(wc -l < v2_mainline_v1.jsonl 2>/dev/null) runs)"
echo "=== smith smoke (off vs on) ==="
PY='import sys,json
d=json.load(sys.stdin); a=d.get("act2_deck") or {}
print("  floor",d["floor"],"upg@a2",a.get("upgraded"),"rests",d.get("rests"),"campfires",d.get("campfires"))'
for s in 144630 10007 41683; do
  echo "seed $s:"
  echo -n "  A(off):"; SMITH=0 python eval_fullrun_analytics_v2.py $s macro 2>/dev/null | python -c "$PY"
  echo -n "  B(on): "; SMITH=1 python eval_fullrun_analytics_v2.py $s macro 2>/dev/null | python -c "$PY"
done
