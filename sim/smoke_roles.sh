#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0; export RELIC_BUY=0
python eval_fullrun_analytics_v2.py 144630 macro 2>/dev/null > one.jsonl
echo "act2_deck field:"; grep -o '"act2_deck": {[^}]*}' one.jsonl | head -c 320; echo
echo "-- aggregate_roles on 1 row --"
python aggregate_roles.py one.jsonl
