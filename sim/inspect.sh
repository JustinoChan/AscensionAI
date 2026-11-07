#!/usr/bin/env bash
cd ~/sts_gym_spike || exit 1
echo "===== JSONL audit ====="
.venv/bin/python - <<'PY'
import json
rows=[json.loads(l) for l in open('v2_mainline_v3.jsonl')]
print("rows:", len(rows), "wins:", sum(bool(r.get('won')) for r in rows))
print("keys:", sorted(rows[0].keys()))
PY
echo
echo "===== file sizes ====="
wc -l eval_fullrun_analytics_v2.py sim_worker_search.py 2>&1
echo
echo "===== env toggles in eval_fullrun_analytics_v2.py ====="
grep -n "os.environ" eval_fullrun_analytics_v2.py
echo
echo "===== argparse / main ====="
grep -n "add_argument\|argparse\|__main__\|def main\|--seed\|--mode\|--out" eval_fullrun_analytics_v2.py
echo
echo "===== def functions (eval) ====="
grep -n "def " eval_fullrun_analytics_v2.py
echo
echo "===== result dict / won / floor / act2 ====="
grep -n "'won'\|\"won\"\|'floor'\|act2_deck\|_deck_roles\|json.dump\|\.write(" eval_fullrun_analytics_v2.py
echo
echo "===== search functions (worker) ====="
grep -n "def \|rollout\|_ff_step\|_search\|N_ROLL\|ply\|greedy" sim_worker_search.py
