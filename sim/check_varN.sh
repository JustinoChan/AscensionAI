#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "trainers: $(pgrep -fc '[t]rain_search')  workers: $(pgrep -fc '[s]im_worker_search')"
echo "bank now: $(wc -l < bank_train.jsonl) (combined=5154 during train)"
echo "--- eval log ---"; cat sim_logs/varN_eval.log 2>/dev/null
echo "--- varN.log tail ---"; tail -n 5 sim_logs/varN.log 2>/dev/null
