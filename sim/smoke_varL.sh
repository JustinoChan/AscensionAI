#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
# tiny end-to-end: 4 workers, ~2 collects + a milestone eval, throwaway tag.
rm -f sim_logs/sim_training_smoke.csv
timeout 180 python train_search.py 9000 4 4000 smoke 2>&1 | tail -n 40
echo "--- exit ---"
echo "leftover workers: $(pgrep -fc '[s]im_worker_search' || echo 0)"
pkill -f '[s]im_worker_search' 2>/dev/null
rm -rf sim_models_smoke sim_logs/sim_training_smoke.csv
