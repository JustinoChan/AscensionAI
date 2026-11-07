#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "=== backup + build combined bank ==="
cp bank_train.jsonl bank_train.act1.bak
python build_combined_bank.py
echo "=== varN smoke (rollout teacher + varL warm-start + act-oversample) ==="
rm -f sim_logs/sim_training_varNsmoke.csv
timeout 220 python train_search.py 12000 4 4000 varNsmoke 2>&1 | tail -n 16
pkill -f '[s]im_worker_search' 2>/dev/null
rm -rf sim_models_varNsmoke sim_logs/sim_training_varNsmoke.csv
echo "=== bank line count (should be combined) ==="
wc -l bank_train.jsonl bank_train.act1.bak
