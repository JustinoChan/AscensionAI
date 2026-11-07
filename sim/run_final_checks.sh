#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== worker shadow test (retry wrapper) =="
bash run_test_retry.sh || exit 1
echo
echo "== FRESH integration smoke (stale varCsmoke artifacts removed) =="
rm -rf sim_models_varCsmoke sim_logs/sim_training_varCsmoke.csv
echo '600 4 300 varCsmoke' > .sim_autorun
timeout 600 python train_sim_mp.py 600 4 300 varCsmoke
if [ -f .sim_autorun ]; then echo 'FAIL: autorun still present'; else echo 'autorun gone OK'; fi
head -1 sim_logs/sim_training_varCsmoke.csv
tail -n 2 sim_logs/sim_training_varCsmoke.csv
