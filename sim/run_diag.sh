#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== forensics: did the v head ever leave init? =="
python diag_forensics.py sim_models_varE/ppo_sim_0050*.pt sim_models_varE/ppo_sim_final.pt sim_models_varF/ppo_sim_0050*.pt sim_models_varF/ppo_sim_final.pt
echo "== instrumented run (150 updates, bare mode) =="
rm -rf diag_dumps
timeout 900 python diag_value.py 2>diag_err.log
echo "== supervised learnability probe =="
timeout 600 python diag_supervised.py 2>>diag_err.log
