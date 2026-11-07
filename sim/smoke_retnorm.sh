#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_sim_mp.py').read()); print('parse OK')" || exit 1
# short pure-fight run: watch ev climb off 0 as value-target normalization kicks in
timeout 400 python train_sim_mp.py 30000 12 100000 chkRN fightpure 2>/dev/null | grep -E '^upd' | awk 'NR%15==0'
rm -rf sim_models_chkRN; rm -f sim_logs/sim_training_chkRN.csv
echo "smoke done"
