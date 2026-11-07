#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_sim_mp.py').read()); print('parse OK')" || exit 1
timeout 400 python train_sim_mp.py 600 12 200 chkP fightpure 2>/dev/null
rm -rf sim_models_chkP; rm -f sim_logs/sim_training_chkP.csv
echo "smoke done"
