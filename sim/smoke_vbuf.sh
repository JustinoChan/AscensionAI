#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_sim_mp.py').read()); print('parse OK')" || exit 1
# short pure-fight run: THE question is whether ev climbs off 0 with the buffer.
timeout 500 python train_sim_mp.py 30000 12 10000 chkVB fightpure 2>&1 | grep -E 'fight eval|train fight|MILESTONE|^upd' | awk 'NR%8==0 || /fight|MILE/'
rm -rf sim_models_chkVB; rm -f sim_logs/sim_training_chkVB.csv
echo "smoke done"
