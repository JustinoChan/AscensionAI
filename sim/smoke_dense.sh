#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_sim_mp.py').read()); print('parse OK')" || exit 1
echo "== fight worker stability with dense reward (60 fights) =="
timeout 300 python test_fight_pipe.py | grep -E 'RESULT|determinism' || exit 1
echo "== short pure-fight run: watch ev + train win-rate =="
timeout 400 python train_sim_mp.py 30000 12 10000 chkD2 fightpure 2>/dev/null | grep -E 'fight eval|train fight|MILESTONE|^upd' | awk 'NR%10==0 || /fight|MILE/'
rm -rf sim_models_chkD2; rm -f sim_logs/sim_training_chkD2.csv
echo "smoke done"
