#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_sim_mp.py').read()); print('parse OK')" || exit 1
timeout 600 python diag_fightmix_ev.py 2>>diag_err.log
