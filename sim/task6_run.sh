#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_sim_mp.py').read()); print('parse OK')"
timeout 300 python train_sim_mp.py 200 4 100000 chkReg bare || exit 1
rm -rf sim_models_chkReg
echo "regression OK"
