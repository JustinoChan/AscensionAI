#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python -c "import ast; ast.parse(open('train_search.py').read()); print('parse OK')" || exit 1
timeout 600 python train_search.py 8000 6 4000 chkS 2>&1 | tail -25
rm -rf sim_models_chkS; rm -f sim_logs/sim_training_chkS.csv
echo "smoke done"
