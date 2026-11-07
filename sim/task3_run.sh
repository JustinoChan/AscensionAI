#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python export_live.py sim_models_varJ/ppo_sim_latest.pt policy_live.npz
python -c "import ast; ast.parse(open('sim_worker_search.py').read()); print('parse OK')"
