#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python probe_fight_api.py 2>&1
echo "===== sts_env.py source ====="
PKG=$(python -c "import gym_sts_lightspeed, os; print(os.path.dirname(gym_sts_lightspeed.__file__))")
cat "$PKG/envs/sts_env.py"
