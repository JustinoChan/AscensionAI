#!/bin/bash
# Locate the native crash: faulthandler prints the Python stack at SIGSEGV.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
for i in 1 2 3; do
    echo "=== FAULTHANDLER TRY $i ==="
    python -X faulthandler test_reward_varc.py
    echo "rc=$?"
done
