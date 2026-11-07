#!/bin/bash
# Hypothesis test: same trajectory (attempt 0 seeds), reward code as the ONLY
# variable. If the no-reward control dies at the same game index as the shadow
# test, the segfault is the fork's native getObservation crash, not Variant C.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "===== SHADOW TEST (with Variant C reward), attempt 0, 2 runs ====="
for i in 1 2; do
    echo "--- shadow run $i ---"
    python test_reward_varc.py 0
    echo "rc=$?"
done
echo "===== CONTROL (no reward override), attempt 0, 3 runs ====="
for i in 1 2 3; do
    echo "--- control run $i ---"
    python control_no_reward.py 0
    echo "rc=$?"
done
