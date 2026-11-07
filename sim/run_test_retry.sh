#!/bin/bash
# The extension has a known ~50% per-process startup segfault (heap-layout
# dependent). Retry the shadow test in fresh processes; stop on any
# non-segfault exit so real assertion failures surface immediately.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
for i in 1 2 3 4 5 6; do
    echo "TRY $i"
    python test_reward_varc.py
    rc=$?
    echo "rc=$rc"
    if [ $rc -ne 139 ]; then
        exit $rc
    fi
done
echo "segfault on all 6 tries -- NOT the startup heisenbug, investigate the reward code"
exit 139
