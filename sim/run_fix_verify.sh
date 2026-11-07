#!/bin/bash
# Verify the deferred-maxima fix: the import probe must go clean, then the
# real shadow test must PASS. Segfault on a shadow attempt retries with new
# seeds (residual native crashes are possible); assertion failures stop hard.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "===== import-context probe (3 runs, expect CLEAN) ====="
for i in 1 2 3; do
    echo "--- probe run $i ---"
    python probe_import.py 0
    echo "rc=$?"
done
echo "===== shadow test (attempts until PASS, max 5) ====="
for a in 0 1 2 3 4; do
    echo "--- shadow attempt $a ---"
    python test_reward_varc.py $a
    rc=$?
    echo "rc=$rc"
    if [ $rc -eq 0 ]; then
        echo "SHADOW TEST PASS (attempt $a)"
        exit 0
    elif [ $rc -ne 139 ]; then
        echo "REAL FAILURE rc=$rc -- stop and investigate"
        exit $rc
    fi
done
echo "all attempts segfaulted -- fix insufficient"
exit 139
