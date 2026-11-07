#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -rf diag_mc2
echo "== MC collection (5 gammas) =="
timeout 600 python diag_collect_mc2.py 2>>diag_err.log || exit 1
echo "== horizon sweep =="
timeout 900 python diag_sweep.py 2>>diag_err.log
