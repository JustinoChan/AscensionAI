#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -rf diag_mc
echo "== MC collection under frozen varE policy =="
timeout 600 python diag_collect_mc.py 2>>diag_err.log || exit 1
echo "== supervised learnability probe (MC targets) =="
timeout 600 python diag_supervised.py diag_mc 2>>diag_err.log
