#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
cp sim_worker.py.varB.bak sim_worker_varB.py
echo "##### PROBE A: original varB worker under the pipe driver (baseline) #####"
timeout 240 python test_worker_pipe.py sim_worker_varB.py
echo
echo "##### PROBE B: varC worker, maxima at original import position (guarded) #####"
timeout 240 python test_worker_pipe.py sim_worker.py
