#!/bin/bash
# Verification chain for the env.nn maxima fix:
# 1) in-process pattern A' probe x3 (expect correct slots, no segfault majority)
# 2) fixed worker through real Popen handshake x4 (expect correct normalized slots)
# 3) worker stability: pipe driver, 60 games (expect varB-grade: <=2 deaths)
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== 1: pattern A' in-process x3 =="
for i in 1 2 3; do python probe_nn_fixed.py; done
echo "== 2: fixed worker handshake via Popen x4 =="
for i in 1 2 3 4; do python spawn_once.py sim_worker.py; done
echo "== 3: stability driver, 60 games =="
timeout 300 python test_worker_pipe.py sim_worker.py
