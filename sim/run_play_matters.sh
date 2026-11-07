#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
for try in $(seq 1 20); do
  python reprobe_play_matters.py 2>>diag_err.log && break
  echo "crashed, retry $try"
done
