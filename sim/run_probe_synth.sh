#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
for i in 0 1 2; do
  echo "===== attempt $i ====="
  python probe_synth_fight.py $i
  echo "exit: $?"
done
