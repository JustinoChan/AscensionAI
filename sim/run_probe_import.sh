#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
for i in 1 2 3; do
    echo "=== import-context probe run $i ==="
    python probe_import.py 0
    echo "rc=$?"
done
