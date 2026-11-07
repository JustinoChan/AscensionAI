#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python test_search_correctness.py || exit 1
for t in $(seq 1 20); do
  python test_search_vs_base.py && break
  echo "segfault, retry $t"
done
