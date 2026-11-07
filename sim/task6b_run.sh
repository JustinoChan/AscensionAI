#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== timing (threads=1 + top-k) =="
python test_search_timing.py || exit 1
echo "== search vs base (still holds with top-k?) =="
for t in $(seq 1 20); do
  python test_search_vs_base.py && break
  echo "segfault, retry $t"
done
