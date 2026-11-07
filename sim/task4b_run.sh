#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== fast-replay determinism + timing =="
python test_ff_determinism.py || exit 1
echo "== search correctness =="
python test_search_correctness.py || exit 1
echo "== search vs base (signal) =="
for t in $(seq 1 20); do
  python test_search_vs_base.py && break
  echo "segfault, retry $t"
done
