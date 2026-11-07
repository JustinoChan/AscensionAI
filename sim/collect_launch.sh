#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f teacher_part_*.npz
# 6 parallel collectors, 3500 each -> ~21k samples; seq-retry over startup segfault
for k in 0 1 2 3 4 5; do
  ( for t in 1 2 3 4 5; do
      python collect_teacher_data.py 3500 teacher_part_${k}.npz ${k} && break
      echo "part $k retry $t"
    done ) &
done
wait
echo "=== parts ==="
ls -la teacher_part_*.npz 2>/dev/null
