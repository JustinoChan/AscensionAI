#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "##### binding source for the suspect properties #####"
grep -n 'deck\|cur_room\|"act"\|upgrade_count\|is_starter' bindings/*.cpp | head -40
echo
for m in deck room; do
    for i in 1 2; do
        echo "=== mode=$m run $i ==="
        python probe_bisect.py $m 0
        echo "rc=$?"
    done
done
