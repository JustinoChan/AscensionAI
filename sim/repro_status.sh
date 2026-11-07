#!/bin/bash
cd ~/sts/sim_logs
for f in parity_v3/seed_*.json; do echo "$f $(( $(stat -c %Y $f) - $(stat -c %W $f) ))s"; done
st=$(cat repro_v3/.start); now=$(date +%s)
done_n=$(find repro_v3 -name 'seed_*.json' -size +0 | wc -l)
echo "repro_v3: elapsed $(( now-st ))s, completed $done_n/150, evals running $(pgrep -fc 'python eval_fullrun')"
