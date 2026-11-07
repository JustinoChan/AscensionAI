#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== A: plain shell (stdout redirected to file, stdin /dev/null) =="
timeout 5 python sim_worker_dbg.py 1 < /dev/null > /tmp/wdbg.out
echo "== B: stdout piped =="
timeout 5 python sim_worker_dbg.py 1 < /dev/null | head -c 0
echo "== C: full Popen pipes x3 (trainer-identical spawn) =="
for i in 1 2 3; do
    python spawn_once.py
done
