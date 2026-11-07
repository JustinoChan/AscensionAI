#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "== D: in-process, exact worker call order =="
for i in 1 2 3; do python probe_nn_workerorder.py; done
echo "== E: production worker, stdout to FILE (retry past startup lottery) =="
for i in 1 2 3 4 5 6; do
    timeout 5 python sim_worker.py 1 < /dev/null > /tmp/w_file.bin 2>/dev/null
    if [ -s /tmp/w_file.bin ]; then break; fi
done
python parse_pack.py /tmp/w_file.bin
echo "== F: production worker, stdout through PIPE =="
for i in 1 2 3 4 5 6; do
    timeout 5 python sim_worker.py 1 < /dev/null 2>/dev/null | cat > /tmp/w_pipe.bin
    if [ -s /tmp/w_pipe.bin ]; then break; fi
done
python parse_pack.py /tmp/w_pipe.bin
