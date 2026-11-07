#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -rf samp_dumps
TARGET=1000
for try in $(seq 1 40); do
  python reprobe_samp_collect.py samp_dumps $TARGET 2>>diag_err.log
  CODE=$?
  if [ "$CODE" -eq 0 ]; then break; fi
  echo "collect crashed (exit $CODE), restarting ($try)"
done
echo "== fit =="
python reprobe_samp_fit.py 2>>diag_err.log
