#!/bin/bash
# Standing backfill (runs only when the job queue is empty; stopped by SIGTERM as soon as a real job is queued).
# Resumable expert-iteration data collection under the current mainline (exi1 root prior c=5 + exi1 play-outs + PV10):
# Act 2/3 elite+boss fights played by exi1 in determinized worlds, visited states labelled with honest K=64 Q.
# Saved to sim_logs/exi_bf/data for the next training round. Finished seeds are skipped on restart.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; D=sim_logs/exi_bf/data; mkdir -p $D
rm -f $D/*.tmp.npz
for w in $(seq 0 13); do
  MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 \
    EXI_K=64 EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434 .venv/bin/python exi_collect.py $P1 $((996000001 + w*31)) 5000 $D >> sim_logs/exi_bf/collect.log 2>/dev/null &
done; wait
