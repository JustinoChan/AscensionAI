#!/bin/bash
cd "$HOME/sts"
NETDIR=nets/rl5e ARM=eval500_rl5e_K64 OUTF=sim_logs/GATE_rl5e_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5d_K64 eval500_rl5e_K64 | sed -n 1,20p; } | tee -a sim_logs/GATE_rl5e_eval500.txt
