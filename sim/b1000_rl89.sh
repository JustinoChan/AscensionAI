#!/bin/bash
# Is round 8 / round 9 (+PV10) a real gain over the mainline 5d+PV10? eval500b (fresh 500 seeds) for each, better
# eval500 first; combined 1000-seed paired McNemar vs the mainline (5d+PV10: 192 + 191).
cd "$HOME/sts"
python3 ledger.py ingest > /dev/null
ORD=$(python3 -c "
import sqlite3; c=sqlite3.connect('sim_logs/results.db')
w=lambda r: (c.execute('select count(*),sum(win) from seeds where run=?',(r,)).fetchone())
a,b=w('eval500_rl8_pv10'),w('eval500_rl9_pv10')
print('rl9 rl8' if b[0]==500 and b[1]>a[1] else 'rl8 rl9')")
for R in $ORD; do
  [ -f nets/$R/pol_gate.npz ] || continue
  SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_${R}_pv10 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/$R/pol_gate.npz POTION_VALUE=10 TRACE=1" "$R + PV10, eval500b (fresh seeds)"
  python3 ledger.py ingest > /dev/null
  { python3 ledger.py compare eval500b_rl5d_pv10 eval500b_${R}_pv10 | sed -n 3,6p; python3 b1000.py rl5d_pv10 ${R}_pv10; } | tee -a sim_logs/B1000_rl89.txt
done
