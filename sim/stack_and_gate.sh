#!/bin/bash
# Phase 3 step 2c (auto): pick the better card-search fix by paired floor delta vs cpp_1ply_K64
# (ties |delta diff| < 0.25 floors -> the cheaper arm by wall time), stack it with MACRO_REST,
# run the stack on eval-150, then gate it on holdout-150 vs holdout_K64.  Resumable.
cd "$HOME/sts"
OUT=sim_logs/GATE_phase3_2.txt
A=ab_mcard_a1;  AENV="MACRO_CARD=1 MACRO_K_CARD=32 MACRO_CARD_ACTS=1"
B=ab_mcard_h17; BENV="MACRO_CARD=1 MACRO_K_CARD=16 MACRO_H_CARD=17"
REST="MACRO_REST=1 MACRO_K_REST=16 MACRO_COMBAT_K_REST=4"
python3 ledger.py ingest > /dev/null
CHOICE=$(python3 - "$A" "$B" <<'PY'
import json, subprocess, sys
a, b = sys.argv[1:]
def cmp(x): return json.loads(subprocess.run(["python3", "ledger.py", "compare", "cpp_1ply_K64", x, "--json"], capture_output=True, text=True).stdout)
def wall(x):
    t = open("sim_logs/honest/%s/SUMMARY.txt" % x).read().split("wall=")[-1].split("s")[0]
    return int(t)
ca, cb = cmp(a), cmp(b)
da, db = ca["floor_diff_b_minus_a"], cb["floor_diff_b_minus_a"]
wa, wb = wall(a), wall(b)
if abs(da - db) < 0.25:
    pick = a if wa <= wb else b; why = "tie (|%.2f-%.2f|<0.25) -> cheaper (%ds vs %ds)" % (da, db, wa, wb)
else:
    pick = a if da > db else b; why = "higher paired floor delta (%.2f vs %.2f)" % (da, db)
print(pick, why.replace(" ", "_"))
PY
)
PICK=${CHOICE%% *}; WHY=${CHOICE#* }
[ "$PICK" = "$A" ] && CENV=$AENV || CENV=$BENV
ENV="HONEST=1 HONEST_CPP=1 HONEST_K=64 $CENV $REST MACRO_LOG=1 TRACE=1"
{ echo "Phase 3 step 2c gate ($(date '+%F %T'))"; echo "card fix chosen: $PICK  because ${WHY//_/ }"; echo "stack env: $ENV"; echo; } > $OUT
for x in $A $B; do python3 ledger.py compare cpp_1ply_K64 $x | sed -n 1,6p >> $OUT; echo >> $OUT; done
grep -q "^n=150 " sim_logs/honest/ab_stack/SUMMARY.txt 2>/dev/null || ./run_honest.sh ab_stack 150 7200 "$ENV" "Phase 3 2c: $PICK card fix + rest search on mainline_h1"
echo "== stack on eval-150 vs K=64" >> $OUT; python3 ledger.py compare cpp_1ply_K64 ab_stack | sed -n 1,16p >> $OUT; echo >> $OUT
grep -q "^n=150 " sim_logs/honest/holdout_stack/SUMMARY.txt 2>/dev/null || SEED0=20011 SEEDSTEP=104729 ./run_honest.sh holdout_stack 150 7200 "$ENV" "Phase 3 2c GATE holdout-150: $PICK + rest search"
echo "== GATE: stack on holdout-150 vs holdout K=64" >> $OUT; python3 ledger.py compare holdout_K64 holdout_stack | sed -n 1,16p >> $OUT
cat $OUT
grep -q "^n=150 " sim_logs/honest/holdout_stack/SUMMARY.txt
