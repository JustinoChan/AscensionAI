#!/bin/bash
# Overnight queue (2026-11-07): chained, resumable, one job at a time.
# State: sim_logs/overnight/state.json (job -> todo|running|done|failed|skipped + note).
# A failed job is logged and the queue moves on. Safe to re-run at any time (the
# watchdog does): finished jobs are skipped, run_honest.sh skips finished seeds.
cd "$HOME/sts"
OD=sim_logs/overnight; mkdir -p "$OD"
LOG=$OD/overnight.log
PIDF=$OD/runner.pid
if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "runner already active"; exit 0; fi
echo $$ > "$PIDF"
export PY=$HOME/sts/.venv/bin/python
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

python3 overnight_state.py init

st()   { python3 overnight_state.py get "$1"; }
set_() { python3 overnight_state.py set "$1" "$2" "${3:-}"; }

run_job() {   # run_job <id> <cmd...>
    local id=$1; shift
    local s=$(st "$id")
    case "$s" in done|failed|skipped) log "skip $id ($s)"; return 0;; esac
    set_ "$id" running
    log "start $id: $*"
    if "$@" >> "$LOG" 2>&1; then set_ "$id" done; log "done $id"
    else set_ "$id" failed "exit $?"; log "FAILED $id"; fi
}

arm_done() {  # arm finished = SUMMARY exists with n=150
    grep -q "^n=150 " "sim_logs/honest/$1/SUMMARY.txt" 2>/dev/null
}
wall() { grep -o "wall=[0-9]*" "sim_logs/honest/$1/SUMMARY.txt" 2>/dev/null | tail -1 | cut -d= -f2; }

# ---- a. harvest ----
j_harvest() {
    while pgrep -f run_harvest_h1.sh > /dev/null; do sleep 60; done     # the original launch is still going
    local n=$(ls sim_logs/honest/harvest_h1/seed_*.json 2>/dev/null | wc -l)
    if [ "$n" -lt 1600 ]; then ./run_harvest_h1.sh; fi
    $PY harvest_status.py sim_logs/harvest/h1 | tee sim_logs/harvest/h1/STATUS.txt
    [ $(ls sim_logs/honest/harvest_h1/seed_*.json | wc -l) -ge 1600 ]
}
# ---- b. train + offline gate ----
j_train() {
    waitp "train_h1.py"
    [ -f nets/h1/GATE.json ] && return 0
    nice -n 5 .venv_train/bin/python train_h1.py sim_logs/harvest/h1 nets/h1 > nets_h1_train.log 2>&1 || { tail -20 nets_h1_train.log; return 1; }
    tail -3 nets_h1_train.log
    python3 overnight_state.py gate nets/h1/train_report.json
}
# ---- c. A/B on eval-150 vs cpp_1ply_K64 (wall 3361 s) ----
gate_flag() { python3 -c "import json;print(int(json.load(open('nets/h1/GATE.json'))['$1']))" 2>/dev/null || echo 0; }
waitp() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }   # never start a duplicate of a still-running job
ab() {   # ab <arm> <env>
    waitp "run_honest.sh $1 "
    arm_done "$1" && return 0
    ./run_honest.sh "$1" 150 7200 "HONEST=1 HONEST_CPP=1 $2" "Phase 3 1c A/B: $2"
    arm_done "$1"
}
NETS="HONEST_PRIOR=nets/h1/prior_h1.npz"
VN="HONEST_VNET=nets/h1/value_h1.npz"
j_c1() { [ "$(gate_flag prior)" = 1 ] || { set_ c1_prior3_K64 skipped "prior failed offline gate"; return 0; }
         ab ab_prior3_K64 "HONEST_K=64 $NETS HONEST_TOPK=3"; }
j_c2() { [ "$(gate_flag value)" = 1 ] || { set_ c2_pv_d8_K64 skipped "value failed offline gate"; return 0; }
         local P=""; [ "$(gate_flag prior)" = 1 ] && P="$NETS HONEST_TOPK=3"
         ab ab_pv_d8_K64 "HONEST_K=64 $P $VN HONEST_DEPTH=8"; }
j_c3() { [ "$(gate_flag value)" = 1 ] || { set_ c3_pv_d8_Keq skipped "value failed offline gate"; return 0; }
         arm_done ab_pv_d8_K64 || { set_ c3_pv_d8_Keq skipped "c2 did not finish"; return 0; }
         local w=$(wall ab_pv_d8_K64); local K=$(python3 -c "print(max(64, min(512, int(round(64*3361/max(1,$w)/16))*16)))")
         echo "equal-wall K = $K (c2 wall ${w}s vs K64 3361s)"
         local P=""; [ "$(gate_flag prior)" = 1 ] && P="$NETS HONEST_TOPK=3"
         echo "$K" > $OD/c3_K
         ab ab_pv_d8_Keq "HONEST_K=$K $P $VN HONEST_DEPTH=8"; }
j_c4() { [ "$(gate_flag value)" = 1 ] || { set_ c4_pv_d16_K64 skipped "value failed offline gate"; return 0; }
         local P=""; [ "$(gate_flag prior)" = 1 ] && P="$NETS HONEST_TOPK=3"
         ab ab_pv_d16_K64 "HONEST_K=64 $P $VN HONEST_DEPTH=16"; }
# ---- d. K=128 holdout check (independent) ----
j_d() { waitp "run_honest.sh holdout_K128 "; arm_done holdout_K128 && return 0
        SEED0=20011 SEEDSTEP=104729 ./run_honest.sh holdout_K128 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=128" "holdout-150: honest K=128 (does K keep paying?)"
        arm_done holdout_K128; }
# ---- e. report ----
j_report() { python3 overnight_report.py; }

run_job harvest   j_harvest
run_job train     j_train
run_job c1_prior3_K64  j_c1
run_job c2_pv_d8_K64   j_c2
run_job c3_pv_d8_Keq   j_c3
run_job c4_pv_d16_K64  j_c4
run_job d_holdout_K128 j_d
run_job report    j_report
log "queue finished"
rm -f "$PIDF"
