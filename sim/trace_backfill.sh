#!/bin/bash
# Re-play an existing arm with TRACE=1 (passive; decisions identical) to get per-fight/deck/offer data
# for the dashboard analytics. Low priority, few workers.  usage: trace_backfill.sh ARM "ENV" [N] [P] [SEED0] [STEP]
cd "$HOME/sts"
arm=$1; envs=$2; n=${3:-150}; P=${4:-2}; export SEED0=${5:-10007} STEP=${6:-7919}
export OUTD=sim_logs/traces/$arm ENVS="$envs TRACE=1" PY=$HOME/sts/.venv/bin/python
mkdir -p $OUTD
seq 0 $((n-1)) | xargs -P $P -I{} bash -c '
  s=$(( SEED0 + {} * STEP )); o=$OUTD/seed_$s.json
  [ -s $o ] && exit 0
  env $ENVS nice -n 19 $PY eval_fullrun_analytics_v2.py $s macro 2>/dev/null > $o.tmp && mv $o.tmp $o || rm -f $o.tmp'
echo "backfill $arm done: $(ls $OUTD/seed_*.json | wc -l) seeds"
