#!/bin/bash
# Phase 1 honest-search measurements on the eval seeds (10007 + i*7919).
# usage: run_honest.sh <arm_name> <n_seeds> <timeout_s> "<ENV=.. ENV=..>" [description]
# Writes sim_logs/honest/<arm>/{RUN.json, run.log, seed_*.json, SUMMARY.txt}
# (see run_status.py for the convention). Resumable: finished seeds are skipped.
set -u
cd "$HOME/sts"
export PY="$HOME/sts/.venv/bin/python"
arm=$1; n=$2; to=$3; envs=$4; desc=${5:-"$envs"}
export SEED0=${SEED0:-10007} SEEDSTEP=${SEEDSTEP:-7919}
export OUTD="sim_logs/honest/$arm" ARMENV="$envs HONEST_HB=sim_logs/honest/$arm" TO="$to"
mkdir -p "$OUTD"
date +%s > "$OUTD/.start"
rm -f "$OUTD/.end"
echo "$envs" > "$OUTD/.env"
PGID=$(ps -o pgid= -p $$ | tr -d ' ')
python3 - "$OUTD" "$arm" "$n" "$envs" "$desc" "$PGID" <<'PY'
import json, sys, time
d, arm, n, envs, desc, pgid = sys.argv[1:]
import subprocess
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
json.dump({"git_commit": commit + ("+dirty" if dirty else ""), "name": arm, "description": desc, "group": "honest", "total": int(n), "parallel": 14,
           "env": envs, "seed_set": "%s + i*%s" % (__import__("os").environ["SEED0"], __import__("os").environ["SEEDSTEP"]), "started_at": time.time(), "pgid": int(pgid),
           "log": d + "/run.log", "out_dir": d}, open(d + "/RUN.json", "w"), indent=1)
PY
echo "[$(date '+%F %T')] start $arm  n=$n  timeout=${to}s  env: $envs" >> "$OUTD/run.log"
seq 0 $((n-1)) | xargs -P 14 -I{} bash -c '
  seed=$(( SEED0 + {} * SEEDSTEP )); out="$OUTD/seed_$seed.json"
  [ -s "$out" ] && exit 0
  env $ARMENV timeout -s KILL $TO "$PY" eval_fullrun_analytics_v2.py $seed macro > "$out.tmp" 2>"$OUTD/seed_$seed.err" \
    && mv "$out.tmp" "$out" \
    || { rm -f "$out.tmp"; echo "{\"seed\":$seed,\"win\":0,\"floor\":0,\"timeout\":1}" > "$out"; }
  [ -s "$OUTD/seed_$seed.err" ] || rm -f "$OUTD/seed_$seed.err"
  rm -f "$OUTD/seed_$seed.prog"
  python3 -c "import json,sys,time;r=json.load(open(sys.argv[1]));print(time.strftime(\"[%F %T]\"),\"seed\",r[\"seed\"],\"floor\",r.get(\"floor\"),\"win\",r.get(\"win\"),\"secs\",r.get(\"secs\",\"?\"),\"TIMEOUT\" if r.get(\"timeout\") else \"\")" "$out" >> "$OUTD/run.log"
'
date +%s > "$OUTD/.end"
"$PY" reduce_draftsearch.py "$OUTD" > "$OUTD/SUMMARY.txt" 2>&1
python3 - "$OUTD" >> "$OUTD/SUMMARY.txt" <<'PY'
import json, glob, sys, statistics as st
d = sys.argv[1]
rows = [json.load(open(p)) for p in glob.glob(d + "/seed_*.json")]
secs = [r["secs"] for r in rows if "secs" in r]
t0 = int(open(d + "/.start").read()); t1 = int(open(d + "/.end").read())
print("n=%d wins=%d avg_floor=%.2f timeouts=%d  secs/seed mean=%.1f median=%.1f max=%.1f  wall=%ds" % (
    len(rows), sum(r.get("win", 0) for r in rows), sum(r.get("floor", 0) for r in rows) / max(1, len(rows)),
    sum(r.get("timeout", 0) for r in rows), st.mean(secs) if secs else 0, st.median(secs) if secs else 0,
    max(secs) if secs else 0, t1 - t0))
PY
cat "$OUTD/SUMMARY.txt" | tee -a "$OUTD/run.log"
echo "[$(date '+%F %T')] done $arm" >> "$OUTD/run.log"
nice -n 19 python3 ledger.py ingest >> "$OUTD/run.log" 2>&1 || true
