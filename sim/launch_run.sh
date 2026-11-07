#!/bin/bash
# Launch a preset run detached (used by the AscensionAI control panel).
# usage: launch_run.sh <preset> [suffix]     presets: see `launch_run.sh list`
#        launch_run.sh stop <run_name>       kill the run's process group
set -u
cd "$HOME/sts"
preset=${1:-list}; sfx=${2:-}
[ -n "$sfx" ] && sfx="_$sfx"
go() {  # name n timeout env desc
  local name="$1$sfx"
  if [ -f "sim_logs/honest/$name/RUN.json" ] && python3 run_status.py --json | python3 -c "import json,sys;r=[x for x in json.load(sys.stdin)['runs'] if x['name']==sys.argv[1]];sys.exit(0 if r and r[0]['state']=='running' else 1)" "$name"; then
    echo "already running: $name"; exit 1
  fi
  mkdir -p "sim_logs/honest/$name"
  setsid nohup ./run_honest.sh "$name" "$2" "$3" "$4" "$5" > "sim_logs/honest/$name/runner.out" 2>&1 < /dev/null &
  echo "launched $name (pid $!)"
}
case "$preset" in
  list)
    echo "repro_v3       mainline_v3 reproduction (clairvoyant, 150 seeds)"
    echo "honest_K1      honest combat search K=1 (150 seeds)"
    echo "honest_K4      honest combat search K=4 (150 seeds)"
    echo "honest_K8      honest combat search K=8 (150 seeds)"
    echo "honest_draft50 honest combat + honest draft teacher K=4 (50 seeds)" ;;
  repro_v3)       go repro_v3 150 1800 "" "mainline_v3 reproduction (deployed, clairvoyant replay search)" ;;
  honest_K1)      go combat_K1 150 1800 "HONEST=1 HONEST_K=1" "honest combat search, K=1" ;;
  honest_K4)      go combat_K4 150 1800 "HONEST=1 HONEST_K=4" "honest combat search, K=4" ;;
  honest_K8)      go combat_K8 150 1800 "HONEST=1 HONEST_K=8" "honest combat search, K=8" ;;
  honest_draft50) go draft_K4_50 50 14400 "HONEST=1 HONEST_K=4 DRAFT_SEARCH=1 HONEST_DRAFT_K=4 HONEST_TAIL_K=1" "honest combat K=4 + honest draft teacher K=4 (tail K=1)" ;;
  stop)
    name=${2:?run name}
    f=$(ls sim_logs/*/RUN.json sim_logs/*/*/RUN.json 2>/dev/null | while read p; do python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(sys.argv[1] if m.get('name')==sys.argv[2] else '')" "$p" "$name"; done | grep . | head -1)
    [ -z "$f" ] && { echo "no such run: $name"; exit 1; }
    pg=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('pgid') or '')" "$f")
    [ -z "$pg" ] && { echo "no pgid for $name"; exit 1; }
    python3 -c "import json,sys;p=sys.argv[1];m=json.load(open(p));m['stopped']=True;json.dump(m,open(p,'w'),indent=1)" "$f"
    kill -TERM -- -"$pg" 2>/dev/null; sleep 1; kill -KILL -- -"$pg" 2>/dev/null
    echo "stopped $name (pgid $pg)" ;;
  *) echo "unknown preset $preset"; exit 2 ;;
esac
