#!/bin/bash
# keepalive.sh runs this every 15 min: restart the overnight runner / job-queue runner if they died
cd "$HOME/sts"
echo "[$(date '+%F %T')] watchdog" >> sim_logs/overnight/watchdog.log
if [ ! -f sim_logs/overnight/HANDED_TO_JOBQ ] && { grep -q '"state": "todo"\|"state": "running"' sim_logs/overnight/state.json 2>/dev/null || [ ! -f sim_logs/overnight/state.json ]; }; then
  if ! { [ -f sim_logs/overnight/runner.pid ] && kill -0 "$(cat sim_logs/overnight/runner.pid)" 2>/dev/null; }; then
    echo "[$(date '+%F %T')] overnight runner not alive -> restarting" >> sim_logs/overnight/watchdog.log
    setsid nohup ./overnight.sh >> sim_logs/overnight/runner.nohup 2>&1 < /dev/null &
  fi
fi
if grep -q '"state": "pending"\|"state": "running"' sim_logs/jobs/queue.json 2>/dev/null && ! pgrep -f "jobq.py run" > /dev/null; then
  echo "[$(date '+%F %T')] job-queue runner -> starting" >> sim_logs/overnight/watchdog.log
  setsid nohup python3 jobq.py run >> sim_logs/jobs/runner.nohup 2>&1 < /dev/null &
fi
# boss-fight combat RL (detached, resumable): restart if it died before finishing its iterations
if [ -f nets/boss1/history.json ] && ! pgrep -f "boss_rl_train.py" > /dev/null; then
  n=$(python3 -c "import json;print(len(json.load(open('nets/boss1/history.json'))['it']))" 2>/dev/null || echo 0)
  if [ "$n" -lt 100 ] && [ ! -f nets/boss1/STOP ]; then
    echo "[$(date '+%F %T')] boss RL not running ($n/100) -> restarting" >> sim_logs/overnight/watchdog.log
    setsid nohup ./boss_rl_run.sh > /dev/null 2>&1 < /dev/null &
  fi
fi
