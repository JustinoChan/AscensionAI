#!/bin/bash
# Self-healing training monitor — runs from cron ON THE VM (independent of any
# external session). Every run it appends an auditable heartbeat line, and if
# training has died *inside its scheduled window* it relaunches it.
#
# Install (once):  crontab -l | grep -q monitor.sh || (crontab -l 2>/dev/null; \
#                  echo '*/10 * * * * /home/USER/ascension/vm/monitor.sh') | crontab -
#
# Inspect anytime:  tail ~/ascension/logs/monitor_heartbeat.log
#
# It does NOT handle spot preemption (when preempted the VM is off, so cron
# can't run) — that still needs an external start. It DOES handle the run dying
# while the VM stays up (run_training.sh crash, all workers gone, etc.).

PROJECT_DIR="$HOME/ascension"
HEARTBEAT="$PROJECT_DIR/logs/monitor_heartbeat.log"
END_FILE="$PROJECT_DIR/logs/.run_end_epoch"
STOP_FILE="$PROJECT_DIR/logs/.stop_training"
TRAINER_LOG="$PROJECT_DIR/logs/trainer.log"

now=$(date +%s)
nowts=$(date -u '+%Y-%m-%d %H:%M:%S')
java=$(pgrep -xc java 2>/dev/null || echo 0)
trainer=$(pgrep -fc '[t]rain_offline' 2>/dev/null || echo 0)
run=$(pgrep -fc '[r]un_training.sh' 2>/dev/null || echo 0)
games=$(( $(wc -l < "$PROJECT_DIR/logs/training_stats.csv" 2>/dev/null || echo 1) - 1 ))
wedged=$(grep -hc WEDGED "$PROJECT_DIR"/logs/worker_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)

# Minutes since the trainer last wrote — the freshest signal that PPO is alive.
tage="na"
[ -f "$TRAINER_LOG" ] && tage=$(( (now - $(stat -c %Y "$TRAINER_LOG")) / 60 ))

end_epoch=0
[ -f "$END_FILE" ] && end_epoch=$(cat "$END_FILE" 2>/dev/null || echo 0)

action="ok"
# Relaunch only if: not intentionally stopped, still inside the scheduled window,
# and the run is actually gone (no orchestrator and no game JVMs).
if [ ! -f "$STOP_FILE" ] && [ "$end_epoch" -gt "$now" ] && [ "$run" -eq 0 ] && [ "$java" -eq 0 ]; then
    remaining_h=$(( (end_epoch - now) / 3600 + 1 ))
    cd "$PROJECT_DIR" && nohup bash vm/run_training.sh --workers 8 --hours "$remaining_h" \
        > /tmp/training_relaunch.log 2>&1 &
    action="RELAUNCHED (run was dead, ${remaining_h}h left in window)"
fi

echo "$nowts UTC | java=$java trainer=$trainer run=$run games=$games trainer_age=${tage}m wedged=$wedged | $action" >> "$HEARTBEAT"
