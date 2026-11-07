#!/bin/bash
# Self-healing for the SIM trainer (cron: */10 * * * *). Mirrors the live rig's
# monitor.sh pattern: while .sim_autorun exists, relaunch train_sim_mp.py after
# any death/preemption/reboot. The trainer resumes from the tag's latest
# checkpoint, so a preemption costs at most one 5k-game milestone of progress.
# Remove ~/sts_gym_spike/.sim_autorun to disable.
D="$HOME/sts_gym_spike"
AUTORUN="$D/.sim_autorun"   # contents = trainer args, e.g.: 150000 12 5000 varB
LOG="$D/sim_logs/sim_monitor.log"
[ -f "$AUTORUN" ] || exit 0
n=$(pgrep -fc '[t]rain_sim_mp'); n=${n:-0}
if [ "$n" -eq 0 ]; then
    cd "$D" || exit 1
    source .venv/bin/activate
    args=$(cat "$AUTORUN")
    tag=$(echo $args | awk '{print $4}')
    setsid nohup python train_sim_mp.py $args >> "$D/sim_logs/${tag:-run}.log" 2>&1 < /dev/null &
    echo "$(date -u '+%F %T') RELAUNCHED: $args" >> "$LOG"
else
    echo "$(date -u '+%F %T') ok ($n procs)" >> "$LOG"
fi
