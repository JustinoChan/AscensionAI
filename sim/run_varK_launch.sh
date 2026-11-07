#!/bin/bash
# varK: train-time-planning expert iteration. ATTENDED run -- the cron watchdog
# only relaunches train_sim_mp via .sim_autorun, so we deliberately do NOT write
# .sim_autorun (that would launch the wrong trainer). No auto-resume on preempt.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f .sim_autorun
python export_live.py sim_models_varJ/ppo_sim_latest.pt policy_live.npz
setsid nohup python train_search.py 1500000 12 100000 varK \
  >> sim_logs/varK.log 2>&1 < /dev/null &
sleep 45
tail -n 12 sim_logs/varK.log
pgrep -fc '[t]rain_search' || echo "LAUNCH FAILED"
