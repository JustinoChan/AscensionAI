#!/bin/bash
# Pre-launch verification + overnight launch:
#   varD = stabilized entropy + Variant C shaped reward, 150k games
#   varE = obs-fix-only control (bare varB reward), 150k games, queued after
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate

echo "== 1: bare worker stability (pipe driver, 60 games) =="
timeout 300 python test_worker_pipe.py sim_worker_bare.py || exit 1

echo "== 2: trainer micro-run, shaped mode (300 games, no eval) =="
timeout 240 python train_sim_mp.py 300 4 100000 chkD shaped || exit 1
rm -rf sim_models_chkD

echo "== 3: trainer micro-run, bare mode (300 games, no eval) =="
timeout 240 python train_sim_mp.py 300 4 100000 chkE bare || exit 1
rm -rf sim_models_chkE

echo "== 4: launch varD (150k shaped) + queue varE (150k bare) =="
echo '150000 12 5000 varD shaped' > .sim_autorun
echo '150000 12 5000 varE bare' > .sim_queue
setsid nohup python train_sim_mp.py 150000 12 5000 varD shaped >> sim_logs/varD.log 2>&1 < /dev/null &
sleep 30
tail -n 4 sim_logs/varD.log
pgrep -fc '[t]rain_sim' || echo "LAUNCH FAILED"
echo "autorun: $(cat .sim_autorun)"
echo "queue:   $(cat .sim_queue)"
