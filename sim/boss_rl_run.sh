#!/bin/bash
# boss-fight combat RL (detached, resumable): 4 workers alongside the job queue
cd "$HOME/sts"
BOSS_WORKERS=${BOSS_WORKERS:-4} nice -n 5 .venv_train/bin/python boss_rl_train.py ${ITERS:-100} >> nets/boss1/train.log 2>&1
