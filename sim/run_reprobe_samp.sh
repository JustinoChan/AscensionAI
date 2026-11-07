#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
timeout 700 python reprobe_fight_sampling.py 2>>diag_err.log
