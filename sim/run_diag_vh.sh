#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
timeout 600 python diag_value_heavy.py 2>>diag_err.log
