#!/bin/bash
# Phase 2 queue D: more samples (starts after queue C).
cd "/home/justinochan/sts"
while pgrep -f run_phase2c.sh > /dev/null; do sleep 20; done
./run_honest.sh cpp_1ply_K64 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64" "1-ply native K=64 everywhere"
./run_honest.sh cpp_adapt_32_64_128 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=32 HONEST_K_ELITE=64 HONEST_K_BOSS=128" "1-ply native, adaptive K: hallway 32 / elite 64 / boss 128"
echo PHASE2D_DONE
