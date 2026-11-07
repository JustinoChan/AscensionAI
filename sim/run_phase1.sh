#!/bin/bash
# Phase 1 queue: honest combat arms (150 seeds), then honest draft teacher (50 seeds).
cd "$HOME/sts"
R=./run_honest.sh
$R combat_K1 150 1800 "HONEST=1 HONEST_K=1"
$R combat_K4 150 1800 "HONEST=1 HONEST_K=4"
$R combat_K8 150 1800 "HONEST=1 HONEST_K=8"
$R combat_K4_macro 150 1800 "HONEST=1 HONEST_K=4 HONEST_MACRO=1"
$R draft_K4_50 50 14400 "HONEST=1 HONEST_K=4 DRAFT_SEARCH=1 HONEST_DRAFT_K=4 HONEST_TAIL_K=1"
echo PHASE1_QUEUE_DONE
