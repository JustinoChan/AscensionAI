#!/bin/bash
# LLM out-of-combat teacher pilot (local Ollama). 3-seed smoke test, then the first 50 eval500 seeds,
# honest K=64 combat, PAR workers (the LLM serializes on the GPU). Paired vs eval500_K64.
# usage: llm_pilot.sh MODEL [THINK=0|1|""] [PAR]
cd "$HOME/sts"
M=${1:-qwen3:8b}; T=${2:-}; export PAR=${3:-4}
TAG=$(echo "$M" | tr ':.' '__')${T:+_think$T}
E="HONEST=1 HONEST_CPP=1 HONEST_K=64 LLM_MACRO=1 LLM_MODEL=$M${T:+ LLM_THINK=$T}"
SEED0=40009 SEEDSTEP=1000003 ./run_honest.sh llm_smoke3_$TAG 3 7200 "$E" "LLM teacher smoke test: $M think=$T, eval500 seeds 0-2"
SEED0=40009 SEEDSTEP=1000003 ./run_honest.sh llm_pilot50_$TAG 50 7200 "$E" "LLM teacher pilot: $M think=$T, eval500 seeds 0-49, K=64 combat"
python3 ledger.py compare eval500_K64 llm_pilot50_$TAG
