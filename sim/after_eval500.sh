#!/bin/bash
# waits for eval500, then launches the LLM pilot (4 workers) alongside the job queue (RL pipeline, 10 workers)
cd "$HOME/sts"
until grep -q "^n=500 " sim_logs/honest/eval500_K64/SUMMARY.txt 2>/dev/null; do sleep 60; done
./llm_pilot.sh qwen3:8b 0 4 > sim_logs/llm_pilot.nohup 2>&1
