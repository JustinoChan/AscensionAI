# Experiment 003: Fixed-Seed Evaluation

**Date:** 2026-05-12  
**Commit:** `af06ee9`  
**Seed file:** `seeds/eval_200.txt`  
**Games per policy:** 25  
**Source artifact:** `logs/eval_stats.csv`  
**Public snapshot:** [docs/experiments/index.json](index.json)

## Purpose

Compare the heuristic, BC checkpoint, and current PPO checkpoint on the same deterministic seed slice. Fixed-seed evaluation keeps the comparison focused on policy quality instead of run-to-run variance from map generation and combat randomness.

## Evaluation Commands

```powershell
python scripts\eval_model.py --policy heuristic --games 25 --seed-file seeds\eval_200.txt --run-tag heuristic_25_20260512_112612
python scripts\eval_model.py --model models\ppo_sts_bc.pt --games 25 --seed-file seeds\eval_200.txt --run-tag bc_25_20260512_112612
python scripts\eval_model.py --model models\ppo_sts.pt --games 25 --seed-file seeds\eval_200.txt --run-tag ppo_current_25_20260512_112612
```

## Results

| Policy | Model | Games | Avg floor | Avg reward | Win rate | Act 2 reach | Floor 20+ | Elite fights | Elite wins | Boss fights | Boss wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heuristic | `heuristic` | 25 | 16.60 | 11.38 | 0.0% | 32.0% | 32.0% | 33 | 27 | 18 | 8 |
| BC checkpoint | `models/ppo_sts_bc.pt` | 25 | 13.08 | -0.62 | 0.0% | 12.0% | 12.0% | 32 | 25 | 10 | 3 |
| PPO checkpoint | `models/ppo_sts.pt` | 25 | 13.08 | -0.62 | 0.0% | 12.0% | 12.0% | 32 | 25 | 10 | 3 |

## Interpretation

The heuristic remains the strongest policy in this snapshot. BC produced a playable warm-start but lost depth and reward against the heuristic. PPO did not yet improve the fixed-seed result, which matches the early training interpretation from Experiment 002.

This evaluation is still valuable because the harness is deterministic, repeatable, and already captures the metrics needed to detect future improvement: average floor, shaped reward, win rate, Act 2 reach, floor 20+ rate, and elite/boss conversion.

## Next Action

Expand this comparison to the full 200-seed file after a longer PPO run. If PPO remains identical to BC, verify checkpoint replacement, policy loading, and trainer update magnitude before increasing exploration.
