# Experiment 001: Behavior-Cloning Baseline

**Date:** 2026-05-12  
**Commit:** `af06ee9`  
**Primary checkpoint:** `models/ppo_sts_bc.pt` (local artifact, intentionally gitignored)  
**Source artifacts:** `logs/bc_stats.csv`, `logs/bc_train_stats.csv`, `logs/eval_stats.csv`  
**Public snapshot:** [docs/experiments/index.json](index.json)

## Purpose

Establish the supervised warm-start baseline before PPO fine-tuning. The behavior-cloning policy imitates the project heuristic across full-game demonstrations so PPO starts from a legal, game-progressing policy instead of random exploration over the 134-action masked action space.

## Run Configuration

| Field | Value |
|---|---:|
| BC collection rows | 410 games across 3 collector runs |
| Supervised training samples | 86,297 |
| Train / validation split | 77,667 / 8,630 |
| Final epoch | 41 |
| Learning rate | 0.0005 |
| Batch size | 256 |
| Weight decay | 0.00001 |
| Label smoothing | 0.02 |
| Patience | 12 |
| Hardware notes | Windows 11 workstation, Ryzen 7 9800X3D, 32 GB RAM, Radeon RX 9070 XT |

The tracked repository does not include raw checkpoints or logs. Those remain local runtime artifacts under `models/` and `logs/`; this report records the publishable summary.

## Supervised Training Result

| Metric | Value |
|---|---:|
| Validation accuracy, first epoch | 77.068% |
| Validation accuracy, final epoch | 84.948% |
| Best validation loss | 0.395827 |
| Choice accuracy | 84.172% |
| Targeted-card accuracy | 83.891% |
| End-turn accuracy | 99.825% |

## BC Collection Outcome Snapshot

The most recent complete collector slice contained 134 games.

| Metric | Value |
|---|---:|
| Average final floor | 15.43 |
| Win rate | 0.0% |
| Act 2 reach rate | 21.6% |
| Floor 20+ rate | 19.4% |
| Elite fights / wins | 169 / 142 |
| Boss fights / wins | 83 / 29 |

## Fixed-Seed Evaluation

Evaluation used the first 25 seeds from `seeds/eval_200.txt`.

| Policy | Games | Avg floor | Avg reward | Win rate | Act 2 reach | Floor 20+ | Elite W/L | Boss W/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heuristic | 25 | 16.60 | 11.38 | 0.0% | 32.0% | 32.0% | 27/33 | 8/18 |
| BC checkpoint | 25 | 13.08 | -0.62 | 0.0% | 12.0% | 12.0% | 25/32 | 3/10 |

## Interpretation

The BC model learned a usable policy over the action surfaces, with validation accuracy near 85% and strong end-turn recognition. On the 25-seed evaluation slice it still trailed the source heuristic on floor depth, reward, Act 2 reach rate, and boss conversion. This is expected for a first warm start: the model is competent enough to launch PPO but has not yet exceeded the heuristic that generated the data.

## Next Action

Use the BC checkpoint as the initialization for parallel PPO, then evaluate the PPO checkpoint on the same seed file after each material training interval. The immediate success criterion is not win rate; it is PPO matching and then exceeding the BC fixed-seed floor/reward metrics without increasing invalid-action or stale-rollout rates.
