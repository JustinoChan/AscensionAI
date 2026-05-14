# Experiment 004: Long PPO Run and Fixed-Seed Evaluation

**Date:** 2026-05-14  
**Commit:** `2670c91`  
**Primary checkpoint:** `models/ppo_sts.pt` (local artifact, intentionally gitignored)  
**Seed file:** `seeds/eval_200.txt`  
**Source artifacts:** `logs/training_stats.csv`, `logs/eval_stats.csv`, `logs/bc_stats.csv`, `logs/bc_train_stats.csv`, `logs/fight_stats.csv`, `logs/training_plot.png`  
**Public snapshot:** [docs/experiments/index.json](index.json)

## Purpose

Publish the first long PPO result after extending parallel rollout training to roughly 2,500 completed games, then compare the current PPO checkpoint against the heuristic and BC checkpoints on the same 25-seed evaluation slice.

## Run Configuration

| Field | Value |
|---|---:|
| Rollout workers observed | 3 |
| Completed PPO games in `training_stats.csv` | 2,496 |
| PPO update rows | 311 |
| Total trainer update transitions | 372,305 |
| Batch size | 8 rollout files per PPO update |
| PPO epochs | 4 |
| Initial learning rate | 0.00003 |
| Latest auto-tuned learning rate | 2.2888184e-05 |
| Latest entropy coefficient | 0.00117128 |
| Latest BC anchor coefficient | 0.01 |
| Clip range | 0.15 |
| Target KL | 0.03 |
| Max rollout lag | 4 updates |
| Stale / legacy / skipped rollouts | 0 / 0 / 0 |

## Training Outcome Snapshot

| Metric | Full run | First 500 | Last 500 | Last 100 |
|---|---:|---:|---:|---:|
| Games | 2,496 | 500 | 500 | 100 |
| Average final floor | 13.80 | 12.99 | 14.92 | 15.00 |
| Median final floor | 16 | 14 | 16 | 16 |
| Best final floor | 50 | 31 | 46 | 46 |
| Average shaped reward | 1.10 | -1.13 | 3.48 | 3.65 |
| Act 2 reach rate | 13.4% | 7.8% | 19.6% | 17.0% |
| Floor 20+ rate | 12.3% | 6.6% | 17.0% | 15.0% |
| Win rate | 0.0% | 0.0% | 0.0% | 0.0% |

Latest trainer row:

| Metric | Value |
|---|---:|
| Approximate KL | 0.00797181 |
| Clip fraction | 0.106109 |
| Normalized entropy | 0.256081 |
| Explained variance | 0.677278 |
| Auto-tune action | `healthy:bc_slow_down` |
| Early-stop rows | 3 |

## Fixed-Seed Evaluation

Evaluation used the same 25-seed slice for the heuristic, BC checkpoint, and current PPO checkpoint.

| Policy | Model | Games | Avg floor | Best floor | Avg reward | Win rate | Act 2 reach | Floor 20+ | Elite W/L | Boss W/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heuristic | `heuristic` | 25 | 16.84 | 29 | 11.87 | 0.0% | 32.0% | 32.0% | 28/33 | 8/18 |
| BC checkpoint | `models/ppo_sts_bc.pt` | 25 | 13.08 | 23 | -0.62 | 0.0% | 12.0% | 12.0% | 25/32 | 3/10 |
| PPO checkpoint | `models/ppo_sts.pt` | 25 | 16.40 | 36 | 9.57 | 0.0% | 24.0% | 24.0% | 21/26 | 7/16 |

## Fight Log Cross-Check

| Source | Fight rows | Elite W/L | Boss W/L | Terminal losses |
|---|---:|---:|---:|---:|
| PPO training workers | 4,204 | 2245/2920 | 340/1284 | 1,619 |
| May 14 eval set | 135 | 74/91 | 18/44 | 43 |

## Interpretation

The long PPO run is the first public snapshot where PPO separates from the BC checkpoint on the fixed-seed slice. PPO improved from BC's 13.08 average floor to 16.40, raised average shaped reward from -0.62 to 9.57, and doubled Act 2 reach from 12.0% to 24.0%.

The heuristic is still narrowly ahead on this 25-seed slice: 16.84 average floor versus PPO's 16.40. PPO's best evaluated run reached floor 36, above the heuristic best of floor 29, but neither policy recorded a full victory in this snapshot.

The training curve also improved late in the run. The first 500 PPO games averaged floor 12.99; the last 500 averaged 14.92. That is a useful signal, but the run still needs larger fixed-seed evaluation before treating the result as stable.

## Next Action

Run a larger deterministic evaluation, ideally the full 200-seed file, before changing the public claim from "PPO nearly matches the heuristic" to anything stronger. Preserve this checkpoint before further PPO updates so the May 14 result remains reproducible.
