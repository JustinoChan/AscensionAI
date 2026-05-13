# Experiment 002: Parallel PPO Fine-Tuning

**Date:** 2026-05-12  
**Commit:** `af06ee9`  
**Primary checkpoint:** `models/ppo_sts.pt` (local artifact, intentionally gitignored)  
**Source artifacts:** `logs/training_stats.csv`, `logs/train_offline_debug.log`, `logs/fight_stats.csv`, `logs/eval_stats.csv`  
**Public snapshot:** [docs/experiments/index.json](index.json)

## Purpose

Test the local distributed training loop: multiple Slay the Spire worker processes collect checkpoint-tagged rollouts while a central offline trainer batches fresh files, rejects stale data, applies PPO updates, and atomically writes the shared checkpoint.

## Run Configuration

| Field | Value |
|---|---:|
| Rollout workers observed | 3 |
| Completed rollout games in `training_stats.csv` | 160 |
| PPO update rows | 19 |
| Rollouts consumed by trainer | 152 |
| Latest trainer transition total | 21,658 |
| Batch size | 8 rollout files per PPO update |
| PPO epochs | 4 |
| Initial learning rate | 0.00003 |
| Latest auto-tuned learning rate | 0.00001875 |
| Clip range | 0.15 |
| Target KL | 0.03 |
| Entropy coefficient | 0.001 initial, auto-tuned |
| Max rollout lag | 4 updates |
| Stale rollouts | 0 |
| Hardware notes | Windows 11 workstation, Ryzen 7 9800X3D, 32 GB RAM, Radeon RX 9070 XT |

Representative trainer command:

```powershell
python scripts\train_offline.py --model models\ppo_sts.pt --data rollouts_shared --delete-consumed --batch-games 8 --lr 3e-5 --bc-coef 0.10 --max-rollout-lag 4 --ent-coef 0.001 --auto-tune
```

## Training Outcome Snapshot

| Metric | Value |
|---|---:|
| Average final floor | 13.15 |
| Average shaped reward | -0.47 |
| Win rate | 0.0% |
| Act 2 reach rate | 11.9% |
| Floor 20+ rate | 8.8% |
| Elite fights / wins | 193 / 143 |
| Boss fights / wins | 74 / 19 |
| Latest approximate KL | 0.0364123 |
| Latest clip fraction | 0.189614 |
| Latest explained variance | 0.177785 |
| Latest early-stop flag | 1 |

## Fixed-Seed Evaluation

Evaluation used the same 25-seed slice as the BC report.

| Policy | Games | Avg floor | Avg reward | Win rate | Act 2 reach | Floor 20+ | Elite W/L | Boss W/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC checkpoint | 25 | 13.08 | -0.62 | 0.0% | 12.0% | 12.0% | 25/32 | 3/10 |
| PPO checkpoint | 25 | 13.08 | -0.62 | 0.0% | 12.0% | 12.0% | 25/32 | 3/10 |

## Interpretation

The systems side of the parallel PPO loop is working: workers produced rollouts, the trainer consumed batches, checkpoint metadata stayed fresh enough that no stale rollouts were rejected, and the auto-tuner reacted to a high-KL update by lowering the learning rate and entropy coefficient. The model-quality result is not yet a win: the current PPO checkpoint did not separate from the BC checkpoint on the fixed-seed slice.

The most important public-facing point is honest: AscensionAI already has the infrastructure for distributed rollout collection and controlled evaluation, while the present training run is still early and has not demonstrated convergence.

## Next Action

Continue PPO for a longer horizon with fixed-seed evaluations every 250-500 rollout games. Preserve each checkpoint under a run-specific name before overwriting `models/ppo_sts.pt`, then add the result to `docs/experiments/index.json` so regression and improvement are visible over time.
