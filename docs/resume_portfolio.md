# Resume and Portfolio Framing

## Short Portfolio Description

AscensionAI is a distributed reinforcement-learning system for Slay the Spire. It combines behavior-cloning warm starts, PPO fine-tuning, a 134-action masked discrete action space, parallel rollout workers, checkpoint-aware offline training, deterministic fixed-seed evaluation, and a Windows control panel for long-running training supervision.

## Resume Bullets

- Built a reinforcement-learning system for Slay the Spire using behavior-cloning warm starts and PPO fine-tuning over a 134-action masked discrete action space.
- Implemented parallel rollout collection with multiple supervised game worker processes, checkpoint metadata, stale rollout filtering, crash recovery, and offline model updates.
- Designed deterministic fixed-seed evaluation comparing heuristic, BC, and PPO checkpoints with metrics for floor depth, shaped reward, Act 2 reach rate, win rate, and elite/boss conversion.
- Built a desktop control panel for hardware-aware worker launch, log streaming, graceful shutdown, checkpoint warm starts, and long-running training supervision.
- Added public experiment reports, an experiment registry, architecture documentation, and a static dashboard so reviewers can inspect the ML systems story without installing the game.

## Technical Summary

The project treats a commercial desktop game as a live reinforcement-learning environment and wraps it in a practical training system. The agent receives a 585-dimensional structured observation vector (19 monster power slots covering all STS1 combat buffs/debuffs), samples from a masked 134-action discrete policy, and trains with supervised imitation followed by PPO. Long-running runs are managed by a local distributed workflow: N game workers write checkpoint-tagged rollouts, a trainer consumes fresh batches, stale files are rejected, and workers periodically reload the shared checkpoint.

The current public snapshot is honest about model quality: the infrastructure is in place, BC reaches a playable warm start, and PPO mechanics are functioning with 21,900+ training games across 2,410+ update batches. The latest 200-game eval shows 38.1% boss win rate (up from 21.7%), 20% Act 2 reach rate, and 8 runs past floor 30 including two Act 3 runs (floors 42 and 46). The agent has not yet won a full game, which leaves a clear next research milestone while preserving the project as a systems, tooling, and evaluation portfolio piece.
