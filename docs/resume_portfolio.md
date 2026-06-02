# Resume and Portfolio Framing

## Short Portfolio Description

AscensionAI is a distributed reinforcement-learning system for Slay the Spire. It combines behavior-cloning warm starts, PPO fine-tuning, a 717-d structured observation, a 134-action masked discrete action space, learned deck-building (RL-controlled card removal/upgrade), parallel rollout workers, checkpoint-aware offline training, deterministic fixed-seed evaluation, a Windows control panel, and a self-healing headless cloud deployment.

## Resume Bullets

- Built a reinforcement-learning system for Slay the Spire using behavior-cloning warm starts and PPO fine-tuning over a 134-action masked discrete action space.
- Made deck-building a learned skill: added a 717-d per-card deck observation vector and moved card removal/upgrade from a heuristic into the RL policy with a potential-based deck-quality reward, migrating the trained model 585→717 via behavior-preserving warm transfer plus a behavior-cloning anchor.
- Implemented parallel rollout collection with multiple supervised game worker processes, checkpoint metadata, stale rollout filtering, crash recovery, and offline model updates.
- Designed deterministic fixed-seed evaluation comparing heuristic, BC, and PPO checkpoints with metrics for floor depth, shaped reward, Act 2 reach rate, win rate, and elite/boss conversion.
- Built a desktop control panel for hardware-aware worker launch, log streaming, graceful shutdown, checkpoint warm starts, and long-running training supervision.
- Deployed the worker/trainer stack headless on a GPU-less GCP spot VM via a one-shot installer, running 8 game instances under per-worker Xvfb virtual displays with software OpenGL; debugged headless-specific failures (display contention, LWJGL native-extraction SIGSEGV races, Java 8 mod loading, OpenAL init, a 10 s mod handshake timeout, and a silent JVM heap OOM).
- Built self-healing, session-independent training ops: a per-worker watchdog plus a VM-side cron that continuously auto-resumes training (with an auditable heartbeat log) and a Cloud Scheduler job that restarts the spot VM after preemption — training runs constantly and recovers in ~15–25 min with no human.
- Added public experiment reports, an experiment registry, architecture documentation, and a static dashboard so reviewers can inspect the ML systems story without installing the game.

## Technical Summary

The project treats a commercial desktop game as a live reinforcement-learning environment and wraps it in a practical training system. The agent receives a 717-dimensional structured observation vector (19 monster power slots covering all STS1 combat buffs/debuffs, plus a per-card deck count vector so it can see its exact deck), samples from a masked 134-action discrete policy, and trains with supervised imitation followed by PPO. As of Path 2, deck-building is learned rather than heuristic: card removal and upgrade selection are RL-controlled, rewarded by potential-based deck-quality shaping. Long-running runs are managed by a distributed workflow: N game workers write checkpoint-tagged rollouts, a trainer consumes fresh batches, stale files are rejected, and workers periodically reload the shared checkpoint. The same stack runs locally under a Windows GUI or **headless on a GPU-less GCP spot VM** — 8 instances under per-worker Xvfb displays with software OpenGL, deployed by a single idempotent installer at ~90+ games/hour, and self-healing (continuous auto-resume + preemption auto-restart) so it runs for a few dollars a day until told to stop.

The project is honest about model quality: the infrastructure is in place, BC reaches a playable warm start, and PPO mechanics are functioning across 24,000+ training games. The last fixed-seed eval (on the 585-d model) shows 38.1% boss win rate, 20% Act 2 reach rate, and runs past floor 30 including two Act 3 runs (floors 42 and 46). The agent has not yet won a full game; the current research direction (learned deck-building) targets the Act 2 wall, leaving a clear next milestone while preserving the project as a systems, tooling, and evaluation portfolio piece.
