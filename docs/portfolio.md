# AscensionAI Portfolio Page

AscensionAI is a distributed reinforcement-learning system for Slay the Spire. It combines behavior-cloning warm starts, PPO fine-tuning, a 717-d structured observation, a 134-action masked discrete action space, **learned deck-building** (the policy observes its full deck and chooses card removals/upgrades), parallel rollout workers, checkpoint-aware offline training, deterministic fixed-seed evaluation, a Windows control panel, and a self-healing headless cloud deployment.

![Architecture diagram](assets/architecture.svg)

## What Makes It Non-Trivial

| Area | Implementation |
|---|---|
| ML training | Behavior cloning, PPO, GAE, entropy control, target-KL early stopping, BC anchor loss. |
| Learned deck-building | 717-d per-card deck count vector + RL-controlled card removal/upgrade + potential-based deck-quality reward; warm-transferred 585→717 with behavior preserved. |
| Environment integration | Live Slay the Spire process controlled through ModTheSpire, Communication Mod, and SpireComm. |
| Action safety | 134 discrete actions masked per game state so illegal commands are not sampled. |
| Distributed systems | N game workers write checkpoint-tagged rollouts; one trainer consumes fresh files and rejects stale data. |
| Evaluation | Heuristic, BC, and PPO policies run on the same deterministic seed file with comparable CSV metrics. |
| Tooling | Desktop control panel launches workers, tails logs, recommends worker counts, and cleans up orphaned processes. |
| Cloud / DevOps | One-shot installer deploys the whole stack **headless on a GPU-less GCP spot VM**; runs constantly and **self-heals** — a VM cron auto-resumes training (with an auditable heartbeat) and Cloud Scheduler restarts the VM after preemption. |

## Learned Deck-Building (Path 2)

The hardest part of Slay the Spire is building a deck that scales — and originally the agent *couldn't learn it*: the encoder only saw a coarse aggregate of the deck, and card removal/upgrade were chosen by a heuristic, not the policy. Path 2 fixed both:

- **Observe the deck** — added a 132-dim per-card count vector over the full Ironclad pool (observation 585 → 717), so the policy sees exactly which/how many of each card it holds.
- **Control deck-building** — card removal (purge) and upgrade (smith) selection moved from the heuristic into the RL policy.
- **Reward deck quality** — replaced the flat per-removal/per-upgrade rewards with potential-based shaping on mean card quality, so cutting junk and drafting/upgrading good cards is rewarded and cutting a key card is penalized, all context-dependent.

The trained 585-d model was **warm-transferred to 717-d** (behavior preserved — new inputs zero-initialized), de-risked with a light BC anchor of fresh heuristic demos, and the learning rate was re-raised so the new inputs learn at a useful pace.

Progress is tracked with a **weight-ratio diagnostic** — the mean magnitude of the 132 new first-layer input columns vs the original 585. It has climbed from ~0.14 to **~0.59** (the deck inputs are now over half the strength of the inputs the model spent its whole prior life learning). Behavior hasn't broken from baseline yet, which is expected this early — and the tuning has been deliberately conservative: one BC-anchor change was tried, regressed training behavior, and was **reverted** (the noisy sampled training-floor is a poor thing to over-tune against; a clean greedy fixed-seed eval is planned once the ratio reaches ~0.70).

## Headless Cloud Deployment

The training loop also runs unattended on a **GCP `c3-standard-22` spot VM** (22 vCPU, no GPU, no display). A single idempotent installer (`vm/install.sh`) provisions Java 8, Xvfb, OpenAL, and a CPU PyTorch venv; one launcher (`vm/run_training.sh`) brings up 8 headless workers plus the offline trainer at ~90+ games/hour.

Making a GUI-bound, mod-loaded desktop game run many times over on a headless server required solving a chain of non-obvious failures: giving each worker its **own Xvfb display** (a shared display ran ~100× slower because OpenGL serializes across windows) and its **own JVM tmpdir** (shared `/tmp` caused LWJGL native-extraction SIGSEGV races), pinning **Java 8** (mods silently fail to load on 17+), wiring up headless **OpenAL**, signaling CommunicationMod's READY handshake **before** the slow `import torch` (its timeout is 10 s), and fixing a silent JVM heap OOM that had been killing workers after ~35 games (a 2 GB heap + 25-game restart lifted throughput from ~55 to ~90+ games/hour). The run is fully **self-healing**: a per-worker watchdog relaunches a wedged JVM, a VM-side cron continuously auto-resumes training (with a 10-minute heartbeat log) after any death or reboot, and a Cloud Scheduler job restarts the VM after spot preemption — so training runs constantly with no session and only stops when told to.

## Public Demo

| Asset | Preview |
|---|---|
| Control panel and log supervision | ![Control panel preview](assets/control_panel_preview.svg) |
| Worker/trainer loop GIF | ![Worker launch demo](assets/worker_launch_demo.gif) |
| Training plot snapshot | ![Training plot](assets/training_plot.png) |

Open the [static results dashboard](dashboard/index.html) to inspect the embedded public snapshot or load local CSV files from a fresh run.

## Current Results

| Policy | Games | Avg floor | Avg reward | Win rate | Act 2 reach | Floor 20+ |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic | 150 | 15.78 | 8.44 | 0.0% | 26.0% | 23.3% |
| BC checkpoint | 150 | 12.81 | -0.55 | 0.0% | 12.0% | 12.0% |
| PPO checkpoint | 150 | 14.70 | 2.37 | 0.0% | 18.7% | 18.0% |

The project is presented honestly: the current PPO checkpoint beats the BC checkpoint on the wider eval, but remains behind the heuristic baseline and has not yet recorded a full win. (These numbers are the last fixed-seed benchmark on the 585-d model; the 717-d learned-deck-building model of Path 2 is training and will be evaluated on the same seeds once its new inputs mature.) The engineering value is in the complete training system, deterministic evaluation harness, dashboard, and reproducible reporting pipeline that make future improvements measurable.

## Links

- [Experiment reports](experiments/)
- [Architecture documentation](architecture.md)
- [Technical writeup](AscensionAI_Technical_Writeup.md)
- [Resume bullets and summary](resume_portfolio.md)
