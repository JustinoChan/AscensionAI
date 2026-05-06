# AscensionAI

## About

Reinforcement learning agent designed to play **Slay the Spire** using self-play, neural networks, and environment simulation.

## Architecture

```
STS Game  <-->  Communication Mod  <-->  Python Agent (stdin/stdout)
                                              |
                                     obs_encoder (530-d vector)
                                     sts_gym_env (134 discrete actions)
                                     PPOTrainer (Actor-Critic MLP)
```

- **Decision screens** (combat, events, card rewards, rest, boss relics, map pathing) are handled by the RL policy network
- **Screen plumbing** (combat rewards, shops, chests, grid select, command-error recovery) is auto-handled with fallback logic for edge cases such as full potions, boss relics, map transitions, card matching events, and empty choice lists
- The observation encoder captures player state, hand cards, monster identity/behavior/intent/powers, screen context, and map path lookahead
- Action masking ensures only legal actions are chosen
- The Python side exposes the game as a Gymnasium-compatible environment, while Communication Mod handles the live STS process bridge

### Training Pipeline

1. **Behavior Cloning (BC)** — a heuristic plays games while the neural network learns to imitate via cross-entropy loss
2. **PPO Fine-Tuning** — the BC-initialized policy improves through online RL with entropy annealing, preserving BC knowledge while exploring better strategies
3. **Parallel Scaling** — multiple game instances collect rollouts independently; an offline trainer merges and updates the shared model

The `train_bc_ppo.py` script runs steps 1 and 2 end-to-end in a single session.

### Monster Knowledge Base

The observation encoder includes a built-in database of all **66 STS monsters** (sourced from spire-archive.com), giving the agent immediate knowledge of each enemy it faces. Per monster slot, the encoder provides:

- **Identity embedding** (8 dims) — unique fingerprint per monster so the network can distinguish enemies
- **Move history** (3 dims) — current, last, and second-last move IDs to help predict attack patterns
- **Behavioral flags** (7 dims) — pre-computed traits: enrages on skills, splits at low HP, scales strength, multi-attacker, retaliates, escapes, spawns minions

This means the agent doesn't need thousands of games to rediscover that Gremlin Nob punishes skill cards or that Cultist gains strength every turn — it knows from the first encounter.

## Prerequisites

1. **Slay the Spire** (Steam)
2. **Mod the Spire** — [GitHub](https://github.com/kiooeht/ModTheSpire)
3. **BaseMod** — [GitHub](https://github.com/daviscook477/BaseMod)
4. **Communication Mod** — [GitHub](https://github.com/ForgottenArbiter/CommunicationMod)
5. **Super Fast Mode** (recommended) — [GitHub](https://github.com/Skrelpoid/SuperFastMode) — raises the in-game speed cap well beyond vanilla Fast Mode
6. **Python 3.10+**

AscensionAI is currently developed and documented for **Windows**. The GUI launcher, PowerShell helper, Steam/ModTheSpire process handling, and `%LOCALAPPDATA%` CommunicationMod config paths assume a Windows setup.

Enable **Fast Mode** in STS game settings (Settings → Fast Mode ON). If you install Super Fast Mode, push the speed slider to 200%+ in its mod config — this alone can 2-3× training throughput on top of Fast Mode.

## Installation

```bash
git clone https://github.com/JustinoChan/AscensionAI.git
cd AscensionAI

python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e external/spirecomm
```

## Quick Start

The easiest way to use AscensionAI is through the **Control Panel** GUI:

```
Double-click AscensionAI.pyw
```

Or from a terminal:

```bash
python AscensionAI.pyw
```

The control panel detects your hardware, recommends how many STS instances to run, and lets you start/stop training with one click. Live log output from all workers and the trainer is displayed in tabbed panels. It also exposes the BC game count, PPO game count, entropy coefficient, and a verbose logging toggle so long warm-up or overnight sessions can be inspected without editing scripts.

### Control Panel modes

| Mode | What it does |
|------|-------------|
| **Parallel Workers** | Multiple STS instances collecting rollouts + offline trainer |
| **Collect Rollouts (No Training)** | Multiple STS instances that only write `.npz` rollouts — no local training. For collaborators contributing data to someone else's trainer. |
| **Single-Instance Training** | One STS instance running PPO training |
| **BC → PPO (End-to-End)** | Behavior cloning warm-start then PPO fine-tuning in one session |
| **Behavior Cloning** | Heuristic plays 50 games, network learns to imitate |
| **Evaluation (Greedy)** | Run games with the trained model (no exploration, no updates) |
| **Game Logger (Passive)** | Record game states while you play manually |
| **Play Game (No AI)** | Launch STS without any AI command — play normally or configure mods |

### Recommended workflow

1. **BC → PPO** — select "BC → PPO (End-to-End)" mode and click Start. The heuristic plays 50 games, the network imitates, then PPO fine-tunes for 200 games with entropy annealing.
2. **Parallel Workers** — once you have a baseline model, switch to "Parallel Workers" for faster training. Multiple STS instances collect experience while an offline trainer updates the model.
3. **Evaluation** — select "Evaluation (Greedy)" to benchmark your model without exploration noise.

During a healthy first run, you should see live output in the Control Panel tabs, new log files appear under `logs/`, and the trained checkpoint saved to `models/ppo_sts.pt`.

### Long-running reliability

AscensionAI is designed to run multiple live STS instances for many hours, but modded STS can still crash or expose awkward intermediate screens. The launcher and screen handler include guardrails for the common stuck states:

- **Worker relaunch** — parallel workers are monitored and relaunched through ModTheSpire if an instance exits unexpectedly.
- **Reward screens** — combat reward, card reward, boss relic, chest, and map transitions are guarded against reopen loops.
- **Boss relics** — boss relic choices are treated as mandatory; the action mask suppresses leave/proceed while relic choices are available.
- **Events** — event choices use CommunicationMod's real `choice_index`, which matters for events with disabled or hidden options.
- **Card grids** — grid confirmations, hand-selection screens, and Match and Keep avoid repeated invalid selections.
- **Shops** — each shop room is entered once per floor so canceling out of the shop does not bounce back into it forever.
- **Command errors** — rejected commands recover through a conservative state/choose fallback instead of repeatedly sending invalid no-ops.

For the first long run after a patch, enable **Verbose Logs** in the Control Panel. It passes `--verbose` to every launch mode and also sets CommunicationMod `verbose=true`, producing step-by-step logs for BC, PPO, workers, evaluation, and passive logging.

### Stopping a run

The Control Panel offers two ways to stop:

- **Stop Now** — immediately kills the trainer and every STS instance. Use this if something is wrong or you need the machine back fast. In-flight games are discarded.
- **Finish && Stop** — graceful shutdown. Each worker is closed individually the moment its current game ends, so the first finisher doesn't sit idle waiting for the slowest one. The trainer keeps running until the last worker is done so its data still gets used.

Both buttons also sweep for orphaned `java.exe` processes belonging to STS — the ModTheSpire launcher detaches the real JVM, so a naive `taskkill` on the launcher PID would otherwise leave game windows behind.

### Collaborating — pooling data across multiple machines

Multiple people can contribute rollouts to a single shared model:

1. The **main trainer** sends their current `models/ppo_sts.pt` to each collaborator so everyone plays with the same policy.
2. **Collaborators** open the Control Panel, select **"Collect Rollouts (No Training)"**, and run some sessions. Rollouts accumulate in `rollouts_shared/` and are never consumed locally.
3. When done, collaborators zip their `rollouts_shared/` folder and send it to the main trainer.
4. The **main trainer** extracts the zip into their own `rollouts_shared/` (files merge cleanly — filenames embed a Unix timestamp so no collisions) and runs any training mode. The offline trainer consumes every `.npz` regardless of origin.
5. The updated `.pt` is shared back for the next round.

## Optimizing Training Speed

Training an RL agent on Slay the Spire is compute-bound by real-time game simulation — each game takes minutes, and PPO typically needs thousands of games to converge. The project ships with several optimizations; here's how to get the most out of your setup:

### In-game speedups (biggest wins)

- **Fast Mode** (Settings → Fast Mode ON) — skips most combat animations.
- **Super Fast Mode** mod — [github.com/Skrelpoid/SuperFastMode](https://github.com/Skrelpoid/SuperFastMode) — adds a game-speed slider. Running at 200–250% stacks on top of Fast Mode for a 2-3× throughput gain.
- **Minimize the STS windows** during training — the engine often runs faster when it doesn't have to render.

### Training hyperparameters (already applied by default)

- **Batched PPO updates** — the trainer accumulates 4 games of transitions per gradient update (`--games-per-update 4`). Larger batches mean less noisy gradients and less time spent blocked on updates. Raise to `8` if you have plenty of memory; drop to `2` for faster feedback during debugging.
- **4 PPO epochs per update** — down from a conservative 10. Each update trains ~2.5× faster with minimal quality loss.
- **Conservative exploration defaults** — the Control Panel uses entropy `0.10` by default for PPO-style modes. BC -> PPO anneals the selected start value toward `0.01`, while the parallel offline trainer defaults to `--ent-coef 0.10`, `--lr 1e-4`, and `--clip 0.15` to reduce catastrophic forgetting from the BC warm start.

### Scaling with hardware

- **Parallel Workers** — the Control Panel auto-detects RAM/CPU and recommends a worker count. If CPU usage stays under 70% during training, bump workers manually for roughly linear throughput.
- **Multiple machines** — see the Collaborating section. A secondary machine running "Collect Rollouts (No Training)" adds rollout data at zero cost to the main trainer's responsiveness.

### What's NOT worth optimizing

- **A GPU is unnecessary** — the policy/value network is a tiny 256×256 MLP; CPU inference is fast. The code explicitly disables CUDA.
- **STS graphics settings** — animations are the bottleneck, not rendering quality. Leave graphics on whatever is stable.

## Command-Line Alternative

If you prefer the terminal, use `launch_workers.ps1`:

| Command | What it does |
|---------|-------------|
| `.\launch_workers.ps1 -Mode bc-ppo` | BC warm-start → PPO fine-tuning (recommended first run) |
| `.\launch_workers.ps1 -Mode bc-ppo -BCGames 100` | Same, with 100 BC games instead of 50 |
| `.\launch_workers.ps1 -Mode train` | 1 STS instance running PPO training |
| `.\launch_workers.ps1` | 3 parallel rollout workers (default) |
| `.\launch_workers.ps1 -NumWorkers 4` | 4 parallel rollout workers |
| `.\launch_workers.ps1 -Mode eval -Games 30` | 30-game greedy evaluation |
| `.\launch_workers.ps1 -Mode logger` | Passive game state recorder |

For parallel mode, start the offline trainer in a separate terminal:

```bash
python scripts\train_offline.py --model models\ppo_sts.pt --data rollouts_shared --delete-consumed --ent-coef 0.10 --clip 0.15
```

## Manual Configuration

If you prefer not to use the launcher, edit the CommunicationMod config directly:

**Config path:** `%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`

### BC → PPO end-to-end

```properties
command=C\:/AscensionAI/.venv/Scripts/python.exe C\:/AscensionAI/scripts/train_bc_ppo.py --bc-games 50 --ppo-games 200 --save models/ppo_sts.pt --ent-start 0.10
```

### Single-instance PPO training

```properties
command=C\:/AscensionAI/.venv/Scripts/python.exe C\:/AscensionAI/scripts/train_ppo.py --save models/ppo_sts.pt --resume models/ppo_sts.pt --save-every 5 --ent-coef 0.10
```

### Behavior cloning only

```properties
command=C\:/AscensionAI/.venv/Scripts/python.exe C\:/AscensionAI/scripts/behavior_clone.py --games 50 --save models/ppo_sts.pt
```

### Parallel rollout workers

For each STS instance, set a unique `--id`:

```properties
command=C\:/AscensionAI/.venv/Scripts/python.exe C\:/AscensionAI/scripts/rollout_worker.py --model models/ppo_sts.pt --out rollouts_shared --id 1
```

Then run the offline trainer separately:

```bash
python scripts/train_offline.py --model models/ppo_sts.pt --data rollouts_shared --delete-consumed --ent-coef 0.10 --clip 0.15
```

Append `--verbose` to any manual command when you want step-by-step decision logging.

## Log Files

All logs are written to the `logs/` directory:

When **Verbose Logs** is enabled in the Control Panel, or when a script is launched with `--verbose`, these files include detailed per-step screen names, choices, selected actions, masks, rollout loading, and recovery events.

| Log file | Source |
|----------|--------|
| `logs/train_bc_ppo_debug.log` | `train_bc_ppo.py` — BC + PPO end-to-end progress |
| `logs/train_debug.log` | `train_ppo.py` — PPO-only training progress |
| `logs/bc_debug.log` | `behavior_clone.py` — demo collection + supervised training |
| `logs/worker_N_debug.log` | `rollout_worker.py` — per-worker game progress |
| `logs/train_offline_debug.log` | `train_offline.py` — offline PPO update stats |
| `logs/eval_debug.log` | `eval_model.py` — evaluation results |
| `logs/game_logger_debug.log` | `game_logger.py` — passive game state logging |
| `logs/bug_debug.log` | Stuck-state detection dumps for debugging freezes |
| `logs/control_panel_debug.log` | `AscensionAI.pyw` — GUI launch, process PIDs, kill results, errors |
| `logs/training_stats.csv` | Per-game training metrics (floor, HP, reward, loss, elite/boss stats) |
| `logs/fight_stats.csv` | Per-fight elite and boss encounter details (monsters, HP before/after, win/loss, terminal loss handling) |
| `logs/elite_stats.csv` | Legacy per-fight elite/boss log kept for older tooling compatibility |

Training stats can be visualized with:

```bash
python scripts/plot_training.py --save logs/training_plot.png
```

## Current Limitations

- The project currently targets **Ironclad** only.
- Training assumes normal live STS gameplay through CommunicationMod; it is not a fast headless simulator.
- Some screens are intentionally heuristic-driven while the RL policy handles the main decision surfaces. This keeps training moving, but means the learned policy is not yet responsible for every possible game choice.
- The README examples assume a local Windows install at `C:/AscensionAI`; adjust paths if your clone lives elsewhere.

## Project Structure

```
AscensionAI/
├── AscensionAI.pyw           # Control Panel GUI (double-click to launch)
├── launch_workers.ps1        # CLI launcher alternative (PowerShell)
├── scripts/
│   ├── train_bc_ppo.py       # End-to-end BC warm-start → PPO fine-tuning
│   ├── train_ppo.py          # Single-instance PPO training
│   ├── behavior_clone.py     # Heuristic imitation pre-training
│   ├── rollout_worker.py     # Parallel rollout collector (per STS instance)
│   ├── train_offline.py      # Offline PPO trainer for parallel workers
│   ├── ppo_model.py          # PPOTrainer & GameBuffer (shared by all scripts)
│   ├── obs_encoder.py        # Game state → 530-d vector (with monster knowledge base)
│   ├── sts_gym_env.py        # Action space (134 actions), masking, rewards
│   ├── screen_handler.py     # Shared non-combat screen handler (heuristic + RL delegation)
│   ├── game_data.py          # Card, relic, and potion databases for the encoder
│   ├── eval_model.py         # Greedy evaluation harness
│   ├── game_logger.py        # Passive game state recorder
│   ├── plot_training.py      # Training stats visualization
│   └── analyze_trace.py      # Game logger trace analyzer
├── logs/                     # Debug logs, training stats, stuck-state dumps
├── external/
│   └── spirecomm/            # SpireComm library (Communication Mod protocol)
├── requirements.txt
├── LICENSE
└── README.md
```

## How It Works

1. **Observation encoding** (`obs_encoder.py`): Converts the full game state into a 530-float vector covering player stats, hand cards, monster identity/behavior/intents/powers, screen context, relic/potion inventories, deck profile, and map path lookahead. Includes a database of all 66 STS monsters with behavioral flags (enrages on skills, splits, scales strength, multi-hit, retaliates, escapes, spawns minions) and a unique identity embedding per monster. Map encoding uses BFS lookahead to provide elite/rest/combat density for each path choice.

2. **Action space** (`sts_gym_env.py`): 134 discrete actions covering targeted/untargeted card plays (50+10), end turn, targeted/untargeted potions (25+5), choice selection (40), proceed, leave, and no-op. Illegal actions are masked out per game state.

3. **Reward shaping** (`sts_gym_env.py`): Dense per-step rewards for gold, relics, max HP, floor progression, combat damage, card management, and act advancement — plus stronger survival incentives (+60 victory, -25 defeat, and higher HP-loss penalty). Urgent targets such as daggers, Gremlin Wizard, Red/Blue Slaver, Gremlin Nob, Book of Stabbing, Exploder, and minion-spawner bosses receive extra damage/kill rewards to teach healthier target priority.

   **Combat analytics**: Elite and boss fight outcomes are tracked per-game in `training_stats.csv` and per-fight in `fight_stats.csv`, including which monsters were fought, HP before/after, win/loss, and whether the fight ended through a terminal death state. The Control Panel progress panel shows aggregate elite and boss win rates from the per-fight log.

4. **Behavior cloning** (`behavior_clone.py`): A hand-coded heuristic plays full games covering every decision surface. The neural network trains on these demonstrations via cross-entropy loss to get a reasonable starting policy.

5. **PPO fine-tuning** (`train_ppo.py` / `train_bc_ppo.py`): The BC-initialized policy improves through online RL. PPO updates use GAE advantages, clipped surrogate loss, and an entropy bonus that anneals from the selected start value toward 0.01 in BC -> PPO mode.

6. **Parallel scaling** (`rollout_worker.py` + `train_offline.py`): Multiple game instances collect rollouts independently, writing `.npz` files. An offline trainer merges batches and updates the model checkpoint, which workers periodically reload.

## Credits

- [Mod the Spire](https://github.com/kiooeht/ModTheSpire)
- [BaseMod](https://github.com/daviscook477/BaseMod)
- [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod)
- [Super Fast Mode](https://github.com/Skrelpoid/SuperFastMode)
- [SpireComm](https://github.com/ForgottenArbiter/spirecomm)
- Monster data sourced from [Spire Archive](https://www.spire-archive.com)
