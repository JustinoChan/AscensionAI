# AscensionAI Architecture

AscensionAI is a local distributed reinforcement-learning system. It does not expose the live game as a hosted service; it coordinates several local Slay the Spire processes, a Communication Mod bridge, rollout collectors, and an offline PPO trainer through files and checkpoints.

![AscensionAI architecture](assets/architecture.svg)

## Topology

```mermaid
flowchart LR
    subgraph Games["Live game processes"]
        G1["STS instance 1"]
        G2["STS instance 2"]
        G3["STS instance N"]
    end

    subgraph Bridge["Communication layer"]
        C1["Communication Mod + SpireComm"]
        C2["stdin/stdout game-state protocol"]
    end

    subgraph Workers["Rollout collectors"]
        W1["rollout_worker.py --id 1"]
        W2["rollout_worker.py --id 2"]
        W3["rollout_worker.py --id N"]
    end

    R["rollouts_shared/*.npz\ncheckpoint-tagged game files"]
    T["train_offline.py\nPPO batch updates"]
    M["models/ppo_sts.pt\natomic checkpoint"]
    E["eval_model.py\nfixed-seed greedy eval"]
    GUI["AscensionAI.pyw\ncontrol panel + process supervisor"]
    L["logs/*.csv\ntraining, BC, eval, fight stats"]

    G1 --> C1 --> W1
    G2 --> C1 --> W2
    G3 --> C1 --> W3
    C1 --> C2
    W1 --> R
    W2 --> R
    W3 --> R
    R --> T --> M
    M --> W1
    M --> W2
    M --> W3
    M --> E --> L
    T --> L
    W1 --> L
    W2 --> L
    W3 --> L
    GUI --> G1
    GUI --> G2
    GUI --> G3
    GUI --> T
```

## Systems-Engineering Details

| Component | Responsibility |
|---|---|
| Slay the Spire instances | Real game simulation. Each process runs the same mod stack and exposes game state through Communication Mod. |
| Communication Mod bridge | Converts the live game loop into a stdin/stdout protocol that Python can read and command. |
| Rollout workers | Load the current policy, play complete games, write `.npz` rollouts, and periodically reload checkpoints. |
| Shared rollout directory | Local file queue. Each file is a self-contained game trajectory with worker and checkpoint metadata. |
| Offline trainer | Batches fresh rollout files, rejects stale or legacy data, applies PPO updates, and atomically saves the shared checkpoint. |
| Checkpoint sync | Workers reload `models/ppo_sts.pt` on an interval so rollout data stays close to the trainer's current policy. |
| Evaluation harness | Runs heuristic or model policies against deterministic seed sets and writes comparable CSV metrics. |
| Control panel | Launches modes, starts/stops game processes, tails logs, recommends worker counts, and cleans up orphaned processes. |

## Failure Handling

| Failure mode | Handling |
|---|---|
| Game process exits | The launcher tracks process IDs and can relaunch workers through ModTheSpire. |
| Detached JVM remains alive | Stop actions sweep for orphaned Slay the Spire Java processes. |
| Worker lags trainer | Rollout metadata records checkpoint IDs; the trainer rejects rollouts beyond `--max-rollout-lag`. |
| Legacy rollout lacks metadata | Rejected by default unless `--allow-legacy-rollouts` is explicitly set. |
| Invalid or stuck screen state | Shared screen handler uses conservative proceed/choice recovery and logs command errors. |
| Interrupted BC collection | Per-game BC progress checkpoints allow restart without losing completed demonstrations. |

## Scaling Points

The current implementation uses a local filesystem queue because the bottleneck is live game simulation, not network transport. The architecture maps cleanly to a larger deployment:

| Local component | Cloud/distributed analogue |
|---|---|
| `rollouts_shared/*.npz` | Object storage bucket or durable queue |
| `models/ppo_sts.pt` | Versioned model registry artifact |
| Worker IDs | Container/task IDs |
| Stale rollout checks | Policy-version validation at ingest |
| Control panel | Job launcher and log viewer |
| CSV logs | Metrics sink or experiment tracker |

The public dashboard and experiment reports are intentionally deployable without the game dependency. A reviewer can inspect the architecture, results, and training loop without installing Slay the Spire.
