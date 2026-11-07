# sim/ — headless-simulator project layer

The code that drives the **headless Slay the Spire simulator** (`sts_lightspeed`, via the
[`sts_gym`](https://github.com/Kiarash-m0hammadi/sts_gym) fork) for full-run Ironclad evaluation,
search, and the honest-search experiments (014+). The live-game control panel and PPO
stack live in the repo root; this folder is the sim era (mainline_v1 → v3 and the honest
rewrite).

What is **not** here: the simulator itself (it is the fork, rebuilt from source + the patches
below), datasets (`*.jsonl`), historical checkpoints, and `sim_logs/`. Those are in the
2026-07-06 VM export archive. The two deployed nets are included:
`policy_varN.npz` (combat, 1235-d) and `policy_varE.npz` (non-combat, 1200-d).

## Build (Linux or WSL2 — no sudo needed)

```bash
# 1. fork source at the pinned commit + this repo's patches
git clone https://github.com/Kiarash-m0hammadi/sts_gym.git sts && cd sts
git checkout 3343e85dccb2f2ad08872c4584cee300f8ba024a && git submodule update --init
git apply ../sim/patches/00_vm_binding_patch_vs_fork_3343e85.patch   # the VM-era binding fixes
git apply ../sim/patches/01_honest_search_vs_deployed.patch           # copy/determinize, C++ rollouts, MCTS
cp -r ../sim/* .                                                      # project scripts + models

# 2. Python 3.10 without root, via uv (the build was verified on Ubuntu 24.04 this way)
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.10.12
$(~/.local/bin/uv python find 3.10.12) -m venv .venv
.venv/bin/pip install numpy==2.2.6 gymnasium==1.3.0 cmake
.venv/bin/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install .            # builds the slaythespire bindings (~2-3 min, needs g++ >= 9)

# 3. smoke: mainline_v3 on eval seed 1 -> floor 28
.venv/bin/python eval_fullrun_analytics_v2.py 10007 macro | grep -o '"floor": [0-9]*'
```

With apt available, `sudo apt install build-essential cmake python3.10-venv` (deadsnakes PPA on
24.04) replaces the uv steps. `patches/VM_EXPORT_MANIFEST.md` and `patches/pip_freeze_vm.txt`
record the original VM environment (Ubuntu 22.04, Python 3.10.12).

## Key entry points

| file | what |
|---|---|
| `eval_fullrun_analytics_v2.py SEED macro` | one full run; prints the run JSON. All behaviour is env toggles (defaults = mainline_v3). |
| `honest.py` | `HONEST=1`: search samples hidden information instead of replaying the true seed. Toggles: `HONEST_K`, `HONEST_CPP`, `HONEST_ENGINE=mcts`, `MCTS_SIMS`, `HONEST_K_ELITE/BOSS`, `HONEST_ENGINE_BY_ACT`, `POTION_SAVE`, `POTION_AUDIT`, `HONEST_MACRO`, `DRAFT_SEARCH` (+`HONEST_DRAFT_K`, `HONEST_TAIL_K`). |
| `run_trace.py` | `TRACE=1`: passive decision-level trace (HP/floor timeline, fights, deck per act, pickups). |
| `run_honest.sh ARM N TIMEOUT "ENV"` | 150-seed (N) parallel runner; writes `sim_logs/honest/ARM/{RUN.json, seed_*.json, run.log, SUMMARY.txt}` and ingests into the ledger. |
| `launch_run.sh PRESET` / `launch_run.sh stop RUN` | detached presets (used by the control panel). |
| `run_status.py [--json]` | progress / ETA for every run (feeds the control panel's Sim Experiments tab). |
| `ledger.py ingest\|list\|compare A B\|taxonomy RUN\|export DIR` | SQLite results ledger; paired comparisons (Wilson CI, exact McNemar, bootstrap floor diff, death buckets). |
| `inspect_seed.py RUN SEED` | replays one seed with `TRACE=1` and writes a self-contained HTML report. |
| `test_honest.py`, `test_cpp_rollout.py`, `bench_engines.py` | determinization parity tests, C++ vs Python rollout agreement, engine cost. |

Eval seeds everywhere: `10007 + i*7919`, i = 0..149.

## Fork patches

* `00_vm_binding_patch_vs_fork_3343e85.patch` — the VM-era fixes to `bindings/` (boss
  `unordered_map::at` and `getCardIdx` OOB guards, observation carry-forward).
* `01_honest_search_vs_deployed.patch` — Phase 1–2 work on top of the deployed tree:
  `GameContext/BattleContext.copy()` and `.determinize(seed)`; `CombatNet`, `rollout_q`/`rollout_qb`
  (native batched, AVX2 rollouts, bit-identical to the Python rollouts); `mcts_root`
  (ensemble of determinized `BattleScumSearcher2` trees); and a fix for five
  `regainControlAction` lambdas in `GameContext.cpp` that implicitly captured `this`.

All toggles are default-off; with every toggle off the outputs are byte-identical to the
original mainline_v3 runs (checked on 6 eval seeds after each change).
