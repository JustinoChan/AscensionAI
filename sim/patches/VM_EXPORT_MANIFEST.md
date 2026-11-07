# VM Export — 2026-07-06 (GCP `ascension-vm` evacuation)

Full evacuation of the GCP VM (`ascension-vm`, us-west1-b, Ubuntu 22.04, Python 3.10.12)
ahead of credit exhaustion. Everything needed to modify and redeploy the sim pipeline on
another Linux machine or VM. Checksums: `SHA256SUMS` (verify with `sha256sum -c SHA256SUMS`).

## Archives

| file | size | contents |
|---|---|---|
| `simfork_source.tar.gz` | 7.7M | The sts_gym fork source AS DEPLOYED — tracked files from the working tree, so the two locally patched C++ binding files (`bindings/bindings-util.cpp`, `bindings/slaythespire.cpp`) are included verbatim. Includes vendored `json/` + `pybind11/` submodules. |
| `spike_core.tar.gz` | 59M | The whole project layer that lived at `~/sts_gym_spike/`: every root script (eval/teacher/trainer/diag `.py`, runner `.sh`), MAINLINE.md, all deployed model `.npz` (policy_varN, draft_v/v2/v3 + variants, combat_q_student), all datasets (`draft_data*.jsonl`, banks, mainline snapshot jsonls), and `sim_logs/` complete (6,388 files: ds_train/2/3 per-seed teacher runs, qland Q-landscape shards, vdraft/vdraft3 A/Bs, probe results, oracle sweeps, var-era training logs). |
| `spike_checkpoints.tar.gz` | 1.8G | Historical training checkpoint dirs (`sim_models*`, varB→varN eras) + `teacher_data.npz`. Superseded by the exported root `.npz` files — keep only if you want to resume/inspect old training states; safe to discard. |
| `ascension_live_era.tar.gz` | 70M | The live-game era project (`~/ascension/`) minus `.venv` and the 443M game install: live-PPO training logs (52 files), model checkpoints (7), scripts, seeds, docs, worker configs. |

## Provenance (`simfork/`, `env/`)

- Fork: `https://github.com/Kiarash-m0hammadi/sts_gym.git` @ `3343e85dccb2f2ad08872c4584cee300f8ba024a`
- `simfork/local_changes.patch` — the 80-line diff of the two binding files (also already applied inside `simfork_source.tar.gz`)
- `env/pip_freeze_spike.txt`, `python_version.txt` (3.10.12), `uname.txt`, `cmake_version.txt`

## Redeploy on a fresh Linux machine (or WSL2 on Windows)

The sim is a C++17 pybind11 build — Linux only in practice. On your main Windows machine
use WSL2 (Ubuntu 22.04) or any Linux VM.

```bash
sudo apt install -y build-essential cmake python3.10-venv
mkdir sts && cd sts
tar xzf simfork_source.tar.gz          # fork source WITH the binding patches
tar xzf spike_core.tar.gz              # project scripts/models/data on top (same dir)
python3.10 -m venv .venv
.venv/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install .                # builds slaythespire bindings (~minutes)
# smoke: mainline_v3 on eval seed 1 -> expect '"floor": 28' for seed 10007
.venv/bin/python eval_fullrun_analytics_v2.py 10007 macro | grep -o '"floor": [0-9]*'
```

Exact package versions, if needed: `env/pip_freeze_spike.txt` (numpy 2.2.6, torch 2.12 cpu).

## State at evacuation (2026-07-06)

- Deployed mainline = **mainline_v3** (9 wins / avg floor 26.06 / 150 eval seeds).
  Toggles: `ROLE_SCORE=1 SMITH=1 SMITH_TARGET=1 GATE_ELITES=0 RELIC_BUY=0`,
  combat `policy_varN.npz`. All newer toggles (`DRAFT_SEARCH/DRAFT_LOG/V_DRAFT/QLAND/COMBAT_NET`)
  default OFF and byte-identical when off.
- Draft teacher (Stage A): 44/150. Distilled draft models: `draft_v.npz` (MSE, deployed A/B
  1/150), `draft_v2_*.npz` (rank, role-only, gate FAIL 0.355), `draft_v3*.npz` (rank +
  card-context, gate PASS 0.571, deployed A/B **5/150 vs control 9** — trajectory-level
  covariate shift suspected; next step would be an expert-iteration turn: regenerate teacher
  data under v3-drafted trajectories).
- Combat student `combat_q_student.npz`: offline median teacher-Q regret 0.000 but run-level
  collapse (0 wins / floor 13.46 vs teacher 24.88 on 50 training seeds).
- Datasets: `draft_data_v3.jsonl` (93,528 rows, RAW context per candidate: post-pick roles,
  scalars, card name/type/rarity/qual/frontload — any future feature set can be built offline
  from this without re-running a teacher). `sim_logs/qland/` (747 shards: obs1235/candidates/
  teacher-Q per live combat decision).
- Eval seeds `10007+i*7919, i<150`; training `i=150..1149`; reserved holdout `20011+i*104729,
  i<150` (never used).
- Design/plan/ledger docs live in the main repo: `docs/2026-07-03-purenet-near100-campaign-design.md`,
  `docs/2026-07-04-purenet-phase01-plan.md`, `docs/experiments/013_learned_draft_value.md`.

## Not included (deliberately)

- Both `.venv/` dirs (rebuildable from pip freeze), the STS game install (`ascension/game/`,
  reinstallable), fork `.git` history (remote + commit + patch recorded above).
