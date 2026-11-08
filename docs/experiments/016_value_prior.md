# 016 Honest Data, Value Net and Candidate Prior (Phase 3, step 1)

**Date:** 2026-11-06 → (in progress)
**Status:** step 1a (harvest) is running. This is a working draft.
**Base engine:** mainline_h1, which is honest C++ 1-ply search with K=64 (see 015). Holdout-150: 7 wins, floor 26.67. Eval-150: 6 wins, floor 24.61. Median 136 s per seed.

Goal: make honest samples cheap. A learned **prior** prunes the root candidates, and a learned **value net** V cuts each rollout at depth d. That spends the same compute on more samples, or the same samples at a fraction of the cost.

---

## 1a. Honest harvest

- **Engine and seeds:** mainline_h1 on **training seeds** 10007 + i·7919 for i = 150 to 1749. That's 1600 seeds, none of them in the eval-150 or holdout sets.
- **Toggle:** `HARVEST=<dir>`, off by default. It is passive: with it on, test seeds play byte-identically to it being off.

| file `seed_<s>.npz` | contents |
|---|---|
| `c_*` (every root combat decision) | `obs1235`, the searched candidates (≤ 6, varN top-6), **all K=64 rollout scores per candidate**, chosen action, HP, block, fight index |
| `f_*` (every fight) | room, floor, act, HP in and out (the out value is read at the next non-combat decision, so it includes post-combat healing), max HP, potions held and used, won |
| `m_*` (every non-combat decision) | run-level `obs1200`, action mask, screen state, action, floor, act, HP, max HP, gold, deck size |
| `run` | seed, final floor, win, act |

About 45 KB per seed and about 13 fights per seed. Throughput is shown in the Sim Experiments tab, and `harvest_status.py` gives running counts.

## 1b. Nets (`train_h1.py`, RTX 3070, separate CUDA venv)

Both nets use the varN MLP layout (1235-512-256-256-out, GELU-tanh) and are warm-started from varN. That means the deployed numpy (`NpNet`) and C++ (`CombatNet`) code can run them unchanged. Train and validation sets are split by seed.

- **Prior:** a listwise ranking loss within each decision, KL(softmax(Q̄/τ) ‖ softmax(logits)) over the searched candidates, with τ = 2. Q̄ is the mean of the 64 samples.
- **Value:** output[0] is the greedy-rollout return from the state (target: Q̄ of the varN-greedy candidate, which is exactly what a K=64 default-policy rollout scores), divided by 50. Output[1] is the HP lost for the rest of the fight as a fraction of max HP. Output[2] is the fight-win logit.

The offline gate compares against held-out fights:
- prior: ranking agreement with the K=64 Q (top-1, regret@m, recall@m), against varN;
- value: R² and MAE of V, and calibration of the win head.

(results pending)

## 1c. Prior pruning + value-leaf rollouts

| toggle (default off; byte-identical when off) | effect |
|---|---|
| `HONEST_PRIOR=<npz>`, `HONEST_TOPK=<m>` | rank root candidates with the prior and keep the top m |
| `HONEST_VNET=<npz>`, `HONEST_DEPTH=<d>` | each rollout runs d default-policy steps and is then scored G += γ^steps · 50 · V(s_leaf) (native batched, `rollout_qb(..., vnet, depth, vscale)`) |

The C++ value head matches numpy to 1e-7. After the rebuild, the rollout parity tests (1848/1848 scores, 60/60 actions) and full seeds are unchanged.

**Gate (A/B on eval-150 against K=64):** same or better floor at ≤ 1/3 of the cost, or better floor at the same cost.

(results pending)
