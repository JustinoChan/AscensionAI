# 015 Native C++ Search — more honest samples per decision

**Dates:** 2026-11-06 → (in progress)
**Status:** Phase 2 in progress — this report is a working draft; the gate run and final write-up follow.
**Best so far:** **honest 1-ply search with K=32 sampled worlds: 5 wins / 150, avg floor 24.40** (honest K=8 baseline: 1 / 22.41).

---

## What changed

All search is honest (014): it samples hidden information, never reads the seed.

| piece | what | check |
|---|---|---|
| `CombatNet` + `rollout_qb` | the varN combat MLP and the 35-d combat-feature block in C++; every (candidate × sample) rollout of a decision advances in lockstep, with AVX2/FMA dot products under a function-level target attribute (the simulator itself compiles exactly as before) | observations 60/60 byte-identical, rollout scores 1848/1848 identical, chosen action 60/60; **~37× faster per rollout** than Python |
| `HONEST_CPP=1` | the honest 1-ply combat search on the native rollouts | identical decisions to the Python search on **150/150** seeds; 150-seed wall time **35m20s → 3m07s** |
| `mcts_root` / `HONEST_ENGINE=mcts` | ensemble of the sim's `BattleScumSearcher2` MCTS trees, one per determinized root; root stats summed per action (root actions are visible, hence identical across samples) | plays potions and card-select actions, values potions in its end-state score |
| `HONEST_K_ELITE/BOSS`, `MCTS_SIMS_*` | sample budget by room type | |
| `POTION_SAVE=1` | no potions in hallway fights unless HP ≤ 25% or visible incoming damage is lethal | |
| `HONEST_ENGINE_BY_ACT` | per-act engine (e.g. 1-ply in Act 1, MCTS from Act 2) | |

## Potion audit (honest K=8, passive; decisions identical on 150/150 seeds)

| room | fights | potions held at start | used / fight | deaths | deaths holding a potion |
|---|---|---|---|---|---|
| hallway | 1619 | 0.47 | 0.44 | 58 | 2 |
| elite | 233 | 0.56 | 0.55 | 37 | 0 |
| boss | 157 | 0.63 | 0.63 | 48 | 0 |

The agent never dies holding potions — it spends them as soon as it gets them, ~75% of them in hallway fights.

## Results so far (150 fixed seeds, all honest)

| arm | wins | avg floor | Act 2 / 3 | A1 boss deaths | A2 boss deaths | A2 hallway deaths | median s/seed |
|---|---|---|---|---|---|---|---|
| 1-ply K=8 (baseline) | 1 | 22.41 | 102 / 6 | 30 | 18 | 55 | 12 |
| MCTS 8 trees × 700 (equal compute) | 2 | 21.25 | 88 / 10 | 39 | 7 | 44 | 12 |
| MCTS 16 × 1000 | 3 | 20.87 | 81 / 10 | 41 | 9 | 39 | 33 |
| **1-ply K=32** | **5** | **24.40** | 115 / 16 | 21 | 14 | 52 | 56 |
| adaptive K 8 / 32 / 64 (hallway/elite/boss) | 4 | 24.18 | 116 / 17 | 21 | 8 | 69 | 49 |
| adaptive + potion-save | 2 | 23.95 | 121 / 14 | 14 | 11 | 70 | 54 |
| MCTS 16 × 1000 + potion-save | 2 | 22.43 | 100 / 12 | 30 | 11 | 51 | 39 |
| hybrid: 1-ply adaptive in Act 1, MCTS from Act 2 | 5 | 24.26 | — | — | — | — | — |
| hybrid + potion-save | 2 | 24.43 | — | — | — | — | — |

Paired on the same seeds (ledger `compare`), K=32 vs K=8: **+1.99 floors (bootstrap 95% CI +0.84 … +3.12)**; wins 1 → 5 (exact McNemar p = 0.125: 4 flips, all one way, but too few to be significant).

Running: K=64 everywhere, adaptive 32 / 64 / 128. The gate (best engine vs the honest K=8 baseline, 150 seeds, with throughput) and the final analysis will be added here.
