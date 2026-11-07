# 015 Native C++ Search: more honest samples per decision

**Date:** 2026-11-06
**Status:** Phase 2 is done and the gate passed on the 150-seed holdout.
**Best honest engine:** 1-ply search with **K=64** sampled worlds.
- Holdout-150: **7 wins, average floor 26.67**. The honest K=8 baseline on the same seeds gets 3 wins and floor 23.10.
- Fixed eval-150: 6 wins, floor 24.61.

---

## What changed

All search here is honest (see 014): it samples hidden information and never reads the seed.

| piece | what | check |
|---|---|---|
| `CombatNet` + `rollout_qb` | The varN combat MLP and the 35-d combat-feature block, ported to C++. All rollouts for one decision (candidates × samples) advance in lockstep. Dot products use AVX2/FMA through a function-level target attribute, so the simulator itself compiles exactly as before. | Observations identical on 60/60, rollout scores identical on 1848/1848, chosen action identical on 60/60. **About 37× faster per rollout** than Python. |
| `HONEST_CPP=1` | The honest 1-ply combat search, running on the native rollouts | Same decisions as the Python search on **150/150** seeds. Wall time for 150 seeds: **35m20s → 3m07s**. |
| `mcts_root` / `HONEST_ENGINE=mcts` | An ensemble of the sim's `BattleScumSearcher2` MCTS trees, one per determinized root. Root stats are summed per action; root actions are visible, so they are identical across samples. | Plays potions and card-select actions, and counts potions in its end-state score |
| `HONEST_K_ELITE/BOSS`, `MCTS_SIMS_*` | Sample budget set per room type | |
| `POTION_SAVE=1` | Blocks potions in hallway fights unless HP ≤ 25% or the visible incoming damage is lethal | |
| `HONEST_ENGINE_BY_ACT` | Picks the engine per act (e.g. 1-ply in Act 1, MCTS from Act 2) | |

Every toggle is off by default. With all toggles off the code is byte-identical to before.

## Potion audit (honest K=8, passive; decisions identical on 150/150 seeds)

| room | fights | potions held at start | used / fight | deaths | deaths holding a potion |
|---|---|---|---|---|---|
| hallway | 1619 | 0.47 | 0.44 | 58 | 2 |
| elite | 233 | 0.56 | 0.55 | 37 | 0 |
| boss | 157 | 0.63 | 0.63 | 48 | 0 |

The agent almost never dies holding a potion. It spends each one as soon as it gets it, and about 75% of them go in hallway fights. A blunt fix (`POTION_SAVE`) did not help, though. It moved deaths from the Act 1 boss to Act 2 hallways and cost wins in every arm it was added to (see below). Whether to use a potion has to be priced by a run-level value (Phase 3); a fixed rule can't make that call.

## Results on the fixed eval-150 (10007 + i·7919), all honest

| arm | wins | avg floor | Act 2 / 3 | median s/seed |
|---|---|---|---|---|
| 1-ply K=8 (baseline) | 1 | 22.41 | 102 / 6 | 12 |
| MCTS 8 trees × 700 sims (equal compute) | 2 | 21.25 | 88 / 10 | 12 |
| MCTS 16 × 1000 | 3 | 20.87 | 81 / 10 | 33 |
| MCTS 16 × 1000 + potion-save | 2 | 22.43 | 100 / 12 | 39 |
| 1-ply K=32 | 5 | 24.40 | 115 / 16 | 56 |
| **1-ply K=64** | **6** | **24.61** | 112 / 16 | 136 |
| adaptive K 8 / 32 / 64 (hallway/elite/boss) | 4 | 24.18 | 116 / 17 | 49 |
| adaptive K 32 / 64 / 128 | 6 | 24.56 | 115 / 18 | 192 |
| adaptive 8/32/64 + potion-save | 2 | 23.95 | 121 / 14 | 54 |
| hybrid: adaptive 1-ply in Act 1, MCTS from Act 2 | 5 | 24.26 | — | 51 |
| hybrid + potion-save | 2 | 24.43 | — | — |

Paired comparisons on the same seeds (ledger `compare`, bootstrap 95% CI on the floor difference, exact McNemar on wins):

| comparison | Δ floor | 95% CI | wins | McNemar p |
|---|---|---|---|---|
| K=32 vs K=8 | +1.99 | +0.84 … +3.12 | 1 → 5 | 0.125 |
| K=64 vs K=8 | +2.20 | +0.90 … +3.47 | 1 → 6 | 0.062 |
| K=64 vs K=32 | +0.21 | n.s. | 5 → 6 | — |
| adaptive 32/64/128 vs K=32 | +0.16 | n.s. | 5 → 6 | — |

Takeaways from the eval set:
1. **More samples is the lever.** Floor rises with K from 1 to 32. The gain from 32 to 64 looked flat on this seed set, but the holdout overturned that (below).
2. **Determinized MCTS loses to 1-ply here.** It wins about as many games but dies far more often at the Act 1 boss (39–41 vs 21). With only 8–16 determinizations, each tree overfits its own sampled world (strategy fusion). It also uses the sim's built-in end-state value instead of the varN policy rollouts.
3. **Spreading the budget by room type doesn't help.** Adaptive K roughly matches uniform K at about the same average cost.
4. **Potion-save hurts** in every arm (see the audit above).

## Gate: holdout-150 (20011 + i·104729), never used for selection

| arm | wins (95% CI) | avg floor | reached Act 2 / Act 3 | median s/seed | wall (14 workers) |
|---|---|---|---|---|---|
| 1-ply K=8 (baseline) | 3 (0.7–5.7%) | 23.10 | 107 / 10 | 13 | 3 min |
| 1-ply K=32 | 2 (0.4–4.7%) | 24.83 | 122 / 16 | 63 | 14 min |
| **1-ply K=64** | **7 (2.3–9.3%)** | **26.67** | **128 / 22** | 154 | 45 min |

| comparison (paired, same holdout seeds) | Δ floor | bootstrap 95% CI | win flips | McNemar p |
|---|---|---|---|---|
| K=32 vs K=8 | **+1.73** | +0.79 … +2.67 | 1 lost, 0 gained | 1.000 |
| **K=64 vs K=8** | **+3.57** | **+2.45 … +4.75** | 0 lost, **4 gained** | 0.125 |
| K=64 vs K=32 | **+1.84** | +0.85 … +2.86 | 0 lost, 5 gained | 0.062 |

Where runs ended (holdout):

| outcome | K=8 | K=32 | K=64 |
|---|---|---|---|
| WIN | 3 | 2 | 7 |
| Act 1 boss | 27 | 17 | **12** |
| Act 1 elite | 11 | 9 | 5 |
| Act 1 other | 5 | 2 | 5 |
| Act 2 monster | 51 | 54 | 52 |
| Act 2 elite | 28 | 37 | 30 |
| Act 2 boss | 18 | 15 | 24 |
| Act 3 | 7 | 14 | 15 |

**Verdict: the gate passes, and the chosen engine is K=64.**
- On the holdout the floor gain over K=8 has a confidence interval well clear of zero (+3.57, CI +2.45 to +4.75).
- Every win flip goes one way (4–0). With so few wins, though, the win difference alone is not significant (p = 0.125).
- K=64 also clearly beats K=32 on the holdout (+1.84, CI +0.85 to +2.86). The eval set had shown no difference between them, so the eval-set plateau was partly noise. Holdout floor 26.67 is above the *clairvoyant* mainline_v3's 26.06, although that number is from the eval set.
- K=32 alone gains floors but not wins, so wins at 150 seeds are still too noisy to rank arms on. The gate uses floors.

**Throughput:** K=64 costs about 12× K=8 per seed (median 154 s against 13 s), or about 45 minutes per 150 seeds on the laptop's 14 workers. That is fine for gating. For Phase 3 data generation it means samples must get cheaper (a value net in place of full rollouts, priors to prune candidates) before K can grow further.

## What the losses look like (eval K=64)

- **Act 2 elites (34) and Act 2 hallway fights (46)** are the biggest buckets.
  - 28 of the hallway deaths are attrition: the run was outpaced over several fights.
  - 18 are fights that were winnable but thrown.
- **Act 1 boss (23)** is next.
- At Act 2 entry, the decks that later died in Act 2 hallways had a median of 20 cards, 9 of them starters, and only 2 upgraded.

This is a macro problem (deck quality, HP management), not a combat-search problem. It is the Phase 3 target: honest search for non-combat decisions and a run-level value net.

## Reproduce

```bash
cd ~/sts
./run_honest.sh cpp_1ply_K64 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64"
./run_gate2.sh            # holdout K8 / K32 / K64 + ledger compares -> sim_logs/honest/GATE_phase2.txt
python3 ledger.py compare holdout_K8 holdout_K64
```

All runs are in the results ledger (`sim_logs/results.db`) and exported to `docs/experiments/sim_runs.json`.
