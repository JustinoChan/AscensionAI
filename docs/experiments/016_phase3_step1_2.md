# 016 Phase 3, Steps 1–2: Learned Search Shortcuts and Honest Macro Search

**Dates:** 2026-11-06 → 2026-11-07
**Status:** step 1 (value net + prior) is negative and shelved. Step 2a (card-reward search) gives the largest floor gain of the project, but wins did not improve. Step 2b (rest vs smith) is neutral.
**Base:** mainline_h1, which is honest C++ 1-ply search with K=64 (015). Eval-150: 6 wins, floor 24.61, median 136 s per seed. Every comparison below is paired on the same 150 eval seeds against this run.

---

## Headline

| arm (eval-150, all honest) | wins | avg floor | Δ floor vs K=64 (95% CI) | win flips (K64-only / arm-only), McNemar p | median s/seed |
|---|---|---|---|---|---|
| **mainline_h1: 1-ply K=64** | **6** | **24.61** | — | — | 136 |
| **2a. + card-reward search** (`MACRO_CARD`, 32 worlds, 6-floor horizon) | 2 | **28.72** | **+4.11 (+2.46 … +5.79)** | 4 / 0, p = 0.125 | 285 |
| 2b. + rest-vs-smith search (`MACRO_REST`, 16 worlds, K=4 combat in rollouts) | 7 | 25.24 | +0.63 (−0.33 … +1.57) | 1 / 2, p = 1.0 | 277 |
| c1. prior top-3, K=64 | 3 | 23.00 | −1.61 (−2.91 … −0.35) | 5 / 2, p = 0.45 | 75 |
| c2. prior top-3 + value leaf at depth 8, K=64 | 2 | 22.37 | −2.23 (−3.55 … −0.89) | 4 / 0, p = 0.125 | 40 |
| c4. prior top-3 + value leaf at depth 16, K=64 | 2 | 23.37 | −1.24 (−2.42 … −0.10) | 5 / 1, p = 0.22 | 58 |
| prior top-3 + value leaf at depth 8, **K=128** | 0 | 20.56 | −4.05 (−5.42 … −2.69) | 6 / 0, p = 0.031 | 115 |
| holdout-150: K=128 vs K=64 | 5 vs 7 | 26.72 vs 26.67 | +0.05 (−0.74 … +0.94) | 3 / 1, p = 0.63 | 4× K=64 |

No arm had a timeout. A data-integrity audit on 2026-11-07 passed. It covered every seed file, the ledger recount, and 12 byte-identical replays at the current code, two of them full-length.

## Step 1: honest harvest, value net, ranking prior

- **Harvest:** mainline_h1 on 1600 training seeds (10007 + i·7919, i = 150…1749; no overlap with the eval or holdout sets).
  - 23,462 fights, 403,231 root combat decisions with all 64 rollout scores per candidate, and 152,034 non-combat decisions with the run outcome.
  - 30 wins, floor 24.50.
  - The `HARVEST` toggle is passive: decisions were byte-identical with it on and off.
- **Nets** (`train_h1.py`, RTX 3070, varN layout, warm-started from varN, train/validation split by seed):
  - **Prior,** with a listwise ranking loss on the K=64 Q: regret@3 0.70 vs varN's 0.81, recall@3 0.83 vs 0.82. That's a small gain, and it passed the offline gate.
  - **Value V(s):** predicts the greedy-rollout return. MAE 14.8 vs 24.5 for a constant predictor, R² 0.45. Its win head is well calibrated (ECE 0.04). It also passed the offline gate.
- **Online, both lose.**
  - Pruning to the prior's top 3 costs 1.6 floors.
  - Adding the value leaf costs more: 2.2 floors at depth 8 and 1.2 at depth 16.
  - **Doubling K makes the value-leaf arm worse:** K=128 gives −4.05 floors and 0 wins.
  - That is the signature of a *biased* leaf value. More samples only average away noise, so they lock in the bias.
  - The likely cause is distribution shift. V was trained on root states reached by K=64 search play, but it is queried at greedy-rollout states 8–16 steps deep, which the search never visits.
- The c3 "equal wall-clock" arm was stopped and quarantined.
  - Its K was sized from c2's total time, and c2's runs are short because that arm dies earlier. So it chose K=384, which costs about 14× K=64 per seed.
  - The K=128 arm above replaces it.
- **Verdict: shelve the value leaf and prior pruning as a speed-up.** They save 2–6× wall time, but nothing gets that loss back.
  - Possible later fixes:
    - train V on rollout-depth states (sample leaves from rollouts during the harvest), or use TD targets;
    - use V only to break ties, or as a control variate, never as a hard cutoff.

## Step 2a: honest card-reward search (`MACRO_CARD=1`)

- **How it decides:** at each card reward it compares taking card i with skipping.
  - It uses 32 determinized worlds, the same 32 for every candidate.
  - Each world continues with the mainline_h1 macro policy and varN-greedy combat for 6 floors, stopping at the act boss.
  - Score: floor reached + 2 × HP fraction + 100 for a win.
- It averages 12.4 card decisions per run. It skipped 266 of 1853 offers, close to the base heuristic's skip rate.
- **What changed:**

| | K=64 | + card search |
|---|---|---|
| reached Act 2 / Act 3 | 112 / 16 | **141 / 25** |
| deaths at the Act 1 boss | 23 | **6** |
| deaths at the Act 2 boss | 16 | **29** |
| Act 3 deaths | 10 | 23 |
| wins | 6 | 2 |
| deck at Act 2 entry: size / starters / upgraded | 19.6 / 8.0 / 2.30 | 18.8 / 8.3 / 2.15 |
| deck at Act 2 entry: AoE / **scaling** cards | 1.81 / **3.16** | 1.33 / **1.48** |

- **Reading:** the 6-floor horizon rewards cards that pay off in the next few fights. It drafts immediate damage and block, and that gets almost every run through Act 1 (Act 1 boss deaths 23 → 6).
  - It also more than halves the scaling cards drafted (3.16 → 1.48 at Act 2 entry).
  - The cost shows at the Act 2 boss, which is a scaling check (16 → 29 deaths), and wins fall from 6 to 2.
  - This is a myopia artifact of the horizon. Card search itself is not the problem.
- **Cost:** 2.1× K=64 per seed, within the 2× budget.

## Step 2b: honest rest vs smith (`MACRO_REST=1`)

- **How it decides:** rest vs smith at each campfire, over 16 worlds with K=4 combat search inside the rollouts and the same horizon and score.
- It picked rest 370 times and smith 87 times. Rests per run went from 2.39 to 2.58.
- Floor +0.63 (CI −0.33 to +1.57) and 7 vs 6 wins: neutral to slightly positive. The Act 1 boss deaths fall from 23 to 15, and the other buckets are unchanged.
- **Cost:** 2.0× K=64 per seed.
- The `smiths` counter in the run JSON is always 0. That's an existing attribution bug: a smith passes through the CARD_SELECT screen, so it is never counted at REST_ROOM. Upgrade counts are taken from the deck instead.

## Recommendation (next queue, not launched)

1. **Fix the card-search myopia.**
   - (a) Run `MACRO_CARD` in Act 1 only, where it is a clear win, and keep the base drafting from Act 2.
   - (b) Use a horizon that includes the next act boss (MACRO_H up to the boss floor), at fewer worlds (16) to stay near 2× cost.
   - A/B both on eval-150.
2. **Stack** the better of (1a, 1b) with `MACRO_REST`. Both are cheap to combine and touch different decisions.
3. **Holdout gate** for the best stacked arm against holdout K=64 (7 wins / floor 26.67), because the floor gain is large and significant.
4. **Step 1 follow-up,** low priority: retrain V on rollout-depth states before trying it again as a leaf.

## Reproduce

```bash
cd ~/sts
./run_honest.sh ab_mcard_K32 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_CARD=1 MACRO_K=32 MACRO_LOG=1 TRACE=1"
./run_honest.sh ab_mrest_K16 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_REST=1 MACRO_K=16 MACRO_COMBAT_K=4 MACRO_LOG=1 TRACE=1"
python3 ledger.py compare cpp_1ply_K64 ab_mcard_K32
```
