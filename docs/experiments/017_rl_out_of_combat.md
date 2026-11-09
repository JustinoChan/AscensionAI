# 017 RL Out-of-Combat Policy (Phase 3, step 3.2)

**Date:** 2026-11-08 → 2026-11-09
**Status:** **gate passed.** The first learned component to clearly beat the hand-written stack.
**Headline:** eval500, honest K=64 combat: **28/500 wins (5.6%, 95% CI 3.9–8.0%) vs 6/500 (1.2%, CI 0.6–2.6%)** for the heuristic mainline. Paired results: 27 seeds won only with RL and 5 only with the baseline, exact McNemar p < 0.001. Average floor **32.14 vs 24.21 (+7.92, bootstrap CI +6.82 … +9.01)**.

---

## Why the old learned out-of-combat policy never worked

The 1200-d observation (`NNInterface::getObservation`) has HP, gold, floor, act, screen, boss, deck counts, relics and potions. It has **nothing about the options on screen**: not the offered cards, shop items, map nodes or event choices. A policy over that observation (varE, PPO era) picks an option index blind. That explains part of why it plateaued, and why the mainline needed hand-written rules on top.

## The policy

- **Honest candidate scorer** (`MACRO_POLICY`, off by default, byte-identical when off). Every legal option at each out-of-combat decision with ≥ 2 options is described by `[obs(now), obs(after option) − obs(now), screen one-hot]`.
  - The "after" state is computed on a **determinized copy**, so unknown map nodes and random event results are sampled, never read.
  - An MLP 2416-256-256-1 scores each option, with a softmax over the decision.
- **Behaviour cloning:** 1,000 K=8 runs of the heuristic mainline on fresh seeds (800000011 + 19i), about 57k decisions.
  - 80% validation agreement, against 70% for the blind observation.
  - Per screen: card reward 86%, card select 83%, rest 91%, map 74%, event 64%, shop 34%, boss relic 38%.
- **PPO on the run outcome:**
  - Reward = floor/50 + win.
  - Advantage normalised per batch and shared by every decision in a game. Clip 0.2, 4 epochs, entropy 0.01, lr 1e-4.
  - 150 iterations × 100 games on fresh RL seeds (600000007 + 13i), honest **K=8** combat in training, 10 workers, about 6.5 h on the laptop.
  - Validation every 10 iterations on 100 fixed seeds (greedy, K=8).

| checkpoint | val floor | val wins / 100 |
|---|---|---|
| heuristic mainline (reference) | 21.39 | 0 |
| BC start | 19.60 | 1 |
| it 9 / 29 / 49 / 69 | 22.16 / 25.19 / 27.48 / 28.71 | 2 / 1 / 1 / 2 |
| **it 99 (gated)** | **29.65** | **7** |
| it 109 / 119 / 129 / 139 / 149 | 29.71 / 29.30 / 28.39 / 30.48 / 29.29 | 0 / 3 / 2 / 1 / 3 |

Training-batch floor rose steadily from 21.7 to about 29 and entropy fell smoothly from 0.53 to 0.30, with no collapse. Validation levels off at about 29.5 after iteration 90.

## Gate: eval500, honest K=64 combat, paired vs eval500_K64

| | heuristic mainline | **RL policy (it 99)** |
|---|---|---|
| wins | 6 (1.2%) | **28 (5.6%)** |
| avg floor | 24.21 | **32.14** |
| Act 1: died / reached | 116 / 500 (23.2%) | **41 / 500 (8.2%)** |
| Act 2: died / reached | 335 / 384 (87.2%) | **330 / 459 (71.9%)** |
| Act 3: died / reached | 43 / 49 (87.8%) | 101 / 129 (78.3%) |

Loss buckets (baseline → RL):

| bucket | baseline | RL | change |
|---|---|---|---|
| Act 1 boss | 74 | 33 | −41 |
| Act 1 elite | 21 | 6 | −15 |
| Act 1 other | 21 | 2 | −19 |
| Act 2 hallway monsters | 187 | 90 | **−97** |
| Act 2 elite | 95 | 76 | −19 |
| Act 2 boss | 53 | 164 | **+111** |
| Act 3 | 43 | 101 | +58 |

**Reading:**
- RL fixed the early game and the Act 2 hallway attrition that dominated the baseline. Act 2 hallway deaths halved, and 459 of 500 runs now reach Act 2.
- The wall has moved to the **Act 2 boss**: 164 deaths, 35% of all losses, the "scaling" bucket. The 3.6 headroom check also found the Act 2 boss to be the one fight type where better combat information still pays (+8 points).
- Next targets: Act 2 boss preparation (deck scaling, upgrades, HP management in late Act 2) and Act 2 boss combat.

## Caveats

- Training used K=8 combat, but evaluation used K=64. The policy was not trained under the combat it is evaluated with. Training at K=16 is next.
- The gated checkpoint was picked on the 100 validation seeds; eval500 is disjoint from those seeds and from every training seed.

## Reproduce

```bash
cd ~/sts
./rl_pipeline.sh                      # BC data -> BC -> PPO (nets/rl)
./rl_gate.sh                          # best checkpoint + K=64 on eval500, paired vs eval500_K64
```
