# 009 Headless-Simulator Era — Variant Series (A–M) and First Wins

**Dates:** 2026-06-10 → 2026-06-14
**Environment:** `sts_lightspeed` headless C++ simulator via the `sts_gym` Gymnasium fork (Ironclad, full-run, 1200-d observation, 128 masked actions)
**Method:** process-separated PPO (torch-free env workers + torch trainer over IPC); staged experiments that change **one lever at a time** for clean attribution; fixed-seed greedy evaluation throughout.

---

## Summary

After the migration to a headless simulator (≈11,000× the live game's throughput), the experiment loop dropped from weeks to under an hour, enabling a rapid sequence of controlled variants. This report documents every variant in that series, the reasoning behind each change, and the measured result.

The arc reached two decisive conclusions:

1. **Combat execution — not reward, capacity, observation, or deck-building — was the wall.** A 1-ply lookahead search at combat decision time nearly **doubled run depth (avg floor 14.7 → 26.4)** and reached Act 3 for the first time, with everything else held fixed.
2. **A simple deck-quality fix on top of search-combat produced the project's first wins (0 → 3 of 48 seeds)** — the first completed runs across ~21,700 live games and the entire simulator series.

A standalone (search-free) combat *policy* could not be trained to search level by any tested method (model-free RL, distillation, on-policy DAgger, value-bootstrap, or added network capacity); good combat currently requires search at decision time.

---

## Variant registry

| Variant | Date | One change vs prior | Reasoning | Result | Verdict |
|---|---|---|---|---|---|
| **Baseline** | Jun 11 | From-scratch PPO on the sim | Establish the sim plateau | Eval floor plateaus ~14.4; entropy collapsed to 0 at ~3.5k games | Plateau is a floor, not a ceiling |
| **A** | Jun 11 | Entropy auto-tune (hold normalized entropy 0.15–0.30) | Test if entropy collapse caused the plateau | Eval floor statistically identical to baseline | Exploration is **not** the constraint |
| **B** | Jun 11 | 3.5× network capacity (512,256,256 GELU, ~836K) | Test if capacity caused the plateau | 150k games: floor 12.9–14.9, no trend; Act 2 23–29%; Act 3 0%; 0 wins | Capacity is **not** the constraint |
| *(obs bug)* | Jun 12 | — | — | **Found:** calling `getNNInterface()` at worker module load corrupts the card-encode map → deck region + in-combat hand/draw/discard (~880 of 1200 dims) collapse to a fallback slot. Baseline/A/B trained **deck- and hand-blind.** | Root-cause of the immovable plateau in those runs |
| **C** | Jun 12 | Fixed obs + deck-quality (potential-based) reward | First sighted run | 50k: floor 14.81, Act 2 29% (gate ≥17/≥40% failed). Eval deck 20.3 cards — the agent now **drafts** | Sighted, but floors flat |
| **D** | Jun 12 | Fixed obs + deck shaping (150k) | Confirm C at full length | Floor 14.67, Act 2 25% | Plateau persists |
| **E** | Jun 12 | Fixed obs + **bare** reward (control, 150k) | Isolate deck shaping vs the obs fix | Floor 14.71, Act 2 27% — **drafts identically to D** | **Plateau is reward-independent; deck shaping adds nothing.** The drafting gain was the obs fix |
| **F** | Jun 12 | Fight-curriculum gym (isolated fights, dense combat reward) | Train combat directly instead of via sparse full-run signal | Fight win rates **froze** at 72/94/56/44 (all/monster/elite/boss) for 290k games; explained variance ≈ 0 | Combat win rates do not move; value head is a near-constant predictor |
| *(G)* | Jun 12 | Combat-obs overhaul (decode intent → incoming damage, 1235-d, separate value trunk, γ0.97) | Hypothesis: weak obs → dead value head | **Premise falsified before launch:** fight returns are 98% learnable offline from the *plain* 1200-d obs; the obs was never the combat bottleneck. **Not launched.** | Obs is not the limiter |
| **H** | Jun 12 | Pure-fight gym + value-target normalization | Remove run-worker dilution; rescale value targets | ev still ~0; fight WR frozen 72/94/56/44 | Value head still won't learn live |
| **I** | Jun 12 | Pure-fight + dense combat reward (damage-dealt + HP) | Denser per-step credit | Frozen 72/94/56/44; train win rate oscillates 76–81% no trend | Model-free combat does not improve |
| **J** | Jun 12 | **Value-replay buffer** (decoupled value head trained on clean Monte-Carlo targets) | Directly fix value learning | **ev 0 → ~0.3 (value head learns for the first time)** but greedy fight WR still frozen at 44% boss | Value baseline was **not** the binding constraint |
| **K** | Jun 13 | **Train-time 1-ply search** (replay-rollout) + distillation | Generate strong targets via lookahead; distill into a pure policy | **Search wins boss 75–88% vs greedy 44%**; ev → 0.97. But the distilled greedy policy stayed 19–38% boss | Search works; distillation does **not** transfer it |
| **L** | Jun 13 | On-policy distillation (**DAgger**, β-annealed) | Fix the covariate shift in K | Greedy boss bounced 19–44%, ended 19% | DAgger also caps at the baseline |
| **M** | Jun 14 | Value-bootstrapped teacher (replace rollout with value head) | Sharper, faster teacher using J/K's value head | **Collapsed** — search boss 4%, every eval boss died (teacher rated all actions ≈ loss) | The rollout *is* the discovery engine; a value bootstrap on a losing policy spirals |

---

## Phase 1 — The plateau is a floor, not a ceiling (Baseline, A, B)

Three controlled runs pinned the plateau at eval floor ~14.4–14.9, matching the live-game project's ~14.7 on a different observation/action space:

- **Baseline** plateaued with entropy collapsed to 0 by ~3.5k games (fixed entropy coefficient swamped by the env's large reward scale).
- **Variant A** held normalized entropy healthy in 0.15–0.30 via an auto-tuner; the eval curve was statistically identical.
- **Variant B** tripled network capacity; 150k games oscillated 12.9–14.9 with no trend, Act 3 reach 0%.

**Conclusion:** the plateau is not optimization hygiene, exploration, or capacity. Then the cause of *those* runs' immovability surfaced: a fork-binding call order silently zeroed ~73% of the observation (deck + in-combat hand/draw/discard). Baseline/A/B had been training effectively blind.

## Phase 2 — Reward and deck shaping are exhausted (C, D, E)

With the observation fixed, the agent visibly began drafting (eval deck size ~20 cards vs the blind era). But:

- **C** (50k) and **D** (150k) with potential-based deck shaping held floor ~14.7.
- **E** (150k), the bare-reward control, drafted *identically* to D and held the same floor.

**Conclusion:** five controlled runs (blind: baseline/A/B; sighted: C/D/E, shaped and bare) all pin at eval floor ~14.4–15.0. The plateau is **reward-independent**; potential-based deck shaping is policy-invariant by construction and adds nothing on top of the obs fix.

## Phase 3 — Combat won't improve, and the value head is the symptom (F, H, I, J)

The fight-curriculum gym trained combat directly on isolated, deterministic fights. Across F (fightmix), H (fightpure + return normalization), and I (fightpure + dense combat reward), the **greedy fight win rates were bit-identically frozen at 72/94/56/44** — equal to a near-untrained network's argmax. Explained variance sat at ≈ 0.

A decisive control settled what this meant: on the same 64 eval fights, **greedy wins boss 44% but a random policy wins 0%** — so combat play matters and there is large headroom; the agent simply converges to its initial argmax behavior and never climbs.

**Variant J** (value-replay buffer) fixed value *learning* for the first time (ev 0 → ~0.3) by training a decoupled value head on clean Monte-Carlo targets — yet the greedy policy still froze at 44% boss. **The value baseline was not the binding constraint; model-free PPO cannot escape the combat local optimum here.** This matches the literature: every Spire-beating agent uses search/planning, because model-free value learning on this sparse, high-variance, long-horizon signal is hard.

## Phase 4 — Train-time search: the breakthrough (K + full-run search eval)

A torch-free search worker reconstructs any fight state by deterministic replay (the fork has no state cloning, but fights are deterministic given seed + action prefix) and runs a 1-ply lookahead: for each candidate action, roll out to the end of the fight and score it.

- On the 64 eval fights, **search wins boss 75–88% / elite ~94%** vs the stuck greedy 44%/56%.
- **Variant K** distilled those search choices into the policy; the value head reached ev 0.97. But the *greedy* (no-search) policy stayed at 19–38% boss — distillation did not transfer.

The decisive measurement came from a **full-run evaluation that held the deck/map/shop policy fixed and replaced only combat decisions with search:**

| Metric | Baseline policy | Search combat |
|---|---|---|
| Avg floor | 14.71 | **26.35** |
| Act-2 reach | 27% | **83%** |
| Act-3 reach | 0% | **14.6%** |
| Max floor | 33 | **50** |
| Win rate | 0% | 0% |

Only combat changed, so the entire **+11.6-floor** gain is attributable to combat quality. **Combat execution was the wall the whole time.**

## Phase 5 — A standalone combat policy hits a representational limit (L, M, capacity probe)

If search plays combat well, can a search-free *policy* be trained to match it?

- **Variant L** (DAgger, on-policy distillation) addressed the covariate shift in K but still capped (greedy boss 19–44%, ended 19%).
- **Variant M** (value-bootstrapped teacher) collapsed: with the rollout removed, the value head — trained on a policy that loses bosses — rated every boss state as a near-certain loss, so the teacher discovered no winning line and spiraled (search boss 4%).
- An **offline capacity probe** trained policies of 0.86M → 5.22M parameters to imitate the search teacher: per-state agreement plateaued at ~70% on boss states (0.690 → 0.727 for 6× capacity). **Capacity is not the limiter.**

**Conclusion:** a feedforward policy cannot absorb the search teacher to search level on long fights — not for lack of reward, observation, value baseline, training distribution, teacher sharpness, or capacity. Good combat currently requires search at decision time, which re-corrects the policy's small per-step errors every step (the AlphaZero pattern: deploy with search, not the bare network).

## Phase 6 — Deck quality on top of search-combat → first wins

With combat handled by search, the *next* bottleneck became measurable for the first time. Failure analytics over 48 full runs showed **69% of deaths in Act 2** (mostly ordinary monsters and elites, dying with the enemy still at ~42% HP — outpaced, not unlucky), with a bloated, weak deck: **29 cards, ~10 un-removed basics, 0 relics bought, ~3 upgrades, 0.8 removes per run.**

A narrow heuristic override — skip filler card rewards, suppress shop card-buys, remove Strikes/Defends — produced the **first wins in the project's history:**

| Metric | Search baseline | + deck override |
|---|---|---|
| **Wins (of 48)** | **0** | **3** |
| Deck size | 29.1 | 21.4 |
| Strikes in deck | 6.0 | 3.2 |
| Shop cards / run | 3.65 | 0.00 |
| Powers (scaling) | 2.2 | 3.9 |
| Elite deaths | 14 | 7 |

The deck-builder was the next real lever (it had been hidden behind weak combat). The override traded some average floor (26.4 → 23.3) for the top-end wins — a higher-variance "snowball or fizzle" profile that the next iterations target (early card-quality / upgrade decisions). Diagnostics on the remaining Act-1-boss deaths point at **upgrade starvation** (those decks carry ~0.3 upgraded cards), the current active lever.

---

## Conclusions

- The universal ~14.7 plateau was **combat execution.** It survived a dozen reward/observation/value/capacity/curriculum experiments because none of them touched the binding constraint; it fell the moment fights were played well.
- **Search is the combat solution.** 1-ply lookahead nearly doubles run depth and reaches Act 3. A standalone policy cannot be trained to match it by imitation or capacity — search functions as a necessary per-step error-correction mechanism on long fights.
- **The first wins are real**, achieved by combining search-combat with basic deck hygiene. The project has moved from "can this work?" to "how do we make it reliable?" — the remaining levers (deck quality, upgrades, Act 2/3 survival) are now narrow and measurable rather than existential.

## Next steps

- Tune early-game card quality / upgrade decisions to recover average floor while keeping wins ≥ 3.
- Layer in relic purchasing and event removal once the upgrade lever is verified.
- Optional research branch: a card-attention / action-scoring policy trained on regret-weighted search targets — the one architecture class not yet tested for a search-free combat policy.
