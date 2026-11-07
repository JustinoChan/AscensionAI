# 014 Honest Search — what mainline_v3 is worth without clairvoyance

**Dates:** 2026-11-06
**Machine:** laptop, WSL2 Ubuntu 24.04 (Ryzen 9 5900HS, 16 threads; 14 seeds in parallel)
**Result:** **Making the search honest costs ~4–9 floors and nearly all wins.** With one sampled world per candidate (K=1), mainline_v3 drops from **26.06 → 17.05** avg floor and **9 → 0** wins / 150. Averaging over more sampled worlds recovers part of it (K=4: 21.13, K=8: 22.41, 1 win each). The **honest draft teacher** (K=4 combat + K=4 draft rollouts) adds +2.6 floors over honest combat alone on the same 50 seeds (19.80 → 22.42) but wins **0/50**, against the clairvoyant teacher's **15/50 (37.5 floor)** on those seeds — the "44/150 teacher" was almost entirely clairvoyance.

---

## Why

Every search in the project rebuilt its rollouts by **replaying the true seed** (`eval_fullrun_search._rollout_q` / `_restore`, `eval_fullrun_analytics_v2._rollout_run`). A rollout started from the true seed knows things a player can't: the real draw order, the enemies' upcoming move rolls, every future card reward, which encounter each `?`/monster node holds, and the act's elite/boss queue. So mainline_v3's 9/150 and the draft teacher's 44/150 measure search *with clairvoyance*, not play.

Decision (Justin): **search may SAMPLE hidden information, never read it.**

## Method

**1. Exact copies + determinization in C++** (`bindings/slaythespire.cpp`).

| binding | what it does |
|---|---|
| `GameContext.copy()` | independent deep copy, **including the map**. The map is held by `shared_ptr` and is overwritten in place on act transitions, so a shallow copy would have let rollouts rewrite the live game's map. |
| `BattleContext.copy()` | exact copy (plain value type; all queued action lambdas take `bc&`, no captured `this`) |
| `GameContext.determinize(s)` | replaces the hidden **seed** (per-floor misc RNG, every battle's ai/shuffle/hp/cardRandom RNGs, and future acts' maps and bosses all derive from `gc.seed`); reseeds all **14** `Random` members (counters preserved); resamples the **unseen remainder** of the monster and elite queues with the game's own generators and no-repeat constraints (already-fought prefix kept); reshuffles the 5 relic pools. Visible state (deck, relics, HP, gold, potions, current map and boss, the open screen) is untouched. |
| `BattleContext.determinize(s)` | reseeds the 6 battle RNGs and Fisher–Yates shuffles the draw pile. Hand, intents, HP, block and statuses are untouched. |

A latent engine bug surfaced on the way: five post-combat event callbacks (`regainControlAction` lambdas in `GameContext.cpp`) used `[=]`, which implicitly captures `this`. Invoked on a copy they would call `createCardReward` / `addPotionRewards` / `miscRng` / `act` on the **original** game. They now go through the `gc` argument — identical behaviour when `gc == *this`, which is always the case without copies.

**2. HONEST toggles** (`honest.py`, all default off; with `HONEST` unset the module isn't even imported).

- `_restore()` ("rebuild the present") → an exact copy of the state at the entry of the innermost decision; no replay of the seed anywhere in the search path.
- Combat 1-ply search: each of the top-6 candidates is scored on **K** determinized copies (common random numbers: every candidate sees the same K sampled worlds), score = mean of the same discounted dense reward ± 50 terminal the deployed search uses.
- Draft teacher (`DRAFT_SEARCH=1`): each card candidate (take *k* / skip) is rolled to run end on `HONEST_DRAFT_K` determinized worlds, value = mean(floor + 1000·win). Combat inside those rollouts uses honest search with `HONEST_TAIL_K` (default 1).
- `HONEST_MACRO=1` (optional): the deployed macro heuristics also label actions with a one-step lookahead on the true state (e.g. what an unknown map node turns into); this runs that lookahead on a determinized copy too.

**3. Tests** (`test_honest.py`)

| test | result |
|---|---|
| (a) exact copy reproduces the replay-from-seed combat rollout value, every candidate | **125/125** identical |
| (a) exact copy reproduces a 400-step full-run continuation from a map screen | **9/9** identical |
| (b) determinize leaves the visible observation (1200/1235-d) unchanged | **187/187** |
| (b) different samples → different futures | combat **25/32** states (the rest are forced lines), run **9/9** |
| (c) all toggles off: 6 eval seeds byte-identical to `sim_logs/vdraft3_A` | **6/6** |

## Results (150 fixed eval seeds, 10007 + i·7919)

| arm | toggles | wins | avg floor | s/seed (median) | wall (14×) |
|---|---|---|---|---|---|
| mainline_v3 (clairvoyant replay search) | — | **9** | **26.06** | ~8 | 4m33s |
| honest combat, K=1 | `HONEST=1 HONEST_K=1` | 0 | 17.05 | 12.7 | 2m43s |
| honest combat, K=4 | `HONEST_K=4` | 1 | 21.13 | 64.9 | 14m47s |
| honest combat, K=8 | `HONEST_K=8` | 1 | **22.41** | 169.1 | 35m20s |
| honest combat K=4 + honest macro lookahead | `HONEST_K=4 HONEST_MACRO=1` | **2** | 21.21 | 65.2 | 15m06s |
| clairvoyant draft teacher (reference, `sim_logs/ds_ceiling`) | `DRAFT_SEARCH=1` | 44 | 37.79 | — | — |

**Draft teacher, 50 seeds** (i = 0..49; the same 50 seeds for every row):

| arm | wins / 50 | avg floor | reached Act 2 / Act 3 | s/seed median (max) | timeouts |
|---|---|---|---|---|---|
| mainline_v3 (clairvoyant) | 5 | 25.94 | — | ~8 | 0 |
| clairvoyant draft teacher (`ds_ceiling`) | **15** | **37.50** | — | — | 3 |
| honest combat K=4 | 1 | 19.80 | 25 / 1 | 65 | 0 |
| **honest draft teacher** (`HONEST_K=4 DRAFT_SEARCH=1 HONEST_DRAFT_K=4 HONEST_TAIL_K=1`) | **0** | **22.42** | 33 / 4 | 1587 (4901) | 0 |

The honest teacher costs **~26 min per seed** (≈2 h wall for 50 seeds at 14×); a 150-seed run would be ~6 h on this laptop and was not run, since 0/50 already settles the comparison with 15/50.

Reading:

- **Clairvoyance was most of the search's edge.** K=1 honest search is barely a search: one sampled future per candidate is noisy enough that the agent plays worse than its network prior on hard fights.
- **More samples help with diminishing returns:** +4.1 floors from K=1→4, +1.3 from K=4→8, at ~2.6× the cost per doubling. K=8 still leaves 3.7 floors and 8 wins on the table versus clairvoyant play.
- **The macro lookahead peek is small:** making it honest too costs nothing measurable (21.13 → 21.21, 1 → 2 wins; within noise).

## Loss taxonomy (`build_loss_taxonomy.py`)

| bucket | mainline_v3 | honest K=8 | honest K=4 + macro |
|---|---|---|---|
| A1 boss underprepared | 17 | **30** | **36** |
| A1 elite over-risk | 7 | **14** | 13 |
| A1 other (monster/event) | 4 | 4 | 11 |
| A2 monster attrition (outpaced) | 49 | 38 | 34 |
| A2 monster thrown winnable | 16 | 17 | 16 |
| A2 elite | 25 | 23 | 23 |
| A2 boss scaling | 10 | **18** | 9 |
| A3 (all) | 13 | 5 | 6 |
| **losses** | 141 | 149 | 148 |

The honest agent dies **earlier and in the big fights**: Act-1 boss deaths roughly double, Act-1 elite deaths double, and at K=8 Act-2 boss deaths nearly double. Act-2 monster attrition falls only because fewer runs get there. Knowing the enemy's move rolls and the draw order is worth the most in long, high-variance boss/elite fights — exactly where the clairvoyant search was quietly carrying the agent. The Act-2-entry deck profile of Act-2 monster losses is unchanged (median 20 cards, 9 starters, 2 upgrades), so this is a combat-play gap, not a drafting one.

**Same 50 seeds, including the draft teachers:**

| bucket | mainline_v3 | clairvoyant teacher | honest K=4 | honest K=8 | honest draft teacher |
|---|---|---|---|---|---|
| A1 boss underprepared | 4 | 0 | 16 | 11 | 11 |
| A1 elite over-risk | 3 | 0 | 7 | 6 | 2 |
| A1 other monster | 1 | 0 | 2 | 0 | 4 |
| A2 monster attrition (outpaced) | 18 | 6 | 8 | 13 | 9 |
| A2 monster thrown winnable | 4 | 1 | 6 | 7 | 8 |
| A2 elite | 11 | 4 | 6 | 7 | 6 |
| A2 boss scaling | 3 | 11 | 4 | 5 | 6 |
| A3 (all) | 1 | 10 | 0 | 0 | 4 |
| other (timeouts) | 0 | 3 | 0 | 0 | 0 |
| **losses** | 45 | 35 | 49 | 49 | 50 |

The honest teacher moves deaths later than honest combat alone (Act-1 losses 25 → 17; 4 runs reach Act 3), i.e. honest drafting is worth something, but it never converts: no run survives the Act-2/3 bosses. Its decks at Act-2 entry are **smaller and lighter on scaling** (median 18 cards, 1 power / 1 scaling vs 19–20 cards, 3 / 3 for every other arm): with noisy one-sample combat in the tail (`HONEST_TAIL_K=1`) the rollouts reward short-term tempo and undervalue scaling cards whose payoff comes in boss fights — the bias the clairvoyant teacher never had, because its tail combat knew every roll.

Note: the honest draft runs log `GameContext was not in a state with card rewards` to stderr; this comes from `get_card_reward()` being probed in `_offered()` on a non-reward screen inside rollouts (caught by its `try`), and is harmless.

## Timing

| arm | mean | median | max s/seed |
|---|---|---|---|
| honest K=1 | 13.7 | 12.7 | 49 |
| honest K=4 | 77.9 | 64.9 | 372 |
| honest K=8 | 189.7 | 169.1 | 804 |
| honest K=4 + macro | 80.3 | 65.2 | 759 |
| honest draft teacher (50 seeds) | 1763.3 | 1586.8 | 4901 |

All arms ran at 14 seeds in parallel on 16 threads; per-seed time is single-threaded Python driving the C++ sim. The cost scales ~linearly in K (one sampled rollout per candidate per K), so the Python rollout loop — not the simulator — is the bottleneck.

## Implications for the plan

- Every number before this experiment that used replay search (mainline_v1–v3, the draft teacher and its distilled V nets, the oracle ceilings) is an **upper bound with clairvoyance**, not achievable play. The honest baseline to beat is **K=8: 22.41 / 1 win** (or K=4 for a 2.6× cheaper loop).
- The biggest honest gap is in **boss and elite combat**, which points at better combat value estimates (a learned value / Q-net trained on honest rollouts, or deeper search with more samples where it matters: bosses and elites), not at drafting.
- The draft teacher's targets (`draft_data*.jsonl`) were labelled by clairvoyant rollouts and should be regenerated honestly before any further distillation.

## Reproduce (in WSL, `~/sts`)

```bash
.venv/bin/python test_honest.py                                         # (a)+(b)
HONEST=1 HONEST_K=4 .venv/bin/python eval_fullrun_analytics_v2.py 10007 macro
./run_honest.sh combat_K4 150 1800 "HONEST=1 HONEST_K=4"                # one arm, 150 seeds
./launch_run.sh honest_K8                                               # detached preset (also in the control panel)
.venv/bin/python run_status.py                                          # progress / ETA
```

Code: `~/sts` local git — `272b578` (bindings + honest.py + tests), `83436e6` (runners), `9d8955f` (run tracking).
