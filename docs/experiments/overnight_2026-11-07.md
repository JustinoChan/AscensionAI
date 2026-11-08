# Overnight report (2026-11-07)

Generated 2026-11-07 17:48:22 PST by `overnight_report.py` (no human or Claude in the loop).

## Jobs

| job | state | note |
|---|---|---|
| a. harvest mainline_h1 K=64, 1600 training seeds | done |  |
| b. train value net + ranking prior (GPU) + offline gate | done | prior PASS, value PASS |
| c1. A/B: prior top-3, K=64 | done | re-queued (state-file bug skipped it at 11:14) |
| c2. A/B: prior top-3 + V leaf depth 8, K=64 | done | re-queued (state-file bug skipped it at 11:14) |
| c3. A/B: prior top-3 + V leaf depth 8, K at equal wall-clock | skipped | stopped 17:35: equal-wall K=384 mis-calibrated (c2 runs were short); ~14x K64 cost/seed, ETA +6h, seeds near 7200s timeout. Replaced by ab_pv_d8_K128 at end of jobq |
| c4. A/B: prior top-3 + V leaf depth 16, K=64 | done | re-queued (state-file bug skipped it at 11:14) |
| d. holdout-150 K=128 check | done |  |
| e. OVERNIGHT_REPORT.md + dashboard export + commit/push | running | runs after the c-arms (queue_c.sh) |

## a. Harvest

```
seeds 1600  fights 23462  combat decisions 403231  macro decisions 152034  wins 30
n=1600 wins=30 avg_floor=24.50 timeouts=0  secs/seed mean=173.5 median=139.1 max=1842.9  wall=1532s
```

## b. Nets + offline gate

Decisions: train 364357, val 38874 (split by seed).

| ranking vs K=64 Q (val) | varN | prior_h1 |
|---|---|---|
| top1_agree | 0.495 | 0.468 |
| recall@1 | 0.507 | 0.482 |
| recall@2 | 0.692 | 0.701 |
| recall@3 | 0.819 | 0.829 |
| regret@1 | 2.649 | 3.081 |
| regret@2 | 1.463 | 1.371 |
| regret@3 | 0.805 | 0.703 |

Value: V R² 0.447, MAE 14.82 (constant predictor 24.48), corr 0.687; HP-lost MAE 0.080 (const 0.142); win Brier 0.0588 (const 0.0895), ECE 0.041.

**Gate:** prior PASS, value PASS — prior: regret@3 0.703 vs varN 0.805, recall@3 0.829 vs varN 0.819 (need both better); value: V_mae 14.82 vs const 24.48 (need <=0.8x), V_r2 0.447 (need >0.2), win ECE 0.041

## c. A/B on eval-150 vs mainline_h1 (cpp_1ply_K64: 6 wins, floor 24.61, wall 3361 s)

### ab_prior3_K64

`HONEST=1 HONEST_CPP=1 HONEST_K=64 HONEST_PRIOR=nets/h1/prior_h1.npz HONEST_TOPK=3`

```
n=150 wins=3 avg_floor=23.00 timeouts=0  secs/seed mean=89.8 median=75.2 max=430.8  wall=1190s
A = cpp_1ply_K64
B = ab_prior3_K64
shared seeds: 150
wins   A   6 (4.0%, 95% CI 1.8-8.5%)   B   3 (2.0%, 95% CI 0.7-5.7%)
win flips: A-only 5, B-only 2  -> exact McNemar p = 0.453
avg floor A 24.61  B 23.00   paired diff B-A = -1.61  (bootstrap 95% CI -2.91 .. -0.35)  *

outcome          A     B    B-A
WIN              6     3     -3
A1_boss         23    25     +2
A1_elite         8     9     +1
A1_other         7     8     +1
A2_monster      46    55     +9
A2_elite        34    32     -2
A2_boss         16    10     -6
A3              10     8     -2
```

### ab_pv_d8_K64

`HONEST=1 HONEST_CPP=1 HONEST_K=64 HONEST_PRIOR=nets/h1/prior_h1.npz HONEST_TOPK=3 HONEST_VNET=nets/h1/value_h1.npz HONEST_DEPTH=8`

```
n=150 wins=2 avg_floor=22.37 timeouts=0  secs/seed mean=46.5 median=40.2 max=267.7  wall=565s
A = cpp_1ply_K64
B = ab_pv_d8_K64
shared seeds: 150
wins   A   6 (4.0%, 95% CI 1.8-8.5%)   B   2 (1.3%, 95% CI 0.4-4.7%)
win flips: A-only 4, B-only 0  -> exact McNemar p = 0.125
avg floor A 24.61  B 22.37   paired diff B-A = -2.23  (bootstrap 95% CI -3.55 .. -0.89)  *

outcome          A     B    B-A
WIN              6     2     -4
A1_boss         23    23     +0
A1_elite         8    11     +3
A1_other         7    10     +3
A2_monster      46    61    +15
A2_elite        34    22    -12
A2_boss         16    13     -3
A3              10     8     -2
```

- ab_pv_d8_Keq: not run
### ab_pv_d16_K64

`HONEST=1 HONEST_CPP=1 HONEST_K=64 HONEST_PRIOR=nets/h1/prior_h1.npz HONEST_TOPK=3 HONEST_VNET=nets/h1/value_h1.npz HONEST_DEPTH=16`

```
n=150 wins=2 avg_floor=23.37 timeouts=0  secs/seed mean=67.8 median=58.4 max=378.3  wall=804s
A = cpp_1ply_K64
B = ab_pv_d16_K64
shared seeds: 150
wins   A   6 (4.0%, 95% CI 1.8-8.5%)   B   2 (1.3%, 95% CI 0.4-4.7%)
win flips: A-only 5, B-only 1  -> exact McNemar p = 0.219
avg floor A 24.61  B 23.37   paired diff B-A = -1.24  (bootstrap 95% CI -2.42 .. -0.10)  *

outcome          A     B    B-A
WIN              6     2     -4
A1_boss         23    24     +1
A1_elite         8    10     +2
A1_other         7     7     +0
A2_monster      46    55     +9
A2_elite        34    27     -7
A2_boss         16    15     -1
A3              10    10     +0
```


## d. Holdout K=128

```
n=150 wins=5 avg_floor=26.72 timeouts=0  secs/seed mean=1270.7 median=1084.9 max=4156.5  wall=14166s
A = holdout_K64
B = holdout_K128
shared seeds: 150
wins   A   7 (4.7%, 95% CI 2.3-9.3%)   B   5 (3.3%, 95% CI 1.4-7.6%)
win flips: A-only 3, B-only 1  -> exact McNemar p = 0.625
avg floor A 26.67  B 26.72   paired diff B-A = +0.05  (bootstrap 95% CI -0.74 .. +0.94)

outcome          A     B    B-A
WIN              7     5     -2
A1_boss         12    11     -1
A1_elite         5     5     +0
A1_other         5     5     +0
A2_monster      52    48     -4
A2_elite        30    33     +3
A2_boss         24    28     +4
A3              15    15     +0
```

## Next

- Read the gates above: an arm passes if floor ≥ K=64 at ≤ 1/3 of the wall time, or a better floor at the same wall time.
- Phase 3 step 2: honest macro search (card reward/skip first), one lever per A/B on eval-150.
