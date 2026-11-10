# 018 RL Round 3: Visible-Map Features + Boss-Kill Bonus

**Status:** auto-generated 2026-11-10 07:00 PST from the run logs; numbers are final, commentary to follow.

## Setup

Same candidate-scorer PPO as 017, with two changes:

- **`MACRO_MAPFEAT=1`:** 28 extra inputs summarising the visible current-act map.
  - For each room type, the min and max count over the paths to the boss, plus floors left and the number of paths.
  - Given for the position now and after each option.
- **`RL_BOSS_BONUS=0.3`:** reward = floor/50 + win + 0.3 per act boss beaten.

**Data and training:**

- Fresh behaviour-cloning data: 1000 K=8 runs, seeds 820000013+19i.
- PPO: 150 iterations × 100 games at K=8 on 14 workers, seeds 640000007+13i.
- Same 100 validation seeds as 017.

BC: `saved nets/rl3/pol_bc.npz, best val acc 0.7855`

## Validation (greedy, K=8, 100 seeds)

| checkpoint | floor | wins |
|---|---|---|
| bc | 19.69 | 0 |
| it009 | 23.00 | 1 |
| it019 | 24.32 | 2 |
| it029 | 26.24 | 2 |
| it039 | 29.05 | 5 |
| it049 | 29.53 | 3 |
| it059 | 30.10 | 5 |
| it069 | 32.73 | 10 |
| it079 | 31.15 | 6 |
| it089 | 31.00 | 5 |
| it099 | 29.66 | 7 |
| it109 | 31.99 | 7 |
| it119 | 30.50 | 5 |
| it129 | 31.99 | 3 |
| it139 | 31.97 | 4 |
| it149 | 31.31 | 7 |

017 reference on the same seeds: it099 = 29.65 floor, 7 wins; heuristics = 21.39, 0.

## eval500 gate (honest K=64)

```
gate policy: checkpoint it069 (val floor 32.73, wins 10/100; best=it069, last=it149)

A = eval500_K64
B = eval500_rl3_K64
shared seeds: 500
wins   A   6 (1.2%, 95% CI 0.6-2.6%)   B  51 (10.2%, 95% CI 7.8-13.2%)
win flips: A-only 6, B-only 51  -> exact McNemar p = 0.000
avg floor A 24.21  B 33.20   paired diff B-A = +8.98  (bootstrap 95% CI +7.80 .. +10.16)  *

outcome          A     B    B-A
WIN              6    51    +45
A1_boss         74    45    -29
A1_elite        21     2    -19
A1_other        21     1    -20
A2_monster     187    92    -95
A2_elite        95    61    -34
A2_boss         53   134    +81
A2_other         0     7     +7
A3              43   107    +64

eval500_K64
  Act 1: reached 500, died 116 (23.2% of reaching)  {'BOSS': 74, 'ELITE': 21, 'MONSTER': 18, 'EVENT': 3}
  Act 2: reached 384, died 335 (87.2% of reaching)  {'MONSTER': 187, 'ELITE': 95, 'BOSS': 53}
  Act 3: reached  49, died  43 (87.8% of reaching)  {'MONSTER': 19, 'ELITE': 11, 'BOSS': 9, 'EVENT': 4}
eval500_rl3_K64
  Act 1: reached 500, died  48 (9.6% of reaching)  {'BOSS': 45, 'ELITE': 2, 'MONSTER': 1}
  Act 2: reached 452, died 294 (65.0% of reaching)  {'BOSS': 134, 'MONSTER': 92, 'ELITE': 61, 'EVENT': 7}
  Act 3: reached 158, died 107 (67.7% of reaching)  {'BOSS': 47, 'ELITE': 29, 'MONSTER': 24, 'EVENT': 7}

LOSS TAXONOMY  (eval500_rl3_K64)
runs 500  wins 51  losses 449

 134  A2_boss_scaling
  61  A2_elite
  55  A2_monster_attrition_outpaced
  47  A3_boss
  45  A1_boss_underprepared
  37  A2_monster_thrown_winnable
  29  A3_elite
  24  A3_monster
   7  A2_other_EVENT
   7  A3_event
   2  A1_elite_overrisk
   1  A1_other_MONSTER

sum check: 449 == 449 : True

--- Act-2-monster losses: deck @ Act-2 entry (n=92) ---
median size 17.0  starters 8.0  upgraded 2.0  upg_tempo 2.0
A = eval500_rl_K64
B = eval500_rl3_K64
shared seeds: 500
wins   A  28 (5.6%, 95% CI 3.9-8.0%)   B  51 (10.2%, 95% CI 7.8-13.2%)
win flips: A-only 23, B-only 46  -> exact McNemar p = 0.008
avg floor A 32.14  B 33.20   paired diff B-A = +1.06  (bootstrap 95% CI -0.15 .. +2.22)

outcome          A     B    B-A
WIN             28    51    +23
A1_boss         33    45    +12
A1_elite         6     2     -4
A1_other         2     1     -1
A2_monster      90    92     +2
A2_elite        76    61    -15
A2_boss        164   134    -30
A2_other         0     7     +7
A3             101   107     +6

fight                                  eval500_rl_K64 won/met (HP) eval500_rl3_K64 won/met (HP)
A1 BOSS HEXAGHOST                      170/188  90.4% (46.4)    173/195  88.7% (44.7)   
A1 BOSS SLIME_BOSS                     138/141  97.9% (23.8)    136/140  97.1% (21.6)   
A1 BOSS THE_GUARDIAN                   151/163  92.6% (41.9)    143/162  88.3% (41.4)   
A1 ELITE GREMLIN_NOB                   123/124  99.2% (15.3)    113/113 100.0% (16.2)   
A1 ELITE LAGAVULIN                     145/147  98.6% (23.5)    135/136  99.3% (21.4)   
A1 ELITE THREE_SENTRIES                137/140  97.9% (19.4)    133/134  99.3% (21.4)   
A2 BOSS AUTOMATON                       46/100  46.0% (51.6)     47/107  43.9% (55.9)   
A2 BOSS CHAMP                           40/97   41.2% (50.5)     61/87   70.1% (43.3)   
A2 BOSS COLLECTOR                       43/96   44.8% (51.0)     50/98   51.0% (50.6)   
A2 ELITE BOOK_OF_STABBING              102/130  78.5% (25.6)     98/108  90.7% (23.6)   
A2 ELITE GREMLIN_LEADER                 89/105  84.8% (20.5)     79/97   81.4% (22.2)   
A2 ELITE SLAVERS                        89/118  75.4% (28.4)     72/102  70.6% (33.9)   
A3 BOSS AWAKENED_ONE                    12/24   50.0% (51.9)     18/30   60.0% (56.5)   
A3 BOSS DONU_AND_DECA                    8/19   42.1% (50.4)     19/39   48.7% (52.8)   
A3 BOSS TIME_EATER                       8/21   38.1% (50.2)     14/29   48.3% (52.6)   
A3 ELITE GIANT_HEAD                     15/30   50.0% (25.4)     34/45   75.6% (30.2)   
A3 ELITE NEMESIS                        22/27   81.5% (27.4)     38/46   82.6% (27.4)   
A3 ELITE REPTOMANCER                    24/32   75.0% (23.2)     24/34   70.6% (32.7)
```

## Overnight branch

```
[2026-11-09 22:18:59] round 3 eval500: 51 wins floor 33.2 vs it099 28 wins floor 32.14 -> branch A
```

### GATE_holdout_rl3_K64.txt

```
A = holdout_K64
B = holdout_rl3_K64
shared seeds: 150
wins   A   7 (4.7%, 95% CI 2.3-9.3%)   B  22 (14.7%, 95% CI 9.9-21.2%)
win flips: A-only 4, B-only 19  -> exact McNemar p = 0.003
avg floor A 26.67  B 34.68   paired diff B-A = +8.01  (bootstrap 95% CI +5.81 .. +10.17)  *

outcome          A     B    B-A
WIN              7    22    +15
A1_boss         12    10     -2
A1_elite         5     2     -3
A1_other         5     0     -5
A2_monster      52    29    -23
A2_elite        30    17    -13
A2_boss         24    32     +8
A2_other         0     1     +1
A3              15    37    +22
wins   A   9 (6.0%, 95% CI 3.2-11.0%)   B  22 (14.7%, 95% CI 9.9-21.2%)
win flips: A-only 7, B-only 20  -> exact McNemar p = 0.019
avg floor A 32.72  B 34.68   paired diff B-A = +1.96  (bootstrap 95% CI -0.33 .. +4.30)
```

### GATE_rl3c_eval500.txt

```
gate policy: checkpoint it099 (val floor 33.82, wins 10/100; best=it099, last=it119)

A = eval500_K64
B = eval500_rl3c_K64
shared seeds: 500
wins   A   6 (1.2%, 95% CI 0.6-2.6%)   B  53 (10.6%, 95% CI 8.2-13.6%)
win flips: A-only 6, B-only 53  -> exact McNemar p = 0.000
avg floor A 24.21  B 35.58   paired diff B-A = +11.36  (bootstrap 95% CI +10.22 .. +12.48)  *

outcome          A     B    B-A
WIN              6    53    +47
A1_boss         74    26    -48
A1_elite        21     3    -18
A1_other        21     0    -21
A2_monster     187    80   -107
A2_elite        95    37    -58
A2_boss         53   151    +98
A2_other         0     5     +5
A3              43   145   +102

eval500_K64
  Act 1: reached 500, died 116 (23.2% of reaching)  {'BOSS': 74, 'ELITE': 21, 'MONSTER': 18, 'EVENT': 3}
  Act 2: reached 384, died 335 (87.2% of reaching)  {'MONSTER': 187, 'ELITE': 95, 'BOSS': 53}
  Act 3: reached  49, died  43 (87.8% of reaching)  {'MONSTER': 19, 'ELITE': 11, 'BOSS': 9, 'EVENT': 4}
eval500_rl3c_K64
  Act 1: reached 500, died  29 (5.8% of reaching)  {'BOSS': 26, 'ELITE': 3}
  Act 2: reached 471, died 273 (58.0% of reaching)  {'BOSS': 151, 'MONSTER': 80, 'ELITE': 37, 'EVENT': 5}
  Act 3: reached 198, died 145 (73.2% of reaching)  {'BOSS': 78, 'ELITE': 29, 'MONSTER': 27, 'EVENT': 11}

LOSS TAXONOMY  (eval500_rl3c_K64)
runs 500  wins 53  losses 447

 151  A2_boss_scaling
  78  A3_boss
  48  A2_monster_attrition_outpaced
  37  A2_elite
  32  A2_monster_thrown_winnable
  29  A3_elite
  27  A3_monster
  26  A1_boss_underprepared
  11  A3_event
   5  A2_other_EVENT
   3  A1_elite_overrisk

sum check: 447 == 447 : True

--- Act-2-monster losses: deck @ Act-2 entry (n=80) ---
median size 17.0  starters 8.5  upgraded 3.0  upg_tempo 2.0
median aoe 1.0  draw_energy 1.0  powers 2.0  scaling 2.0
A = eval500_rl_K64
B = eval500_rl3c_K64
shared seeds: 500
wins   A  28 (5.6%, 95% CI 3.9-8.0%)   B  53 (10.6%, 95% CI 8.2-13.6%)
win flips: A-only 21, B-only 46  -> exact McNemar p = 0.003
avg floor A 32.14  B 35.58   paired diff B-A = +3.44  (bootstrap 95% CI +2.29 .. +4.58)  *

outcome          A     B    B-A
WIN             28    53    +25
A1_boss         33    26     -7
A1_elite         6     3     -3
A1_other         2     0     -2
A2_monster      90    80    -10
A2_elite        76    37    -39
A2_boss        164   151    -13
A2_other         0     5     +5
A3             101   145    +44
```
