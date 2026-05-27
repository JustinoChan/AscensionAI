# 008 PPO 19k + HP-Urgency Heal Reward + Network Upgrade

**Date:** 2026-05-27
**Commit:** `f5d6790`
**Training games:** 19,493
**PPO updates:** 2,429
**Eval:** 200 fixed-seed games (greedy policy, no exploration)

---

## Summary

This experiment covers the training period from ~12,088 to ~19,493 games, during which two major changes were deployed:

1. **Network upgrade** from (256, 256) Tanh MLP (~236K params) to (512, 256, 256) GELU MLP (~504K params) via warm transfer
2. **HP-urgency-scaled heal reward** at rest sites to fix the agent choosing upgrades over healing at critical HP
3. **Elite win bonus** increased to +4.0

The result is the strongest eval to date: boss win rate jumped from 21.7% to 38.1%, and 8 runs reached floor 30+ including two Act 3 runs (floors 42 and 46).

## Changes Since Experiment 007

### Network Upgrade (May 21)

| Field | Before | After |
|---|---|---|
| Architecture | (256, 256) Tanh | (512, 256, 256) GELU |
| Parameters | ~236K | ~504K |
| Transfer method | — | Warm transfer: copy compatible weights, zero-pad widened layers, identity-init new layers |

The upgrade doubled representational capacity to break the ~15 avg floor plateau observed since game 3,000. Warm transfer preserved learned behavior with no regression in the first 1,300 games post-upgrade.

### HP-Urgency Heal Reward (May 27)

Rest-site heal reward scaled by missing HP fraction:

```
heal_reward = 0.025 * hp_healed * (1 - hp_before / max_hp)
```

| Scenario | HP% before | Heal reward | Upgrade reward | Agent picks |
|---|---|---|---|---|
| Critical (23/80 HP) | 29% | ~0.45 | 0.30 | Heal |
| Moderate (48/80 HP) | 60% | ~0.25 | 0.30 | Upgrade |
| Healthy (64/80 HP) | 80% | ~0.12 | 0.30 | Upgrade |

This fixes the pre-boss "upgrade at 23 HP" behavior that was causing preventable deaths.

### Elite Win Bonus (May 22)

Elite win bonus increased from +3.0 to +4.0 to make elite-seeking paths more clearly positive-EV.

### HP-Scaled Floor Advance (May 22)

Replaced flat +0.75 floor advance reward with hybrid formula: 0.50 base + 0.25 x (HP / max HP). Gives a dense per-floor signal that arriving healthy is better than arriving at low HP.

## 200-Game Eval Results

| Metric | PPO 12k (May 20) | PPO 19k (May 27) | Heuristic |
|---|---|---|---|
| Avg floor | 14.83 | **14.66** | 15.78 |
| Best floor | — | **46** | 33 |
| Avg reward | -0.95 | **9.60** | 8.44 |
| Boss conversion | 21.7% (28/129) | **38.1% (43/113)** | 39.0% (39/100) |
| Elite win rate | 77.3% (153/198) | 69.6% (133/191) | 81.9% (176/215) |
| Act 2+ reach | 13.5% | **20.0% (40/200)** | 26.0% |
| Floor 20+ | 10.5% | **19.5% (39/200)** | 23.3% |
| Floor 30+ | — | **8 runs** | — |
| Victories | 0 | 0 | 0 |

### Key Observations

- **Boss conversion nearly matched heuristic** (38.1% vs 39.0%), up from 21.7% at 12k games
- **Act 2 reach doubled** from 13.5% to 20.0%
- **Average reward flipped positive** from -0.95 to +9.60, now exceeding the heuristic (+8.44)
- **8 runs past floor 30** including two Act 3 runs (floors 42 and 46)
- Elite win rate dropped from 77.3% to 69.6% — the policy may be taking riskier elite fights or prioritizing boss preparation over elite optimization

### Floor Distribution

The floor-16 death wall remains present but less dominant. With 38.1% boss conversion (up from 21.7%), the agent is clearing the Act 1 boss nearly twice as often. Deaths are now more distributed across Act 2 floors.

## Training Metrics at 19k

| Metric | Value |
|---|---|
| Total games | 19,493 |
| PPO updates | 2,429 |
| Workers | 5 |
| Last 500 avg floor | 14.40 |
| Latest normalized entropy | 0.179 |
| Latest approx KL | 0.0046 |
| Latest clip fraction | 0.072 |
| Latest VF loss | 13.56 |
| Stale rollouts | 0 |

## Analysis

The combination of network upgrade + HP-urgency heal reward + elite bonus increase produced the largest single-eval improvement in the project's history. Boss conversion nearly doubled (21.7% to 38.1%), and the average reward swung from -0.95 to +9.60.

The network upgrade was necessary infrastructure — the old 256x256 network had been plateaued for 8,000+ games. The heal reward directly addressed an observable behavioral flaw (upgrading at 23 HP before bosses). The elite bonus increase reinforced the positive-EV path of seeking elite fights.

Entropy has settled lower (0.179) compared to the 0.27 range at 12k. This is expected with a larger network that can represent more deterministic strategies without collapsing exploration prematurely. KL remains well below the 0.03 target.

## Next Steps

- Monitor elite win rate recovery (69.6% is below the 77-82% historical range)
- Continue training toward 25k+ games with current reward structure
- Target first full game victory as the next milestone
- Consider Act 2+ specific reward signals if the agent continues to struggle past floor 20
