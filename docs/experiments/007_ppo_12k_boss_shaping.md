# 007 PPO 12k + Boss Reward Shaping

**Date:** 2026-05-21
**Commit:** `f8f93b7`
**Training games:** 12,088
**PPO updates:** 1,488
**Eval:** 200 fixed-seed games (greedy policy, no exploration)

---

## Summary

This experiment covers the training period from ~7,100 to ~12,088 games, during which three major changes were deployed:

1. **Boss reward shaping** for all Act 1-3 bosses
2. **Accelerated BC anchor decay** (floor 0.01 → 0.001, faster decay rates)
3. **Heuristic overhaul** (card tiers, event handling, boss relic picks)

The policy is currently in an "exploration valley" — boss conversion dropped from 31.1% to 21.7% as the BC anchor loosened and the policy began exploring beyond heuristic habits. The 256x256 MLP (236k params) appears to be at representational capacity, having plateaued at ~15 avg floor since game 3,000.

## Changes Since Experiment 005

### Boss Reward Shaping

Per-boss combat signals added to the reward function:

| Boss | Mechanic | Signal |
|---|---|---|
| Guardian | Sharp Hide (3 dmg per attack card) | HP loss penalty while Sharp Hide active |
| Hexaghost | Big damage hits (Inferno, Divider) | Extra penalty for hits >= 20 HP |
| Slime Boss | Splits at 50% HP | Bonus for overkill past the split threshold |
| Bronze Automaton | Hyper Beam (45 dmg) then Stun | Big hit penalty |
| Champ | Enrages (Str >= 6) | HP loss penalty while enraged |
| Donu + Deca | Donu buffs Deca | Priority kill target for Donu |

Boss kill reward: +8.0 base + up to +8.0 scaled by HP preserved at kill.

### BC Anchor Decay

| Parameter | Before | After |
|---|---|---|
| Floor | 0.01 | 0.001 |
| Fast decay | 0.90 | 0.85 |
| Slow decay | 0.98 | 0.95 |

Auto-tune now oscillates BC coefficient between 0.001-0.009 based on policy improvement signals.

### Heuristic Overhaul

- Card tier rework across all Ironclad cards
- Battle Trance: S-tier first copy, -15 score penalty for duplicates
- Event handler fixes for edge cases
- Boss relic pick priority improvements

## 200-Game Eval Results

| Metric | PPO 5k (May 17) | PPO 12k (May 20) | Heuristic |
|---|---|---|---|
| Avg floor | 15.44 | 14.83 | 15.78 |
| Avg reward | 4.03 | -0.95 | 8.44 |
| Boss conversion | 31.1% (41/132) | 21.7% (28/129) | 39.0% (39/100) |
| Elite win rate | 79.9% (175/219) | 77.3% (153/198) | 81.9% (176/215) |
| Act 2+ reach | 20.0% | 13.5% | 26.0% |
| Victories | 0 | 0 | 0 |

### Floor Distribution

49.5% of all deaths in the 12k eval occur on floor 16 (Act 1 boss), up from 42.5% in the 5k eval. The floor-16 wall remains the primary bottleneck.

## Training Metrics at 12k

| Metric | Value |
|---|---|
| Explained variance | 0.78 (flat since game 4,000) |
| Normalized entropy | 0.27 (healthy, up from 0.19) |
| BC coefficient | 0.007 (auto-tuned) |
| KL divergence | 0.003 (stable) |
| Stale rollouts | 0 (down from ~48/update after reducing 7→5 workers) |
| Value loss | 12.5 (declining slowly) |

## Diagnosis

The policy regression from 5k→12k eval is explained by two factors:

1. **Exploration valley**: The BC anchor loosened from 0.01 to 0.001, releasing the policy from heuristic imitation. Boss conversion dipped from ~24% to ~19% in training, with early signs of recovery in the latest windows.

2. **Network capacity ceiling**: The 256x256 MLP reached ~15 avg floor by game 3,000 and has not meaningfully improved since. Explained variance has been flat at 0.78 for 8,000+ games. The network cannot represent boss-specific strategies or Act 2+ play patterns.

## Next Steps

- **Network upgrade** to (512, 256, 256) with GELU activation via warm transfer from current checkpoint
- Re-evaluate at ~15k games on the larger network
- Monitor boss conversion recovery as the new reward signals propagate
