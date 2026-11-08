# Seed sets (all Ironclad A0, sts_lightspeed)

| set | generator | n | use |
|---|---|---|---|
| eval-150 | 10007 + i*7919, i = 0..149 | 150 | development A/B (every arm since mainline_v1) |
| training | 10007 + i*7919, i = 150..1749 | 1600 | Phase 3 harvest (nets/h1); never evaluated on |
| holdout-150 | 20011 + i*104729, i = 0..149 | 150 | gates only |
| eval500 | 40009 + i*1000003, i = 0..499 | 500 | large baseline evals (tight CIs, death taxonomy); added 2026-11-08 |

Overlap check (2026-11-08): eval500 shares 0 seeds with eval-150, training, holdout-150, or any seed ever run under sim_logs/.
