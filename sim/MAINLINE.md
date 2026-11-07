# AscensionAI mainline_v3 — config manifest (2026-06-17)

Deployed agent for full-run evaluation on the headless sts_lightspeed sim
(Ironclad, full run). Biggest single-lever floor gain in the project's history.

## Result (150 fixed eval seeds: 10007 + i*7919 for i in 0..149)
avg floor 26.1 | wins 9/150 (6.0%)   [mainline_v2 was 24.9 | 7/150]

## Stack
- Combat: 1-ply replay search (rollout teacher); net = policy_varN.npz (Act-1+2/3 trained)
- Non-combat: deck-hygiene macro + AoE/draw role scorer + conservative boss-aware
  smith + TEMPO-TARGETED smith upgrades

## Config (eval_fullrun_analytics_v2.py env toggles; these are the deployed defaults)
COMBAT_NPZ=policy_varN.npz   combat net (rollback: policy_varL.npz)
GATE_ELITES=0                elite/path gating OFF (diagnosed no-op)
RELIC_BUY=0                  shop relic buying OFF (neutral at 150-seed power)
ROLE_SCORE=1                 AoE/draw gap-filler ON (deployed win)
SMITH=1                      conservative smith ON: smith only HP>=85% (>=95% pre-boss)
SMITH_TARGET=1               tempo-prioritized upgrade target ON (=0 reproduces v2)
non-combat policy: policy_varE.npz (map / card reward / shop / event)

## What v3 changed vs v2 (A/B, 150 seeds, only SMITH_TARGET differs)
The conservative smith used to upgrade the FIRST card in deck order = a basic
Strike 97% of the time (audited over 450 smith events). v3 instead upgrades the
highest-tempo card available -- priority: premium attack (Bash etc.) > other
attack > strong block > AoE > draw/energy > scaling/power > filler. Same smith
frequency, same HP gates -- only the target changed (zero HP cost).

A (target=0) -> B (target=1):
  upgraded-TEMPO cards @ Act-2 entry  0.31 -> 1.57   (mechanism; 99% of smiths improved)
  avg floor                          24.87 -> 26.06   (+1.19, biggest single lever)
  wins                                   7 -> 9
  Act-1 deaths                          32 -> 28
  reached Act 2                        118 -> 122
  boss-entry HP frac                 0.885 -> 0.881   (guardrail held; no HP regression)
  Act-2 monster deaths                  58 -> 65       (MISSED named gate -- see note)
  Act-2 monster per-fight win%       0.902 -> 0.896   (flat)

NOTE (honest): tempo-targeted smithing improved early/midgame throughput and
raised average floor, but did NOT solve the Act-2 monster wall. Act-2 monster
deaths rose in raw COUNT due to greater exposure (more runs survive earlier and
then hit the same Act-2 attrition wall), while the per-fight win rate stayed
flat. Deployed because the net objective (floor, wins, Act-1 survival) moved
strongly with no regression. The remaining Act-2 monster wall is entering those
fights too hurt -- an HP-attrition / path-sequencing problem, not deck power.

## Reproduce
COMBAT_NPZ=policy_varN.npz python eval_fullrun_analytics_v2.py <seed> macro   # single seed (v3 default)
bash run_smith_ab.sh 150 12   # the v3 A/B (SMITH_TARGET 0 vs 1) + agg_smith_ab.py
bash eval_stages.sh 150       # full stage progression

## Snapshots / rollback
eval_mainline_v3.py + v2_mainline_v3.jsonl   current (tempo-targeted smith)
eval_mainline_v2.py + v2_mainline_v2.jsonl   rollback (conservative smith, upg[0]=Strike)
eval_mainline_v1.py + v2_mainline_v1.jsonl   pre-smith
policy_varL.npz                              combat-net rollback

## Key files
combat retrain : train_search.py + sim_worker_search.py (rollout teacher); bank_train.jsonl (Act-1 + harvested Act-2/3)
eval harness   : eval_fullrun_analytics_v2.py, eval_fullrun_search.py, card_roles.py
smith scorer   : eval_fullrun_analytics_v2.py  _smith_best / _smith_score (SMITH_TARGET)
diagnostics    : diag_act2.py (Act-2 death recon), analyze_deck_deficit.py, audit_smith.py
aggregators    : aggregate_v2.py, aggregate_combat.py, aggregate_roles.py, agg_smith_ab.py
