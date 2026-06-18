# 011 Tempo-Targeted Smithing — the mainline_v3 Checkpoint

**Dates:** 2026-06-16 → 2026-06-17
**Result:** **mainline_v3 — avg floor 26.1, 9 wins / 150 fixed seeds (6.0%)** — the largest single-lever floor gain in the project so far (+1.2 over mainline_v2), at zero HP cost.

---

## Summary

This phase opened with a specific target — *reduce the Act-2 monster deaths (58/150) that dominate mainline_v2's failures* — and a hypothesized lever: surgical deeper combat search. A **diagnose-first** recon redirected it twice before any code was written, and it ended in a deck-efficiency lever, not a search one.

The core thesis: narrow deeper search and raw deck-*power* explanations were both ruled out by diagnostics; **upgrade efficiency** was the cleaner lever. Tempo-targeted smithing preserved HP while upgrading higher-impact cards, producing the biggest single-lever floor gain to date. It did **not** solve the Act-2 monster wall — raw Act-2 monster deaths *rose* because more runs reached Act 2, while the per-fight win rate stayed flat.

## The diagnostic chain (why the lever is what it is)

Every step was a measurement, not a guess:

| step | finding | consequence |
|---|---|---|
| **Act-2 death recon** (150 seeds) | of the 58 deaths: median **28% HP at fight entry**, **88% "lost on arrival"** (the rollout already predicts a loss from move 1), enemy at **~40% HP** when we die, **93%** facing lethal damage at the death turn | deeper search *at the death turn* is futile — every line already loses. **Original premise killed.** |
| **Lever-attribution cross-tab** (entered ≤40% HP × enemy ≤20% at death) | entered-low **&** crushed = **60%**; "threw a winnable fight" (the only deeper-search-addressable bucket) = **16%** | the wall is *upstream* — entering too hurt + out-tempo'd — not in-fight search |
| **Deck-deficit pinpoint** (same data, no new run) | losing decks are **not** damage-deficient — equal damage, *more* scaling/powers than decks that cleared Act 2; the only monotonic success-correlates are **upgrades** (winners 2.7 vs losers 1.8–1.0) and block | a deck-*value*/drafting model would have been a wrong build. The lever is **upgrade efficiency**, not deck power. |
| **Smith-target audit** (450 smith events) | the conservative smith upgraded a basic **Strike/Defend 97% of the time** (it picked the first card in deck order), missing a premium target 98% of the time | smith *targeting* is real, untapped, and HP-free |

The HP-safe form matters: the upgrade *count* gap (1.8→2.7) is HP-coupled — smithing more often means skipping heals, which worsens the very attrition the recon found (and is exactly what an earlier naive "smith when safe" did, regressing wins). So the lever keeps the **same smith frequency and HP gates** and changes **only the target**.

## The change

At the smith upgrade screen, instead of upgrading the first card in deck order, score every available card by Act-2 tempo value and upgrade the best:

> premium attack (Bash, Heavy Blade, …) > other attack > strong block > AoE > draw/energy > scaling/power > basic Strike/Defend / junk

Paired over 450 smith events, the new scorer upgrades a premium tempo card **100%** of the time (Bash→Bash+ being the most common — 10 damage + 3 Vulnerable, both frontload *and* damage amplification), improving the tempo tier on **99%** of smiths and never worsening it.

## A/B result (150 fixed seeds, only the smith target differs)

| metric | mainline_v2 | **mainline_v3** | |
|---|---|---|---|
| upgraded-tempo cards @ Act-2 entry | 0.31 | **1.57** | ✅ mechanism |
| **avg floor** | 24.87 | **26.06** | ✅ +1.19 (largest single lever) |
| **wins** | 7 | **9** | ✅ |
| Act-1 deaths | 32 | **28** | ✅ |
| reached Act 2 | 118 | 122 | ✅ |
| **boss-entry HP** | 88.5% | 88.1% | ✅ unchanged (no HP cost) |
| Act-2 monster deaths | 58 | **65** | ❌ *up* |
| Act-2 monster per-fight win% | 90.2% | 89.6% | ≈ flat |

**Honest note on the named gate.** The phase's stated objective — fewer Act-2 monster deaths — was *missed*: the raw count rose 58→65. But the **per-fight Act-2 monster win rate is flat (90.2→89.6%)**, and **more runs now reach Act 2 (118→122) and push deeper**. So the rise is **exposure**, not a combat regression — more runs survive earlier and then hit the same Act-2 attrition wall. Better early tempo (Bash+, harder attacks) wins Act-1 fights cleaner and carries the agent further; it does not fix Act-2 monster combat. mainline_v3 was deployed because the net objective (floor, wins, Act-1 survival) moved strongly with zero regression, and the missed proxy gate is recorded here plainly.

## mainline_v3 — deployed config

- **Combat:** 1-ply replay search (rollout teacher), net `policy_varN.npz`.
- **Non-combat:** deck-hygiene macro + AoE/draw role scorer + conservative boss-aware smith + **tempo-targeted smith upgrades**.
- **Toggles:** `GATE_ELITES=0`, `RELIC_BUY=0`, `ROLE_SCORE=1`, `SMITH=1`, `SMITH_TARGET=1`, `COMBAT_NPZ=policy_varN.npz`.
- **Rollback:** `SMITH_TARGET=0` (or `eval_mainline_v2.py`) reproduces mainline_v2 exactly; `policy_varL.npz` for combat.

## Conclusion & next phase

Tempo-targeted smithing is the strongest single lever found and is now the deployed checkpoint. Crucially, this phase also *localized* the remaining wall by elimination: deck tempo improved, per-fight Act-2 monster combat stayed flat, and boss-entry HP did not regress — so the Act-2 monster wall is **entering those fight chains too hurt** (28% median entry HP), an HP-attrition / path-sequencing problem rather than deck power or combat skill.

**Next: an HP-attrition / path-sequencing diagnostic against mainline_v3** — is there rest/path slack to arrive at Act-2 monster rooms healthier? That is a fresh diagnostic loop and a deliberate new phase.
