# AscensionAI: Path to Max Win Rate on All Characters and Ascensions

Date: 2026-11-06
Status: adopted. Phases 0–2 are done (`docs/experiments/014_honest_search.md`, `015_cpp_search.md`); Phase 3 is next.

## 1. Diagnosis (why Ironclad was stuck)

Every search component replayed the run from the **true seed**: combat rollouts saw the real draw order and enemy rolls, the draft teacher played each candidate to run end on the real seed, and the oracle's best-of-8 routing picked outcomes after the fact. So mainline_v3 (9/150), the draft teacher (44/150) and the 76% oracle ceiling were partly clairvoyant. That also explains why every distillation failed (varK/L/M capped at ~70% agreement whatever the capacity; draft V3 passed offline and lost its A/B; the combat Q student had median regret 0, mean regret 8.3, and collapsed): a student can't copy a teacher that sees hidden information. Phase 1 confirmed this (§6).

Never built or never used: potions (75% spent in hallway fights), HP-aware routing/rest (+30 oracle ceiling), learned shop/event/Neow/boss-relic/map policy (still policy_varE + heuristics), and the sim's native C++ MCTS (BattleScumSearcher2).

## 2. Decided rules (2026-11-06)
1. **Honest information:** search may sample hidden information but never read the seed's real future.
2. **Search at play time allowed:** the finished agent is network + honest search; a pure network is a stretch goal only.

## 3. Existing data
Old draft labels (draft_data*.jsonl, draft V nets), ds_train teacher runs and the qland shards were all labelled clairvoyantly. Use them only as priors or as a feature testbed, never as targets, unless relabelled honestly. policy_varN is fine as a prior. Keep the live CommunicationMod harness for validating the sim against the real game. spike_checkpoints.tar.gz can be discarded.

## 4. Phases (gate on fixed-150, confirmed on holdout-150; move to 500-seed evals once cheap)
- **Phase 0 — Reproduce (DONE):** WSL2 rebuild; mainline_v3 9/150 byte-identical; sim layer in git (`sim/`).
- **Phase 1 — Honest search (DONE):** copy + determinize bindings; honest combat and draft teacher; honest taxonomy (014).
- **Phase 2 — C++ search engine (DONE):** batched C++ rollouts, determinized MCTS, adaptive samples for bosses and elites, potion conservation. Gate: best honest engine vs the K=8 baseline (1/150, floor 22.41).
- **Phase 3 — Consistent Ironclad A0 wins:** expert iteration with search at play time. Honest K-sample macro search for every non-combat decision (cards, map, rest/smith, shop, events, Neow, boss relic, potions); learned prior + value nets trained on honest search outputs with a within-decision ranking loss, used to prune search and replace rollouts, then iterate; one run-level value that prices HP, gold, potions and deck together; attack the biggest taxonomy bucket first. Milestones: 25% → 50% → 75% → 85%+.
- **Phase 4 — Ironclad ascensions:** audit each level against the real game; ascension in the observation, one set of nets; curriculum by band (A1–5, A6–10, A11–15, A16–20) with a replay mix of lower levels. Targets: A10 70%+, A15 55%+, A20 40%+, then push (top humans win ~50–70% at A20 Ironclad).
- **Phase 5 — Silent:** the sim has no Silent card logic. Check for an existing fork first; otherwise implement ~75 cards, Silent relics and class events. Differential-test against the real game via CommunicationMod, with unit tests per card. Make the observation character-agnostic (card-ID embeddings + set encoder + character token). Then rerun Phases 3–4, warm-started from Ironclad.
- **Phase 6 — Defect and Watcher:** same recipe (orbs; stances/mantra/scry/retain).
- **Phase 7 — Completion:** Act 4 and the Heart; real-game validation per character × ascension; final win-rate table with CIs on held-out seeds.

## 5. Compute
Primary machine: ASUS ROG Zephyrus G15 (GA503QR) running 24/7: Ryzen 9 5900HS 8c/16t, ~40 GB RAM, RTX 3070 Laptop (CUDA in WSL2 for net training), 1 TB SSD; ~14 sim workers; lid action "do nothing". If more is needed later: Oracle Cloud Always Free ARM, Hetzner/OVH dedicated (~€40–80/month), or spot VMs for bursts.

## 6. Progress log
- **Phase 0 (2026-11-06):** mainline_v3 reproduces 9/150, floor 26.06; all 150 outputs byte-identical to the GCP originals.
- **Phase 1 (2026-11-06):** honest search drops mainline_v3 to 1/150 at K=8 (floor 22.41); the draft teacher drops from 15/50 to 0/50 on the same seeds.
- **Phase 2 (2026-11-06, gate passed):** C++ rollouts are identical to Python and ~11× faster in wall-clock. Chosen engine 1-ply K=64: eval 6/150, floor 24.61; holdout-150 7 wins / floor 26.67 vs honest K=8 3 / 23.10 (+3.57 floors, 95% CI +2.45..+4.75). Determinized MCTS and a potion-saving rule both lost to 1-ply. Throughput: ~45 min per 150 seeds at K=64 (12× K=8), so Phase 3 must make samples cheaper (value net, priors).

## 7. Open decisions
1. Definition of a win: Act 3 boss for the ladder phases, Heart for the final scoreboard (recommended).
2. Whether AGENTS.md's patch-only workflow applies to `sim/` work.
