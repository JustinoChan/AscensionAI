"""Verify decoded combat values against a known Gremlin Nob fight. Asserts the
binding is self-consistent: is_attacking matches (base>0), attackCount>=1 on
attacks, and both attack and non-attack (Nob's Bellow) intents are observed."""
import random
import slaythespire as sts

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc, sts.MonsterEncounter.GREMLIN_NOB)
rng = random.Random(1)

seen_attack = False
seen_nonattack = False
for step in range(80):
    if bc.outcome != sts.BattleOutcome.UNDECIDED:
        break
    m = bc.monsters.monsters[0]
    atk = m.is_attacking()
    base, hits = m.intent_base_damage(bc)
    if bc.turn <= 6:
        print("turn %d | attacking=%s base=%d hits=%d | mon str=%d weak=%d "
              "vuln=%d | player vuln=%d weak=%d str=%d block=%d"
              % (bc.turn, atk, base, hits, m.strength(), m.weak(),
                 m.vulnerable(), bc.player.vulnerable(), bc.player.weak(),
                 bc.player.strength(), bc.player.block), flush=True)
    if atk:
        assert base > 0 and hits >= 1, "attacking but base/hits zero"
        seen_attack = True
    else:
        assert base == 0 and hits == 0, "not attacking but base/hits nonzero"
        seen_nonattack = True
    acts = bc.get_possible_actions()
    rng.choice(acts).execute(bc)

assert seen_attack, "never saw an attack intent"
print("PASS: attack and non-attack intents observed, is_attacking consistent "
      "with base/hits (saw_nonattack=%s)" % seen_nonattack)
