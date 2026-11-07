"""Verify the two-arg init binding picks the requested encounter and the
one-arg form still works. No gym import, no getNNInterface call."""
import slaythespire as sts


def describe(bc):
    mg = bc.monsters
    try:
        return [str(m.id) for m in mg]
    except TypeError:
        return repr(mg)


for name in ("GREMLIN_NOB", "LAGAVULIN", "SLIME_BOSS"):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    bc = sts.BattleContext()
    bc.init(gc, getattr(sts.MonsterEncounter, name))
    print("%-12s -> %s" % (name, describe(bc)))

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc)
print("one-arg      ->", describe(bc))
print("PASS criteria: three different monster groups matching the names,")
print("and the one-arg line shows the seed's natural first encounter.")
