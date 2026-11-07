"""Round 3: verify two-arg init via monster count + HP (m.id is an
unregistered type in the bindings; HP identifies the encounter fine:
Gremlin Nob 82-86, Lagavulin 109-111, Slime Boss 140, Sentries 38-45 x3)."""
import slaythespire as sts


def describe(bc):
    mg = bc.monsters
    hps = []
    for m in mg.monsters:
        if m.max_hp > 0:
            hps.append("%d/%d" % (m.cur_hp, m.max_hp))
    return "count=%d alive=%d hp=[%s]" % (
        mg.monster_count, mg.monsters_alive, ", ".join(hps))


for name in ("GREMLIN_NOB", "LAGAVULIN", "SLIME_BOSS", "THREE_SENTRIES"):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    bc = sts.BattleContext()
    bc.init(gc, getattr(sts.MonsterEncounter, name))
    print("%-14s -> %s" % (name, describe(bc)))

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc)
print("%-14s -> %s" % ("one-arg", describe(bc)))
