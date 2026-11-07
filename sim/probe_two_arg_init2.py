"""Round 2: introspect MonsterGroup, then verify the two-arg init picks the
requested encounter by reading monster identities/HP."""
import slaythespire as sts

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc, sts.MonsterEncounter.GREMLIN_NOB)
mg = bc.monsters
print("MonsterGroup members:", [n for n in dir(mg) if not n.startswith("_")])
print("has __len__:", hasattr(mg, "__len__"), "has __getitem__:",
      hasattr(mg, "__getitem__"))


def describe(bc):
    mg = bc.monsters
    out = []
    # try every plausible access pattern
    if hasattr(mg, "__len__") and hasattr(mg, "__getitem__"):
        for i in range(len(mg)):
            m = mg[i]
            out.append("%s hp=%d/%d" % (m.id, m.cur_hp, m.max_hp))
        return out
    for attr in ("monsters", "monster_list", "list"):
        if hasattr(mg, attr):
            for m in getattr(mg, attr):
                out.append("%s hp=%d/%d" % (m.id, m.cur_hp, m.max_hp))
            return out
    for n in dir(mg):
        if n.startswith("_"):
            continue
        try:
            v = getattr(mg, n)
            out.append("%s=%s" % (n, str(v)[:80]))
        except Exception as e:
            out.append("%s=<err %s>" % (n, e))
    return out


for name in ("GREMLIN_NOB", "LAGAVULIN", "SLIME_BOSS", "THREE_SENTRIES"):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    bc = sts.BattleContext()
    bc.init(gc, getattr(sts.MonsterEncounter, name))
    print("%-14s -> %s" % (name, describe(bc)))

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc)
print("%-14s -> %s" % ("one-arg", describe(bc)))
