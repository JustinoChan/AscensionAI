"""Pragmatic Ironclad card-role knowledge base for the Act-2-entry deck
diagnostic. Keys = CardId enum name (str(card.id).split('.')[-1]). Not a perfect
tier list -- just enough role tags to answer: is the Act-2 deck MISSING roles, or
does it have the roles but at low power? Cards not listed fall back to type."""

STARTERS = {"STRIKE_RED", "DEFEND_RED", "BASH"}

AOE = {"CLEAVE", "WHIRLWIND", "THUNDERCLAP", "IMMOLATE", "REAPER",
       "SWORD_BOOMERANG", "SHOCKWAVE"}

BLOCK = {"DEFEND_RED", "SHRUG_IT_OFF", "IRON_WAVE", "GHOSTLY_ARMOR", "TRUE_GRIT",
         "POWER_THROUGH", "IMPERVIOUS", "ENTRENCH", "SECOND_WIND", "FLAME_BARRIER",
         "SENTINEL", "DEFEND_BLUE", "DEFEND_GREEN", "DEFEND_PURPLE", "BLOODLETTING"}

# scaling = repeatable damage/block growth (most are POWER type; some are engines)
SCALING = {"DEMON_FORM", "INFLAME", "LIMIT_BREAK", "SPOT_WEAKNESS", "RUPTURE",
           "FEEL_NO_PAIN", "DARK_EMBRACE", "COMBUST", "FIRE_BREATHING", "EVOLVE",
           "METALLICIZE", "BARRICADE", "JUGGERNAUT", "BRUTALITY", "CORRUPTION",
           "BERSERK", "MAYHEM", "MAGNETISM", "PANACHE"}

DRAW = {"BATTLE_TRANCE", "POMMEL_STRIKE", "WARCRY", "OFFERING", "BURNING_PACT",
        "TRUE_GRIT", "SHRUG_IT_OFF"}  # cards that draw as a side effect

ENERGY = {"SEEING_RED", "BLOODLETTING", "OFFERING"}

# strong frontload / burst single-target attacks worth boosting
FRONTLOAD = {"BASH", "HEAVY_BLADE", "UPPERCUT", "CLOTHESLINE", "POMMEL_STRIKE",
             "TWIN_STRIKE", "IRON_WAVE", "HEMOKINESIS", "CARNAGE", "DROPKICK",
             "PUMMEL", "BLOOD_FOR_BLOOD", "RAMPAGE", "REAPER", "BLUDGEON",
             "FIEND_FIRE", "FEED", "WHIRLWIND", "SEVER_SOUL"}

JUNK = {"ASCENDERS_BANE", "CLUMSY", "DAZED", "BURN", "WOUND", "SLIMED", "VOID",
        "INJURY", "NORMALITY", "PAIN", "PARASITE", "REGRET", "SHAME", "DOUBT",
        "WRITHE", "DECAY", "CURSE_OF_THE_BELL", "NECRONOMICURSE", "PRIDE",
        "CLASH", "WILD_STRIKE", "RECKLESS_CHARGE"}  # last 3: situational/conditional


def classify(name, ctype):
    """Return a set of role tags. ctype is the CardType name (ATTACK/SKILL/POWER/
    CURSE/STATUS)."""
    roles = set()
    if name in STARTERS:
        roles.add("starter")
    if ctype in ("CURSE", "STATUS") or name in JUNK:
        roles.add("junk")
    if name in AOE:
        roles.add("aoe")
    if name in BLOCK:
        roles.add("block")
    if name in SCALING or ctype == "POWER":
        roles.add("scaling")
    if name in DRAW:
        roles.add("draw")
    if name in ENERGY:
        roles.add("energy")
    if name in FRONTLOAD:
        roles.add("frontload")
    elif ctype == "ATTACK" and "aoe" not in roles:
        roles.add("frontload")   # any other attack = baseline single-target damage
    return roles
