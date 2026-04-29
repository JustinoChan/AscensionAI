"""
game_data.py — STS1 game knowledge databases for the observation encoder.

Contains functional effect data for all Ironclad-relevant relics, potions,
and cards.  Imported by obs_encoder.py to enrich the observation vector.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ===================================================================
# CARD IDENTITY LIST
# ===================================================================
# Authoritative list of all card IDs the Ironclad agent can encounter.
# Order determines identity-embedding index (deterministic hash seed).
# Sourced from IroncladPriority.CARD_PRIORITY_LIST + extras.
CARD_ID_LIST: List[str] = [
    # --- Ironclad Starter ---
    "Strike_R", "Defend_R", "Bash",
    # --- Common Attacks ---
    "Anger", "Body Slam", "Clash", "Cleave", "Clothesline",
    "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike",
    "Pommel Strike", "Sword Boomerang", "Thunderclap", "Twin Strike",
    "Wild Strike",
    # --- Common Skills ---
    "Armaments", "Flex", "Havoc", "Shrug It Off", "True Grit", "Warcry",
    # --- Uncommon Attacks ---
    "Blood for Blood", "Carnage", "Dropkick", "Hemokinesis", "Pummel",
    "Rampage", "Reckless Charge", "Searing Blow", "Sever Soul",
    "Uppercut", "Whirlwind",
    # --- Uncommon Skills ---
    "Battle Trance", "Bloodletting", "Burning Pact", "Disarm",
    "Dual Wield", "Entrench", "Flame Barrier", "Ghostly Armor",
    "Infernal Blade", "Intimidate", "Offering", "Power Through",
    "Rage", "Second Wind", "Seeing Red", "Sentinel", "Shockwave",
    "Spot Weakness",
    # --- Uncommon Powers ---
    "Combust", "Corruption", "Dark Embrace", "Evolve",
    "Feel No Pain", "Fire Breathing", "Inflame", "Metallicize", "Rupture",
    # --- Rare Attacks ---
    "Bludgeon", "Feed", "Fiend Fire", "Immolate", "Reaper",
    # --- Rare Skills ---
    "Double Tap", "Exhume", "Impervious", "Limit Break",
    # --- Rare Powers ---
    "Barricade", "Berserk", "Brutality", "Demon Form", "Juggernaut",
    # --- Colorless ---
    "Apotheosis", "Bandage Up", "Bite", "Blind", "Dark Shackles",
    "Deep Breath", "Discovery", "Dramatic Entrance", "Enlightenment",
    "Finesse", "Flash of Steel", "Forethought", "Good Instincts",
    "Ghostly", "HandOfGreed", "Impatience", "J.A.X.", "Jack Of All Trades",
    "Madness", "Magnetism", "Master of Strategy", "Mayhem",
    "Metamorphosis", "Mind Blast", "Panacea", "Panache", "PanicButton",
    "Purity", "RitualDagger", "Sadistic Nature", "Secret Technique",
    "Secret Weapon", "Shiv", "Swift Strike", "The Bomb",
    "Thinking Ahead", "Transmutation", "Trip", "Violence",
    # --- Status ---
    "Wound", "Burn", "Dazed", "Slimed", "Void",
    # --- Curses ---
    "AscendersBane", "Clumsy", "Decay", "Doubt", "Injury",
    "Necronomicurse", "Normality", "Pain", "Parasite", "Pride",
    "Regret", "Shame", "Writhe",
]

# ===================================================================
# CARD MECHANICS
# ===================================================================
# card_id -> (draws, is_aoe, is_multi_hit, strength_scale, quality)
#   draws:           extra cards drawn when played
#   is_aoe:          hits all enemies
#   is_multi_hit:    hits multiple times in one play
#   strength_scale:  multiplier on strength for damage (0 = non-attack,
#                    1 = normal, 3 = Heavy Blade, 5 = Heavy Blade+)
#   quality:         subjective 0-10 card quality for deck profiling
CARD_MECHANICS: Dict[str, Tuple[int, bool, bool, float, float]] = {
    # Starter
    "Strike_R":        (0, False, False, 1.0, 1.0),
    "Defend_R":        (0, False, False, 0.0, 2.0),
    "Bash":            (0, False, False, 1.0, 3.0),
    # Common Attacks
    "Anger":           (0, False, False, 1.0, 4.0),
    "Body Slam":       (0, False, False, 0.0, 5.0),
    "Clash":           (0, False, False, 1.0, 3.0),
    "Cleave":          (0, True,  False, 1.0, 4.0),
    "Clothesline":     (0, False, False, 1.0, 3.5),
    "Headbutt":        (0, False, False, 1.0, 4.0),
    "Heavy Blade":     (0, False, False, 3.0, 5.5),
    "Iron Wave":       (0, False, False, 1.0, 3.5),
    "Perfected Strike":(0, False, False, 1.0, 6.0),
    "Pommel Strike":   (1, False, False, 1.0, 5.0),
    "Sword Boomerang": (0, False, True,  1.0, 3.5),
    "Thunderclap":     (0, True,  False, 1.0, 5.0),
    "Twin Strike":     (0, False, True,  1.0, 4.0),
    "Wild Strike":     (0, False, False, 1.0, 2.5),
    # Common Skills
    "Armaments":       (0, False, False, 0.0, 3.5),
    "Flex":            (0, False, False, 0.0, 3.0),
    "Havoc":           (0, False, False, 0.0, 3.0),
    "Shrug It Off":    (1, False, False, 0.0, 6.0),
    "True Grit":       (0, False, False, 0.0, 4.0),
    "Warcry":          (2, False, False, 0.0, 3.5),
    # Uncommon Attacks
    "Blood for Blood": (0, False, False, 1.0, 3.5),
    "Carnage":         (0, False, False, 1.0, 4.0),
    "Dropkick":        (1, False, False, 1.0, 4.5),
    "Hemokinesis":     (0, False, False, 1.0, 3.5),
    "Pummel":          (0, False, True,  1.0, 4.0),
    "Rampage":         (0, False, False, 1.0, 3.0),
    "Reckless Charge": (0, False, False, 1.0, 3.0),
    "Searing Blow":    (0, False, False, 1.0, 3.5),
    "Sever Soul":      (0, False, False, 1.0, 3.5),
    "Uppercut":        (0, False, False, 1.0, 5.0),
    "Whirlwind":       (0, True,  True,  1.0, 7.0),
    # Uncommon Skills
    "Battle Trance":   (3, False, False, 0.0, 7.0),
    "Bloodletting":    (0, False, False, 0.0, 3.5),
    "Burning Pact":    (2, False, False, 0.0, 4.5),
    "Disarm":          (0, False, False, 0.0, 5.5),
    "Dual Wield":      (0, False, False, 0.0, 3.5),
    "Entrench":        (0, False, False, 0.0, 3.5),
    "Flame Barrier":   (0, False, False, 0.0, 5.0),
    "Ghostly Armor":   (0, False, False, 0.0, 4.0),
    "Infernal Blade":  (0, False, False, 0.0, 3.5),
    "Intimidate":      (0, True,  False, 0.0, 3.5),
    "Offering":        (3, False, False, 0.0, 8.0),
    "Power Through":   (0, False, False, 0.0, 4.0),
    "Rage":            (0, False, False, 0.0, 5.5),
    "Second Wind":     (0, False, False, 0.0, 4.0),
    "Seeing Red":      (0, False, False, 0.0, 4.0),
    "Sentinel":        (0, False, False, 0.0, 3.0),
    "Shockwave":       (0, True,  False, 0.0, 5.5),
    "Spot Weakness":   (0, False, False, 0.0, 4.5),
    # Uncommon Powers
    "Combust":         (0, False, False, 0.0, 4.0),
    "Corruption":      (0, False, False, 0.0, 6.0),
    "Dark Embrace":    (0, False, False, 0.0, 4.5),
    "Evolve":          (0, False, False, 0.0, 4.0),
    "Feel No Pain":    (0, False, False, 0.0, 5.5),
    "Fire Breathing":  (0, True,  False, 0.0, 3.5),
    "Inflame":         (0, False, False, 0.0, 6.0),
    "Metallicize":     (0, False, False, 0.0, 5.5),
    "Rupture":         (0, False, False, 0.0, 3.5),
    # Rare Attacks
    "Bludgeon":        (0, False, False, 1.0, 5.0),
    "Feed":            (0, False, False, 1.0, 5.5),
    "Fiend Fire":      (0, False, False, 1.0, 5.0),
    "Immolate":        (0, True,  False, 1.0, 7.0),
    "Reaper":          (0, True,  False, 1.0, 6.0),
    # Rare Skills
    "Double Tap":      (0, False, False, 0.0, 6.0),
    "Exhume":          (0, False, False, 0.0, 4.5),
    "Impervious":      (0, False, False, 0.0, 8.0),
    "Limit Break":     (0, False, False, 0.0, 7.5),
    # Rare Powers
    "Barricade":       (0, False, False, 0.0, 7.0),
    "Berserk":         (0, False, False, 0.0, 5.0),
    "Brutality":       (0, False, False, 0.0, 4.0),
    "Demon Form":      (0, False, False, 0.0, 8.5),
    "Juggernaut":      (0, False, False, 0.0, 4.0),
    # Colorless
    "Apotheosis":      (0, False, False, 0.0, 9.0),
    "Bandage Up":      (0, False, False, 0.0, 2.0),
    "Bite":            (0, False, False, 1.0, 3.0),
    "Blind":           (0, True,  False, 0.0, 3.0),
    "Dark Shackles":   (0, False, False, 0.0, 4.0),
    "Deep Breath":     (1, False, False, 0.0, 2.0),
    "Discovery":       (0, False, False, 0.0, 3.0),
    "Dramatic Entrance":(0, True, False, 1.0, 2.5),
    "Enlightenment":   (0, False, False, 0.0, 3.5),
    "Finesse":         (1, False, False, 0.0, 3.5),
    "Flash of Steel":  (1, False, False, 1.0, 4.0),
    "Forethought":     (0, False, False, 0.0, 2.0),
    "Good Instincts":  (0, False, False, 0.0, 3.0),
    "Ghostly":         (0, False, False, 0.0, 7.0),
    "HandOfGreed":     (0, False, False, 1.0, 2.5),
    "Impatience":      (0, False, False, 0.0, 2.5),
    "J.A.X.":          (0, False, False, 0.0, 4.0),
    "Jack Of All Trades": (0, False, False, 0.0, 2.5),
    "Madness":         (0, False, False, 0.0, 4.0),
    "Magnetism":       (0, False, False, 0.0, 2.0),
    "Master of Strategy": (0, False, False, 0.0, 5.0),
    "Mayhem":          (0, False, False, 0.0, 3.0),
    "Metamorphosis":   (0, False, False, 0.0, 3.0),
    "Mind Blast":      (0, False, False, 0.0, 2.5),
    "Panacea":         (0, False, False, 0.0, 3.5),
    "Panache":         (0, False, False, 0.0, 3.5),
    "PanicButton":     (0, False, False, 0.0, 5.0),
    "Purity":          (0, False, False, 0.0, 3.0),
    "RitualDagger":    (0, False, False, 1.0, 2.0),
    "Sadistic Nature": (0, False, False, 0.0, 3.0),
    "Secret Technique":(0, False, False, 0.0, 3.5),
    "Secret Weapon":   (0, False, False, 0.0, 3.5),
    "Shiv":            (0, False, False, 1.0, 2.0),
    "Swift Strike":    (0, False, False, 1.0, 2.0),
    "The Bomb":        (0, False, False, 0.0, 3.5),
    "Thinking Ahead":  (1, False, False, 0.0, 3.0),
    "Transmutation":   (0, False, False, 0.0, 2.0),
    "Trip":            (0, False, False, 0.0, 3.5),
    "Violence":        (0, False, False, 0.0, 3.0),
    # Status
    "Wound":           (0, False, False, 0.0, -1.0),
    "Burn":            (0, False, False, 0.0, -2.0),
    "Dazed":           (0, False, False, 0.0, -1.5),
    "Slimed":          (0, False, False, 0.0, -1.0),
    "Void":            (0, False, False, 0.0, -3.0),
    # Curses
    "AscendersBane":   (0, False, False, 0.0, -1.0),
    "Clumsy":          (0, False, False, 0.0, -1.0),
    "Decay":           (0, False, False, 0.0, -3.0),
    "Doubt":           (0, False, False, 0.0, -2.0),
    "Injury":          (0, False, False, 0.0, -1.0),
    "Necronomicurse":  (0, False, False, 0.0, -3.0),
    "Normality":       (0, False, False, 0.0, -4.0),
    "Pain":            (0, False, False, 0.0, -3.0),
    "Parasite":        (0, False, False, 0.0, -2.0),
    "Pride":           (0, False, False, 0.0, -1.5),
    "Regret":          (0, False, False, 0.0, -2.0),
    "Shame":           (0, False, False, 0.0, -2.0),
    "Writhe":          (0, False, False, 0.0, -2.0),
}


# ===================================================================
# RELIC DATABASE
# ===================================================================
# Aggregate feature indices for the 25-dim relic observation vector.
# When encoding, we iterate over all held relics and sum contributions.
R_ENERGY        = 0   # extra energy per turn
R_STRENGTH      = 1   # passive strength bonus
R_DEXTERITY     = 2   # passive dexterity bonus
R_MAX_HP        = 3   # max HP bonus (signal, not duplicate of player HP)
R_HEAL_COMBAT   = 4   # HP healed at end of combat
R_CARD_DRAW     = 5   # extra cards drawn per turn
R_BLOCK_RETAIN  = 6   # block retention amount
R_NO_INTENTS    = 7   # can't see enemy intents (binary)
R_VULN_BONUS    = 8   # vulnerability deals extra damage
R_WEAK_IMMUNE   = 9   # immune to weak (binary)
R_THORNS        = 10  # thorns damage when hit
R_PLATED_ARMOR  = 11  # plated armor / orichalcum
R_DMG_REDUCE    = 12  # flat damage reduction
R_NO_POTIONS    = 13  # potions disabled (binary)
R_EXHAUST_SYN   = 14  # exhaust synergy (Dead Branch etc.)
R_PLAY_LIMIT    = 15  # card play limit per turn
R_COST_RANDOM   = 16  # card costs randomized (binary)
R_HAND_RETAIN   = 17  # hand doesn't discard (binary)
R_ARTIFACT      = 18  # artifact at combat start
R_STR_SCALING   = 19  # gains strength over time
R_ENEMY_STR     = 20  # enemies gain strength
R_FIRST_TURN    = 21  # first-turn bonuses (energy/draw/block)
R_NO_HEAL       = 22  # healing disabled (binary)
R_RELIC_COUNT   = 23  # total relics held
R_RELIC_VALUE   = 24  # sum of quality scores

RELIC_FEATURE_DIM = 25

# Normalization divisors for each relic feature
RELIC_FEATURE_NORMS = [
    4.0,   # energy: max ~3-4
    5.0,   # strength
    5.0,   # dexterity
    50.0,  # max_hp
    20.0,  # heal_combat
    5.0,   # card_draw
    20.0,  # block_retain
    1.0,   # no_intents (binary)
    1.0,   # vuln_bonus
    1.0,   # weak_immune (binary)
    5.0,   # thorns
    10.0,  # plated_armor
    3.0,   # dmg_reduce
    1.0,   # no_potions (binary)
    3.0,   # exhaust_syn
    10.0,  # play_limit
    1.0,   # cost_random (binary)
    1.0,   # hand_retain (binary)
    3.0,   # artifact
    3.0,   # str_scaling
    3.0,   # enemy_str
    5.0,   # first_turn
    1.0,   # no_heal (binary)
    20.0,  # relic_count
    50.0,  # relic_value
]

# RELIC_DB: relic_id -> (quality_score, {feature_idx: raw_value})
# Only relics with combat-relevant effects have feature entries.
# All relics contribute to R_RELIC_COUNT and R_RELIC_VALUE.
RELIC_DB: Dict[str, Tuple[float, Dict[int, float]]] = {
    # === Ironclad Character Relics ===
    "Burning Blood":         (3.0, {R_HEAL_COMBAT: 6.0}),
    "Black Blood":           (4.5, {R_HEAL_COMBAT: 12.0}),

    # === Common Relics ===
    "Akabeko":               (2.5, {}),
    "Anchor":                (2.5, {R_PLATED_ARMOR: 10.0}),
    "Ancient Tea Set":       (2.0, {R_FIRST_TURN: 0.5}),
    "Art of War":            (2.0, {}),
    "Bag of Marbles":        (2.5, {R_VULN_BONUS: 0.3}),
    "Bag of Preparation":    (3.0, {R_FIRST_TURN: 2.0}),
    "Blood Vial":            (1.5, {R_HEAL_COMBAT: 2.0}),
    "Bronze Scales":         (2.5, {R_THORNS: 3.0}),
    "Centennial Puzzle":     (2.0, {R_FIRST_TURN: 1.0}),
    "Ceramic Fish":          (1.5, {}),
    "Dream Catcher":         (2.0, {}),
    "Happy Flower":          (2.5, {R_ENERGY: 0.33}),
    "Juzu Bracelet":         (2.0, {}),
    "Lantern":               (2.5, {R_FIRST_TURN: 1.0}),
    "Maw Bank":              (1.0, {}),
    "Meal Ticket":           (1.5, {}),
    "Nunchaku":              (2.0, {R_ENERGY: 0.1}),
    "Oddly Smooth Stone":    (3.0, {R_DEXTERITY: 1.0}),
    "Omamori":               (2.5, {}),
    "Orichalcum":            (3.0, {R_PLATED_ARMOR: 6.0}),
    "Pen Nib":               (2.5, {}),
    "Potion Belt":           (2.0, {}),
    "Preserved Insect":      (2.0, {}),
    "Regal Pillow":          (2.0, {}),
    "Smiling Mask":          (1.0, {}),
    "Strawberry":            (2.0, {R_MAX_HP: 7.0}),
    "The Boot":              (1.5, {}),
    "Tiny Chest":            (1.5, {}),
    "Toy Ornithopter":       (1.5, {}),
    "Vajra":                 (3.5, {R_STRENGTH: 1.0}),
    "War Paint":             (2.0, {}),
    "Whetstone":             (2.0, {}),

    # === Uncommon Relics ===
    "Blue Candle":           (2.0, {}),
    "Bottled Flame":         (2.5, {}),
    "Bottled Lightning":     (2.5, {}),
    "Bottled Tornado":       (2.5, {}),
    "Darkstone Periapt":     (2.0, {R_MAX_HP: 6.0}),
    "Eternal Feather":       (2.0, {}),
    "Frozen Egg":            (3.0, {}),
    "Frozen Eye":            (1.0, {}),
    "Gambling Chip":         (2.0, {}),
    "Gremlin Horn":          (3.0, {R_ENERGY: 0.2}),
    "Horn Cleat":            (2.5, {R_FIRST_TURN: 1.0}),
    "Ink Bottle":            (2.5, {R_CARD_DRAW: 0.1}),
    "Kunai":                 (3.5, {R_DEXTERITY: 0.33, R_STR_SCALING: 0.3}),
    "Letter Opener":         (2.0, {}),
    "Matryoshka":            (1.5, {}),
    "Meat on the Bone":      (2.5, {R_HEAL_COMBAT: 6.0}),
    "Mercury Hourglass":     (2.0, {}),
    "Molten Egg":            (3.0, {}),
    "Mummified Hand":        (3.0, {R_ENERGY: 0.2}),
    "Ornamental Fan":        (2.5, {}),
    "Pantograph":            (2.0, {}),
    "Paper Krane":           (3.0, {R_DMG_REDUCE: 0.5}),
    "Paper Phrog":           (3.0, {R_VULN_BONUS: 0.5}),
    "Pear":                  (2.0, {R_MAX_HP: 10.0}),
    "Question Card":         (2.0, {}),
    "Shuriken":              (3.5, {R_STRENGTH: 0.33, R_STR_SCALING: 0.5}),
    "Singing Bowl":          (1.5, {}),
    "Strike Dummy":          (1.0, {}),
    "Sundial":               (2.0, {R_ENERGY: 0.1}),
    "The Courier":           (2.0, {}),
    "Toxic Egg":             (3.0, {}),

    # === Rare Relics ===
    "Bird-Faced Urn":        (2.5, {R_HEAL_COMBAT: 2.0}),
    "Calipers":              (4.0, {R_BLOCK_RETAIN: 15.0}),
    "Captain's Wheel":       (2.5, {}),
    "Champion Belt":         (3.0, {}),
    "Charon's Ashes":        (3.0, {R_EXHAUST_SYN: 1.0}),
    "Clockwork Souvenir":    (3.0, {R_ARTIFACT: 1.0}),
    "Dead Branch":           (4.5, {R_EXHAUST_SYN: 2.0}),
    "Du-Vu Doll":            (2.0, {R_STRENGTH: 0.5}),
    "Emotion Chip":          (2.0, {}),
    "Fossilized Helix":      (3.0, {}),
    "Ginger":                (3.5, {R_WEAK_IMMUNE: 1.0}),
    "Girya":                 (2.5, {R_STRENGTH: 1.0}),
    "Golden Eye":            (1.5, {}),
    "Ice Cream":             (4.0, {R_ENERGY: 0.5}),
    "Incense Burner":        (4.0, {}),
    "Lizard Tail":           (3.5, {}),
    "Magic Flower":          (2.5, {}),
    "Mango":                 (2.5, {R_MAX_HP: 14.0}),
    "Old Coin":              (2.0, {}),
    "Peace Pipe":            (2.5, {}),
    "Pocketwatch":           (3.0, {R_CARD_DRAW: 0.5}),
    "Prayer Wheel":          (2.5, {}),
    "Shovel":                (2.0, {}),
    "Stone Calendar":        (2.0, {}),
    "Thread and Needle":     (3.0, {R_PLATED_ARMOR: 4.0}),
    "Tingsha":               (2.5, {R_EXHAUST_SYN: 0.5}),
    "Torii":                 (3.5, {R_DMG_REDUCE: 1.0}),
    "Tough Bandages":        (2.5, {}),
    "Tungsten Rod":          (3.5, {R_DMG_REDUCE: 1.0}),
    "Turnip":                (2.0, {}),
    "Unceasing Top":         (3.0, {R_CARD_DRAW: 0.5}),
    "Wing Boots":            (2.0, {}),
    "Self-Forming Clay":     (2.5, {}),
    "Medical Kit":           (2.0, {}),
    "Orange Pellets":        (3.5, {}),
    "Nilry's Codex":         (3.0, {}),
    "Red Skull":             (2.5, {R_STRENGTH: 1.5}),
    "Necronomicon":          (4.0, {}),
    "Chemical X":            (3.5, {}),
    "Lee's Waffle":          (3.5, {R_MAX_HP: 7.0, R_HEAL_COMBAT: 10.0}),
    "Sling of Courage":      (2.5, {R_STRENGTH: 2.0}),

    # === Boss Relics ===
    "Astrolabe":             (3.5, {}),
    "Black Star":            (3.0, {}),
    "Busted Crown":          (3.0, {R_ENERGY: 1.0}),
    "Calling Bell":          (2.0, {}),
    "Coffee Dripper":        (3.0, {R_ENERGY: 1.0}),
    "Cursed Key":            (3.5, {R_ENERGY: 1.0}),
    "Ectoplasm":             (2.5, {R_ENERGY: 1.0}),
    "Empty Cage":            (3.0, {}),
    "Fusion Hammer":         (3.0, {R_ENERGY: 1.0}),
    "Mark of Pain":          (3.0, {R_ENERGY: 1.0}),
    "Orrery":                (2.5, {}),
    "Pandora's Box":         (3.0, {}),
    "Philosopher's Stone":   (3.0, {R_ENERGY: 1.0, R_ENEMY_STR: 1.0}),
    "Runic Cube":            (2.5, {R_CARD_DRAW: 0.3}),
    "Runic Dome":            (3.5, {R_ENERGY: 1.0, R_NO_INTENTS: 1.0}),
    "Runic Pyramid":         (4.0, {R_HAND_RETAIN: 1.0}),
    "Sacred Bark":           (3.0, {}),
    "Snecko Eye":            (4.0, {R_CARD_DRAW: 2.0, R_COST_RANDOM: 1.0}),
    "Sozu":                  (3.0, {R_ENERGY: 1.0, R_NO_POTIONS: 1.0}),
    "Tiny House":            (2.0, {}),
    "Velvet Choker":         (2.5, {R_ENERGY: 1.0, R_PLAY_LIMIT: 6.0}),
    "White Beast Statue":    (2.0, {}),

    # === Event / Special Relics ===
    "Bloody Idol":           (1.5, {}),
    "Cultist Mask":          (0.5, {}),
    "Enchiridion":           (2.5, {R_FIRST_TURN: 1.0}),
    "Face Of Cleric":        (1.5, {R_MAX_HP: 1.0}),
    "Golden Idol":           (2.0, {}),
    "Mark of the Bloom":     (-1.0, {R_NO_HEAL: 1.0}),
    "Mutagenic Strength":    (2.0, {R_STRENGTH: 3.0}),
    "Nloth's Gift":          (1.0, {}),
    "Red Mask":              (2.0, {}),
    "Spirit Poop":           (0.0, {}),
    "Ssserpent Ring":        (1.5, {}),
    "Warped Tongs":          (2.5, {}),
    "Brimstone":             (3.0, {R_STRENGTH: 2.0, R_ENEMY_STR: 1.0}),
    "Circlet":               (0.5, {}),
    "Dollys Mirror":         (2.0, {}),
    "Membership Card":       (2.0, {}),
    "Strange Spoon":         (2.0, {}),
    "The Abacus":            (2.0, {}),
    "Toolbox":               (2.0, {}),
    "Cauldron":              (2.0, {}),
    "Prismatic Shard":       (2.0, {}),
}


# ===================================================================
# POTION DATABASE
# ===================================================================
# potion_id -> (deals_damage, gives_block, gives_strength, gives_dex, heals)
# These 5 flags describe what the potion does functionally.
# The encoder adds: present, can_use, requires_target from the potion object.
POTION_EFFECTS: Dict[str, Tuple[int, int, int, int, int]] = {
    "Potion Slot":           (0, 0, 0, 0, 0),
    "Attack Potion":         (0, 0, 0, 0, 0),
    "Block Potion":          (0, 1, 0, 0, 0),
    "Blood Potion":          (0, 0, 0, 0, 1),
    "Colorless Potion":      (0, 0, 0, 0, 0),
    "Cultist Potion":        (0, 0, 0, 0, 0),
    "Dexterity Potion":      (0, 0, 0, 1, 0),
    "Distilled Chaos":       (0, 0, 0, 0, 0),
    "Duplication Potion":    (0, 0, 0, 0, 0),
    "Elixir":                (0, 0, 0, 0, 0),
    "Energy Potion":         (0, 0, 0, 0, 0),
    "Entropic Brew":         (0, 0, 0, 0, 0),
    "Essence of Darkness":   (0, 0, 0, 0, 0),
    "Essence of Steel":      (0, 1, 0, 0, 0),
    "Explosive Potion":      (1, 0, 0, 0, 0),
    "Fairy in a Bottle":     (0, 0, 0, 0, 1),
    "Fear Potion":           (0, 0, 0, 0, 0),
    "Fire Potion":           (1, 0, 0, 0, 0),
    "Flex Potion":           (0, 0, 1, 0, 0),
    "Focus Potion":          (0, 0, 0, 0, 0),
    "Fruit Juice":           (0, 0, 0, 0, 1),
    "Gambler's Brew":        (0, 0, 0, 0, 0),
    "Ghost In A Jar":        (0, 0, 0, 0, 0),
    "Heart of Iron":         (0, 1, 0, 0, 0),
    "Liquid Bronze":         (0, 0, 0, 0, 0),
    "Liquid Memories":       (0, 0, 0, 0, 0),
    "Ancient Potion":        (0, 0, 0, 0, 0),
    "Poison Potion":         (1, 0, 0, 0, 0),
    "Power Potion":          (0, 0, 0, 0, 0),
    "Regen Potion":          (0, 0, 0, 0, 1),
    "Skill Potion":          (0, 0, 0, 0, 0),
    "Smoke Bomb":            (0, 0, 0, 0, 0),
    "Snecko Oil":            (0, 0, 0, 0, 0),
    "Speed Potion":          (0, 0, 0, 1, 0),
    "Stance Potion":         (0, 0, 0, 0, 0),
    "Strength Potion":       (0, 0, 1, 0, 0),
    "Swift Potion":          (0, 0, 0, 0, 0),
    "Weak Potion":           (0, 0, 0, 0, 0),
    "BlessingOfTheForge":    (0, 0, 0, 0, 0),
    "Ambrosia":              (0, 0, 0, 0, 0),
}
