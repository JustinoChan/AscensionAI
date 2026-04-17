"""
obs_encoder.py — Rich observation encoder for Slay the Spire RL.

Converts a SpireComm Game object into a fixed-size numeric vector that
captures what a human player actually sees: cards in hand, monster identity
and behavior, intents, buffs/debuffs, player resources, and screen context.

Designed for Ironclad; extend CARD_STATS for other characters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Monster knowledge base — every STS1 monster with behavioral metadata.
# Sourced from spire-archive.com API (66 monsters).
#
# Fields per monster:
#   type   — 0=Normal, 1=Elite, 2=Boss
#   act    — 0=exordium(Act1), 1=city(Act2), 2=beyond(Act3), 3=ending(Act4)
#   n_moves — number of distinct moves
#   flags  — 6 behavioral booleans:
#       [0] enrages_on_skill  (Nob-like: punishes skill cards)
#       [1] splits_low_hp     (slime boss, large slimes split at ~50% HP)
#       [2] scales_strength   (Cultist ritual, Demon Form-like buff loops)
#       [3] multi_attacker    (has a move that hits 2+ times)
#       [4] retaliates        (thorns, curl-up, reactive damage)
#       [5] escapes           (gremlin escape, looter flee)
# ---------------------------------------------------------------------------
MONSTER_DB: Dict[str, Tuple[int, int, int, Tuple[int, ...]]] = {
    # --- Act 1 Normal ---
    "AcidSlime_L":     (0, 0, 4, (0, 1, 0, 0, 0, 0)),
    "AcidSlime_M":     (0, 0, 3, (0, 0, 0, 0, 0, 0)),
    "AcidSlime_S":     (0, 0, 2, (0, 0, 0, 0, 0, 0)),
    "SpikeSlime_L":    (0, 0, 3, (0, 1, 0, 0, 0, 0)),
    "SpikeSlime_M":    (0, 0, 2, (0, 0, 0, 0, 0, 0)),
    "SpikeSlime_S":    (0, 0, 1, (0, 0, 0, 0, 0, 0)),
    "JawWorm":         (0, 0, 3, (0, 0, 1, 0, 0, 0)),
    "Cultist":         (0, 0, 2, (0, 0, 1, 0, 0, 0)),
    "FungiBeast":      (0, 0, 2, (0, 0, 1, 0, 0, 0)),
    "GremlinFat":      (0, 0, 2, (0, 0, 0, 0, 0, 1)),
    "GremlinWarrior":  (0, 0, 2, (0, 0, 0, 0, 0, 1)),
    "GremlinThief":    (0, 0, 2, (0, 0, 0, 0, 0, 1)),
    "GremlinTsundere": (0, 0, 3, (0, 0, 0, 0, 0, 1)),
    "GremlinWizard":   (0, 0, 3, (0, 0, 0, 0, 0, 1)),
    "FuzzyLouseNormal":(0, 0, 2, (0, 0, 1, 0, 1, 0)),
    "FuzzyLouseDefensive":(0, 0, 2, (0, 0, 0, 0, 1, 0)),
    "SlaverBlue":      (0, 0, 2, (0, 0, 0, 0, 0, 0)),
    "SlaverRed":       (0, 0, 3, (0, 0, 0, 0, 0, 0)),
    "Looter":          (0, 0, 2, (0, 0, 0, 0, 0, 1)),
    "Apology Slime":   (0, 0, 2, (0, 0, 0, 0, 0, 0)),
    # --- Act 1 Elite ---
    "GremlinNob":      (1, 0, 3, (1, 0, 1, 0, 0, 0)),
    "Lagavulin":       (1, 0, 4, (0, 0, 0, 0, 0, 0)),
    "Sentry":          (1, 0, 2, (0, 0, 0, 0, 0, 0)),
    # --- Act 1 Boss ---
    "Hexaghost":       (2, 0, 6, (0, 0, 0, 1, 0, 0)),
    "TheGuardian":     (2, 0, 7, (0, 0, 0, 1, 1, 0)),
    "SlimeBoss":       (2, 0, 4, (0, 1, 0, 0, 0, 0)),
    # --- Act 2 Normal ---
    "Chosen":          (0, 1, 5, (0, 0, 0, 1, 0, 0)),
    "Byrd":            (0, 1, 6, (0, 0, 0, 1, 0, 0)),
    "ShelledParasite":  (0, 1, 4, (0, 0, 0, 1, 1, 0)),
    "SnakePlant":      (0, 1, 2, (0, 0, 0, 1, 0, 0)),
    "Centurion":       (0, 1, 3, (0, 0, 0, 1, 0, 0)),
    "Healer":          (0, 1, 3, (0, 0, 0, 0, 0, 0)),
    "Snecko":          (0, 1, 3, (0, 0, 0, 0, 0, 0)),
    "Mugger":          (0, 1, 2, (0, 0, 0, 0, 0, 1)),
    "SphericGuardian": (0, 1, 4, (0, 0, 0, 1, 0, 0)),
    "BanditBear":      (0, 1, 1, (0, 0, 0, 0, 0, 0)),
    "BanditChild":     (0, 1, 1, (0, 0, 0, 1, 0, 0)),
    "BanditLeader":    (0, 1, 1, (0, 0, 0, 0, 0, 0)),
    "TorchHead":       (0, 1, 1, (0, 0, 0, 0, 0, 0)),
    # --- Act 2 Elite ---
    "GremlinLeader":   (1, 1, 3, (0, 0, 0, 1, 0, 0)),
    "SlaverBoss":      (1, 1, 1, (0, 0, 0, 0, 0, 0)),
    "BookOfStabbing":  (1, 1, 2, (0, 0, 1, 0, 0, 0)),
    # --- Act 2 Boss ---
    "BronzeAutomaton": (2, 1, 5, (0, 0, 0, 1, 0, 0)),
    "TheCollector":    (2, 1, 5, (0, 0, 0, 0, 0, 0)),
    "Champ":           (2, 1, 7, (0, 0, 1, 1, 0, 0)),
    # --- Act 3 Normal ---
    "Darkling":        (0, 2, 5, (0, 0, 0, 1, 0, 0)),
    "OrbWalker":       (0, 2, 2, (0, 0, 0, 0, 0, 0)),
    "Spiker":          (0, 2, 2, (0, 0, 0, 0, 1, 0)),
    "Repulsor":        (0, 2, 2, (0, 0, 0, 0, 0, 0)),
    "Exploder":        (0, 2, 2, (0, 0, 0, 0, 0, 0)),
    "Maw":             (0, 2, 4, (0, 0, 1, 0, 0, 0)),
    "Serpent":         (0, 2, 3, (0, 0, 0, 0, 0, 0)),
    "Dagger":          (0, 2, 2, (0, 0, 0, 0, 0, 0)),
    # --- Act 3 Elite ---
    "GiantHead":       (1, 2, 3, (0, 0, 1, 0, 0, 0)),
    "Nemesis":         (1, 2, 3, (0, 0, 0, 1, 0, 0)),
    "Reptomancer":     (1, 2, 3, (0, 0, 0, 1, 0, 0)),
    "Transient":       (1, 2, 1, (0, 0, 1, 0, 0, 0)),
    "WrithingMass":    (1, 2, 5, (0, 0, 0, 1, 0, 0)),
    # --- Act 3 Boss ---
    "AwakenedOne":     (2, 2, 6, (0, 0, 1, 1, 0, 0)),
    "Deca":            (2, 2, 2, (0, 0, 0, 1, 0, 0)),
    "Donu":            (2, 2, 2, (0, 0, 1, 1, 0, 0)),
    "TimeEater":       (2, 2, 4, (0, 0, 1, 0, 0, 0)),
    # --- Act 4 (Ending) ---
    "CorruptHeart":    (2, 3, 4, (0, 0, 1, 1, 1, 0)),
    "SpireShield":     (1, 3, 3, (0, 0, 0, 0, 0, 0)),
    "SpireSpear":      (1, 3, 3, (0, 0, 0, 1, 0, 0)),
    # --- Bronze Automaton's minion ---
    "BronzeOrb":       (1, 1, 3, (0, 0, 0, 0, 0, 0)),
}

# Normalised ID → index mapping (case-insensitive, no spaces/underscores)
_MID_TO_IDX: Dict[str, int] = {}
_MONSTER_LIST = list(MONSTER_DB.keys())
for _i, _mid in enumerate(_MONSTER_LIST):
    _MID_TO_IDX[_mid.lower().replace(" ", "").replace("_", "")] = _i

MONSTER_ID_DIM = len(_MONSTER_LIST)  # ~66 — used for one-hot encoding


def _monster_id_index(monster_id: str) -> int:
    """Return the index of a monster in MONSTER_DB, or -1 if unknown."""
    key = monster_id.lower().replace(" ", "").replace("_", "")
    return _MID_TO_IDX.get(key, -1)


# Per-monster-slot feature counts
_MONSTER_ID_EMBED_DIM = 8   # compressed identity embedding
_MONSTER_MOVE_HIST_DIM = 3  # move_id, last_move_id, second_last_move_id
_MONSTER_BEHAV_DIM = 6      # behavioral flags from MONSTER_DB
_MONSTER_BASE_DIM = 12      # existing: present, hp_ratio, max_hp, block, intent(5), dmg, hits, spare
_MONSTER_SLOT_DIM = _MONSTER_BASE_DIM + _MONSTER_ID_EMBED_DIM + _MONSTER_MOVE_HIST_DIM + _MONSTER_BEHAV_DIM  # 29

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
MAX_HAND = 10
MAX_MONSTERS = 5
MAX_POTIONS = 5
MAX_CHOICES = 40

PLAYER_STATE_DIM = 15
SCREEN_TYPE_DIM = 14
HAND_DIM = MAX_HAND * 10                     # 100
MONSTER_DIM = MAX_MONSTERS * _MONSTER_SLOT_DIM  # 5 * 29 = 145
PLAYER_POWER_DIM = 20
MONSTER_POWER_DIM = MAX_MONSTERS * 8          # 40
CHOICE_DIM = 7

OBS_SIZE = (
    PLAYER_STATE_DIM
    + SCREEN_TYPE_DIM
    + HAND_DIM
    + MONSTER_DIM
    + PLAYER_POWER_DIM
    + MONSTER_POWER_DIM
    + CHOICE_DIM
)  # 341


# ---------------------------------------------------------------------------
# Ironclad card stats: (base_damage, base_block, upgraded_damage, upgraded_block)
# Cards not listed default to (0, 0, 0, 0).
# ---------------------------------------------------------------------------
CARD_STATS: dict[str, Tuple[int, int, int, int]] = {
    # --- Starter ---
    "strike_r":        (6,  0,  9,  0),
    "defend_r":        (0,  5,  0,  8),
    "bash":            (8,  0,  10, 0),
    # --- Common Attacks ---
    "anger":           (6,  0,  8,  0),
    "body slam":       (0,  0,  0,  0),
    "clash":           (14, 0,  18, 0),
    "cleave":          (8,  0,  11, 0),
    "clothesline":     (12, 0,  14, 0),
    "headbutt":        (9,  0,  12, 0),
    "heavy blade":     (14, 0,  18, 0),
    "iron wave":       (5,  5,  7,  7),
    "perfected strike":(6,  0,  6,  0),
    "pommel strike":   (9,  0,  10, 0),
    "sword boomerang": (3,  0,  3,  0),
    "thunderclap":     (4,  0,  7,  0),
    "twin strike":     (5,  0,  7,  0),
    "wild strike":     (12, 0,  17, 0),
    # --- Common Skills ---
    "armaments":       (0,  5,  0,  5),
    "flex":            (0,  0,  0,  0),
    "havoc":           (0,  0,  0,  0),
    "shrug it off":    (0,  8,  0,  11),
    "true grit":       (0,  7,  0,  9),
    "warcry":          (0,  0,  0,  0),
    # --- Uncommon Attacks ---
    "blood for blood": (18, 0,  22, 0),
    "carnage":         (20, 0,  28, 0),
    "dropkick":        (5,  0,  8,  0),
    "hemokinesis":     (15, 0,  20, 0),
    "pummel":          (2,  0,  2,  0),
    "rampage":         (8,  0,  8,  0),
    "reckless charge": (7,  0,  10, 0),
    "searing blow":    (12, 0,  16, 0),
    "sever soul":      (16, 0,  22, 0),
    "uppercut":        (13, 0,  18, 0),
    "whirlwind":       (5,  0,  8,  0),
    # --- Uncommon Skills ---
    "battle trance":   (0,  0,  0,  0),
    "burning pact":    (0,  0,  0,  0),
    "disarm":          (0,  0,  0,  0),
    "dual wield":      (0,  0,  0,  0),
    "entrench":        (0,  0,  0,  0),
    "flame barrier":   (0,  12, 0,  16),
    "ghostly armor":   (0,  10, 0,  13),
    "infernal blade":  (0,  0,  0,  0),
    "intimidate":      (0,  0,  0,  0),
    "offering":        (0,  0,  0,  0),
    "power through":   (0,  15, 0,  20),
    "rage":            (0,  0,  0,  0),
    "second wind":     (0,  0,  0,  0),
    "seeing red":      (0,  0,  0,  0),
    "sentinel":        (0,  5,  0,  8),
    # --- Uncommon Powers ---
    "combust":         (0,  0,  0,  0),
    "corruption":      (0,  0,  0,  0),
    "dark embrace":    (0,  0,  0,  0),
    "evolve":          (0,  0,  0,  0),
    "feel no pain":    (0,  0,  0,  0),
    "fire breathing":  (0,  0,  0,  0),
    "inflame":         (0,  0,  0,  0),
    "metallicize":     (0,  0,  0,  0),
    "rupture":         (0,  0,  0,  0),
    # --- Rare Attacks ---
    "bludgeon":        (32, 0,  42, 0),
    "feed":            (10, 0,  12, 0),
    "fiend fire":      (7,  0,  10, 0),
    "immolate":        (21, 0,  28, 0),
    "reaper":          (4,  0,  5,  0),
    # --- Rare Skills ---
    "double tap":      (0,  0,  0,  0),
    "exhume":          (0,  0,  0,  0),
    "impervious":      (0,  30, 0,  40),
    "limit break":     (0,  0,  0,  0),
    # --- Rare Powers ---
    "barricade":       (0,  0,  0,  0),
    "berserk":         (0,  0,  0,  0),
    "brutality":       (0,  0,  0,  0),
    "demon form":      (0,  0,  0,  0),
    "juggernaut":      (0,  0,  0,  0),
    # --- Status cards ---
    "wound":           (0,  0,  0,  0),
    "burn":            (0,  0,  0,  0),
    "dazed":           (0,  0,  0,  0),
    "slimed":          (0,  0,  0,  0),
    "void":            (0,  0,  0,  0),
}

# Fallback lookup by display name for cards whose card_id has a class suffix
# (e.g. "Strike_R" -> look up "strike")
_CARD_STATS_BY_NAME: dict[str, Tuple[int, int, int, int]] = {}
for _cid, _vals in CARD_STATS.items():
    _stripped = _cid.replace("_r", "").replace("_g", "").replace("_b", "").replace("_p", "")
    if _stripped != _cid:
        _CARD_STATS_BY_NAME[_stripped] = _vals


def _card_damage_block(card: Any) -> Tuple[float, float]:
    """Look up base damage and block for a card, accounting for upgrades."""
    cid = str(getattr(card, "card_id", "") or "").lower()
    name = str(getattr(card, "name", "") or "").lower().rstrip("+")
    upgraded = int(getattr(card, "upgrades", 0) or 0) > 0

    stats = CARD_STATS.get(cid) or CARD_STATS.get(name) or _CARD_STATS_BY_NAME.get(name)
    if stats is None:
        return 0.0, 0.0
    base_d, base_b, up_d, up_b = stats
    if upgraded:
        return float(up_d), float(up_b)
    return float(base_d), float(base_b)


# ---------------------------------------------------------------------------
# Screen type encoding
# ---------------------------------------------------------------------------
SCREEN_TYPE_NAMES = [
    "EVENT", "CHEST", "SHOP_ROOM", "REST", "CARD_REWARD",
    "COMBAT_REWARD", "MAP", "BOSS_REWARD", "SHOP_SCREEN", "GRID",
    "HAND_SELECT", "GAME_OVER", "COMPLETE", "NONE",
]
_SCREEN_IDX = {name: i for i, name in enumerate(SCREEN_TYPE_NAMES)}


def _screen_type_str(gs: Any) -> str:
    st = getattr(gs, "screen_type", None)
    if st is None:
        return "NONE"
    name = getattr(st, "name", st)
    return str(name) if name else "NONE"


# ---------------------------------------------------------------------------
# Power encoding
# ---------------------------------------------------------------------------
PLAYER_POWER_IDS = [
    "strength", "dexterity", "vulnerable", "weakened", "frail",
    "artifact", "metallicize", "platedarmor", "thorns", "barricade",
    "rage", "demonform", "combust", "brutality", "darkembrace",
    "evolve", "feelnopain", "firebreathing", "corruption", "juggernaut",
]

MONSTER_POWER_IDS = [
    "strength", "vulnerable", "weakened", "artifact",
    "ritual", "curlup", "thorns", "angry",
]


def _encode_powers(powers: list, power_ids: list[str]) -> np.ndarray:
    """Encode a list of Power objects into a fixed-size vector."""
    out = np.zeros(len(power_ids), dtype=np.float32)
    lookup = {pid: i for i, pid in enumerate(power_ids)}
    for p in (powers or []):
        pid = str(getattr(p, "power_id", "") or "").lower()
        # Power IDs from the game vary in casing; normalize aggressively
        pid_clean = pid.replace(" ", "").replace("_", "")
        idx = lookup.get(pid_clean)
        if idx is None:
            pname = str(getattr(p, "power_name", "") or "").lower().replace(" ", "").replace("_", "")
            idx = lookup.get(pname)
        if idx is not None:
            amt = float(getattr(p, "amount", 0) or 0)
            out[idx] = np.clip(amt / 10.0, -3.0, 3.0)
    return out


# ---------------------------------------------------------------------------
# Intent encoding
# ---------------------------------------------------------------------------
_ATTACK_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
_DEFEND_INTENTS = {"DEFEND", "DEFEND_BUFF", "DEFEND_DEBUFF", "ATTACK_DEFEND"}
_BUFF_INTENTS = {"BUFF", "ATTACK_BUFF", "DEFEND_BUFF"}
_DEBUFF_INTENTS = {"DEBUFF", "STRONG_DEBUFF", "ATTACK_DEBUFF", "DEFEND_DEBUFF"}


def _intent_vec(intent: Any) -> np.ndarray:
    """5-dim binary vector: [attack, defend, buff, debuff, other]."""
    name = str(getattr(intent, "name", intent) or "UNKNOWN").upper()
    vec = np.zeros(5, dtype=np.float32)
    matched = False
    if name in _ATTACK_INTENTS:
        vec[0] = 1.0; matched = True
    if name in _DEFEND_INTENTS:
        vec[1] = 1.0; matched = True
    if name in _BUFF_INTENTS:
        vec[2] = 1.0; matched = True
    if name in _DEBUFF_INTENTS:
        vec[3] = 1.0; matched = True
    if not matched:
        vec[4] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Monster identity embedding
# ---------------------------------------------------------------------------
# Pre-compute a compressed 8-dim embedding for each monster using a
# deterministic hash of its index. This is much smaller than a full one-hot
# (66 dims) and gives the network a unique, learnable fingerprint per enemy.
_MONSTER_EMBED_CACHE: Dict[int, np.ndarray] = {}


def _monster_embed(monster_id: str) -> np.ndarray:
    """Return an 8-dim identity vector for a monster."""
    idx = _monster_id_index(monster_id)
    if idx in _MONSTER_EMBED_CACHE:
        return _MONSTER_EMBED_CACHE[idx]

    vec = np.zeros(_MONSTER_ID_EMBED_DIM, dtype=np.float32)
    if idx < 0:
        _MONSTER_EMBED_CACHE[idx] = vec
        return vec

    rng = np.random.RandomState(seed=idx + 42)
    vec = rng.randn(_MONSTER_ID_EMBED_DIM).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    _MONSTER_EMBED_CACHE[idx] = vec
    return vec


def _monster_behavior_vec(monster_id: str) -> np.ndarray:
    """Return 6-dim behavioral flags + 3-dim metadata for a monster."""
    idx = _monster_id_index(monster_id)
    if idx < 0:
        return np.zeros(_MONSTER_BEHAV_DIM, dtype=np.float32)

    mid = _MONSTER_LIST[idx]
    mtype, act, n_moves, flags = MONSTER_DB[mid]
    return np.array(flags, dtype=np.float32)


def _encode_move_history(m: Any) -> np.ndarray:
    """Encode move_id, last_move_id, second_last_move_id as normalised floats."""
    vec = np.zeros(_MONSTER_MOVE_HIST_DIM, dtype=np.float32)
    mid = int(getattr(m, "move_id", -1) or -1)
    last = getattr(m, "last_move_id", None)
    second = getattr(m, "second_last_move_id", None)
    vec[0] = (mid + 1) / 10.0
    vec[1] = ((int(last) + 1) / 10.0) if last is not None else 0.0
    vec[2] = ((int(second) + 1) / 10.0) if second is not None else 0.0
    return vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_alive(m: Any) -> bool:
    if getattr(m, "is_gone", False):
        return False
    hp = int(getattr(m, "current_hp", 0) or 0)
    return hp > 0 or getattr(m, "half_dead", False)


def living_monsters(monsters: list) -> list:
    return [m for m in (monsters or []) if _is_alive(m)]


def _card_type_str(card: Any) -> str:
    ct = getattr(card, "type", None)
    return str(getattr(ct, "name", ct) or "").upper()


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------
def encode_game_state(gs: Any) -> np.ndarray:
    """Convert a SpireComm Game object to a fixed-size float32 vector."""
    if gs is None:
        return np.zeros(OBS_SIZE, dtype=np.float32)

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    offset = 0

    # === PLAYER STATE (15) ===
    hp = float(getattr(gs, "current_hp", 0) or 0)
    max_hp = float(getattr(gs, "max_hp", 1) or 1)

    player = getattr(gs, "player", None)
    energy = float(getattr(player, "energy", 0) or 0) if player else 0.0
    block = float(getattr(player, "block", 0) or 0) if player else 0.0

    hand = list(getattr(gs, "hand", []) or [])
    deck = list(getattr(gs, "deck", []) or [])
    draw_pile = list(getattr(gs, "draw_pile", []) or [])
    discard_pile = list(getattr(gs, "discard_pile", []) or [])
    exhaust_pile = list(getattr(gs, "exhaust_pile", []) or [])
    monsters = list(getattr(gs, "monsters", []) or [])
    potions = list(getattr(gs, "potions", []) or [])

    real_potions = sum(
        1 for p in potions
        if str(getattr(p, "potion_id", "") or "").lower() != "potion slot"
    )
    deck_attacks = sum(1 for c in deck if _card_type_str(c) == "ATTACK")
    deck_skills = sum(1 for c in deck if _card_type_str(c) == "SKILL")
    deck_powers = sum(1 for c in deck if _card_type_str(c) == "POWER")

    obs[offset:offset + 15] = [
        hp / max(1.0, max_hp),                                  # 0  hp ratio
        energy / 4.0,                                            # 1  energy
        block / 40.0,                                            # 2  block
        float(getattr(gs, "gold", 0) or 0) / 999.0,            # 3  gold
        float(getattr(gs, "floor", 0) or 0) / 55.0,            # 4  floor
        float(getattr(gs, "act", 0) or 0) / 4.0,               # 5  act
        float(getattr(gs, "turn", 0) or 0) / 20.0,             # 6  turn
        len(deck) / 40.0,                                        # 7  deck size
        len(draw_pile) / 40.0,                                   # 8  draw pile
        len(discard_pile) / 40.0,                                # 9  discard pile
        len(exhaust_pile) / 20.0,                                # 10 exhaust pile
        real_potions / 5.0,                                      # 11 potions
        deck_attacks / 20.0,                                     # 12 attacks in deck
        deck_skills / 20.0,                                      # 13 skills in deck
        deck_powers / 10.0,                                      # 14 powers in deck
    ]
    offset += PLAYER_STATE_DIM

    # === SCREEN TYPE (14 one-hot) ===
    screen_name = _screen_type_str(gs)
    sidx = _SCREEN_IDX.get(screen_name, _SCREEN_IDX["NONE"])
    obs[offset + sidx] = 1.0
    offset += SCREEN_TYPE_DIM

    # === HAND CARDS (10 × 10 = 100) ===
    for i in range(MAX_HAND):
        base = offset + i * 10
        if i < len(hand):
            card = hand[i]
            dmg, blk = _card_damage_block(card)
            ct = _card_type_str(card)
            cost = float(getattr(card, "cost", 0) or 0)
            obs[base + 0] = 1.0                                         # present
            obs[base + 1] = max(0.0, cost) / 4.0                        # cost
            obs[base + 2] = dmg / 40.0                                   # damage
            obs[base + 3] = blk / 40.0                                   # block
            obs[base + 4] = 1.0 if ct == "ATTACK" else 0.0
            obs[base + 5] = 1.0 if ct == "SKILL" else 0.0
            obs[base + 6] = 1.0 if ct == "POWER" else 0.0
            obs[base + 7] = 1.0 if ct in ("STATUS", "CURSE") else 0.0
            obs[base + 8] = 1.0 if getattr(card, "is_playable", False) else 0.0
            obs[base + 9] = 1.0 if getattr(card, "has_target", False) else 0.0
    offset += HAND_DIM

    # === MONSTERS (5 × 29 = 145) ===
    alive = living_monsters(monsters)
    for i in range(MAX_MONSTERS):
        base = offset + i * _MONSTER_SLOT_DIM
        if i < len(alive):
            m = alive[i]
            m_hp = float(getattr(m, "current_hp", 0) or 0)
            m_max = float(getattr(m, "max_hp", 1) or 1)
            m_block = float(getattr(m, "block", 0) or 0)
            intent = getattr(m, "intent", None)
            ivec = _intent_vec(intent)
            m_dmg = float(getattr(m, "move_adjusted_damage", 0) or 0)
            if m_dmg <= 0:
                m_dmg = float(getattr(m, "move_base_damage", 0) or 0)
            m_hits = float(getattr(m, "move_hits", 0) or 0)

            m_id_str = str(getattr(m, "monster_id", "") or "")

            p = base
            obs[p] = 1.0;                         p += 1   # present
            obs[p] = m_hp / max(1.0, m_max);      p += 1   # hp ratio
            obs[p] = m_max / 300.0;                p += 1   # max hp (normalised)
            obs[p] = m_block / 40.0;               p += 1   # block
            obs[p:p + 5] = ivec;                   p += 5   # intent (5)
            obs[p] = max(0.0, m_dmg) / 50.0;      p += 1   # incoming damage
            obs[p] = m_hits / 4.0;                 p += 1   # num hits
            obs[p] = 0.0;                          p += 1   # spare (base=12)

            obs[p:p + _MONSTER_ID_EMBED_DIM] = _monster_embed(m_id_str)
            p += _MONSTER_ID_EMBED_DIM                       # identity (8)

            obs[p:p + _MONSTER_MOVE_HIST_DIM] = _encode_move_history(m)
            p += _MONSTER_MOVE_HIST_DIM                      # move history (3)

            obs[p:p + _MONSTER_BEHAV_DIM] = _monster_behavior_vec(m_id_str)
            p += _MONSTER_BEHAV_DIM                          # behavior flags (6)
    offset += MONSTER_DIM

    # === PLAYER POWERS (20) ===
    player_powers = getattr(player, "powers", []) if player else []
    obs[offset:offset + PLAYER_POWER_DIM] = _encode_powers(
        player_powers, PLAYER_POWER_IDS
    )
    offset += PLAYER_POWER_DIM

    # === MONSTER POWERS (5 × 8 = 40) ===
    for i in range(MAX_MONSTERS):
        base = offset + i * 8
        if i < len(alive):
            m_powers = getattr(alive[i], "powers", [])
            obs[base:base + 8] = _encode_powers(m_powers, MONSTER_POWER_IDS)
    offset += MONSTER_POWER_DIM

    # === CHOICE CONTEXT (7) ===
    choice_list = list(getattr(gs, "choice_list", []) or [])
    obs[offset] = len(choice_list) / float(MAX_CHOICES)
    offset += CHOICE_DIM

    return obs
