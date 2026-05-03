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

from game_data import (
    CARD_ID_LIST,
    CARD_MECHANICS,
    POTION_EFFECTS,
    RELIC_DB,
    RELIC_FEATURE_DIM,
    RELIC_FEATURE_NORMS,
    R_RELIC_COUNT,
    R_RELIC_VALUE,
)


# ---------------------------------------------------------------------------
# Monster knowledge base — every STS1 monster with behavioral metadata.
# Sourced from spire-archive.com API (66 monsters).
#
# Fields per monster:
#   type   — 0=Normal, 1=Elite, 2=Boss
#   act    — 0=exordium(Act1), 1=city(Act2), 2=beyond(Act3), 3=ending(Act4)
#   n_moves — number of distinct moves
#   flags  — 7 behavioral booleans:
#       [0] enrages_on_skill  (Nob-like: punishes skill cards)
#       [1] splits_low_hp     (slime boss, large slimes split at ~50% HP)
#       [2] scales_strength   (Cultist ritual, Demon Form-like buff loops)
#       [3] multi_attacker    (has a move that hits 2+ times)
#       [4] retaliates        (thorns, curl-up, reactive damage)
#       [5] escapes           (gremlin escape, looter flee)
#       [6] spawns_minions    (Gremlin Leader, Reptomancer, Bronze Automaton)
# ---------------------------------------------------------------------------
MONSTER_DB: Dict[str, Tuple[int, int, int, Tuple[int, ...]]] = {
    # --- Act 1 Normal ---
    "AcidSlime_L":     (0, 0, 4, (0, 1, 0, 0, 0, 0, 0)),
    "AcidSlime_M":     (0, 0, 3, (0, 0, 0, 0, 0, 0, 0)),
    "AcidSlime_S":     (0, 0, 2, (0, 0, 0, 0, 0, 0, 0)),
    "SpikeSlime_L":    (0, 0, 3, (0, 1, 0, 0, 0, 0, 0)),
    "SpikeSlime_M":    (0, 0, 2, (0, 0, 0, 0, 0, 0, 0)),
    "SpikeSlime_S":    (0, 0, 1, (0, 0, 0, 0, 0, 0, 0)),
    "JawWorm":         (0, 0, 3, (0, 0, 1, 0, 0, 0, 0)),
    "Cultist":         (0, 0, 2, (0, 0, 1, 0, 0, 0, 0)),
    "FungiBeast":      (0, 0, 2, (0, 0, 1, 0, 0, 0, 0)),
    "GremlinFat":      (0, 0, 2, (0, 0, 0, 0, 0, 1, 0)),
    "GremlinWarrior":  (0, 0, 2, (0, 0, 0, 0, 0, 1, 0)),
    "GremlinThief":    (0, 0, 2, (0, 0, 0, 0, 0, 1, 0)),
    "GremlinTsundere": (0, 0, 3, (0, 0, 0, 0, 0, 1, 0)),
    "GremlinWizard":   (0, 0, 3, (0, 0, 0, 0, 0, 1, 0)),
    "FuzzyLouseNormal":(0, 0, 2, (0, 0, 1, 0, 1, 0, 0)),
    "FuzzyLouseDefensive":(0, 0, 2, (0, 0, 0, 0, 1, 0, 0)),
    "SlaverBlue":      (0, 0, 2, (0, 0, 0, 0, 0, 0, 0)),
    "SlaverRed":       (0, 0, 3, (0, 0, 0, 0, 0, 0, 0)),
    "Looter":          (0, 0, 2, (0, 0, 0, 0, 0, 1, 0)),
    "Apology Slime":   (0, 0, 2, (0, 0, 0, 0, 0, 0, 0)),
    # --- Act 1 Elite ---
    "GremlinNob":      (1, 0, 3, (1, 0, 1, 0, 0, 0, 0)),
    "Lagavulin":       (1, 0, 4, (0, 0, 0, 0, 0, 0, 0)),
    "Sentry":          (1, 0, 2, (0, 0, 0, 0, 0, 0, 0)),
    # --- Act 1 Boss ---
    "Hexaghost":       (2, 0, 6, (0, 0, 0, 1, 0, 0, 0)),
    "TheGuardian":     (2, 0, 7, (0, 0, 0, 1, 1, 0, 0)),
    "SlimeBoss":       (2, 0, 4, (0, 1, 0, 0, 0, 0, 0)),
    # --- Act 2 Normal ---
    "Chosen":          (0, 1, 5, (0, 0, 0, 1, 0, 0, 0)),
    "Byrd":            (0, 1, 6, (0, 0, 0, 1, 0, 0, 0)),
    "ShelledParasite":  (0, 1, 4, (0, 0, 0, 1, 1, 0, 0)),
    "SnakePlant":      (0, 1, 2, (0, 0, 0, 1, 0, 0, 0)),
    "Centurion":       (0, 1, 3, (0, 0, 0, 1, 0, 0, 0)),
    "Healer":          (0, 1, 3, (0, 0, 0, 0, 0, 0, 0)),
    "Snecko":          (0, 1, 3, (0, 0, 0, 0, 0, 0, 0)),
    "Mugger":          (0, 1, 2, (0, 0, 0, 0, 0, 1, 0)),
    "SphericGuardian": (0, 1, 4, (0, 0, 0, 1, 0, 0, 0)),
    "BanditBear":      (0, 1, 1, (0, 0, 0, 0, 0, 0, 0)),
    "BanditChild":     (0, 1, 1, (0, 0, 0, 1, 0, 0, 0)),
    "BanditLeader":    (0, 1, 1, (0, 0, 0, 0, 0, 0, 0)),
    "TorchHead":       (0, 1, 1, (0, 0, 0, 0, 0, 0, 0)),
    # --- Act 2 Elite ---
    "GremlinLeader":   (1, 1, 3, (0, 0, 0, 1, 0, 0, 1)),
    "SlaverBoss":      (1, 1, 1, (0, 0, 0, 0, 0, 0, 0)),
    "BookOfStabbing":  (1, 1, 2, (0, 0, 1, 0, 0, 0, 0)),
    # --- Act 2 Boss ---
    "BronzeAutomaton": (2, 1, 5, (0, 0, 0, 1, 0, 0, 1)),
    "TheCollector":    (2, 1, 5, (0, 0, 0, 0, 0, 0, 0)),
    "Champ":           (2, 1, 7, (0, 0, 1, 1, 0, 0, 0)),
    # --- Act 3 Normal ---
    "Darkling":        (0, 2, 5, (0, 0, 0, 1, 0, 0, 0)),
    "OrbWalker":       (0, 2, 2, (0, 0, 0, 0, 0, 0, 0)),
    "Spiker":          (0, 2, 2, (0, 0, 0, 0, 1, 0, 0)),
    "Repulsor":        (0, 2, 2, (0, 0, 0, 0, 0, 0, 0)),
    "Exploder":        (0, 2, 2, (0, 0, 0, 0, 0, 0, 0)),
    "Maw":             (0, 2, 4, (0, 0, 1, 0, 0, 0, 0)),
    "Serpent":         (0, 2, 3, (0, 0, 0, 0, 0, 0, 0)),
    "Dagger":          (0, 2, 2, (0, 0, 0, 0, 0, 0, 0)),
    # --- Act 3 Elite ---
    "GiantHead":       (1, 2, 3, (0, 0, 1, 0, 0, 0, 0)),
    "Nemesis":         (1, 2, 3, (0, 0, 0, 1, 0, 0, 0)),
    "Reptomancer":     (1, 2, 3, (0, 0, 0, 1, 0, 0, 1)),
    "Transient":       (1, 2, 1, (0, 0, 1, 0, 0, 0, 0)),
    "WrithingMass":    (1, 2, 5, (0, 0, 0, 1, 0, 0, 0)),
    # --- Act 3 Boss ---
    "AwakenedOne":     (2, 2, 6, (0, 0, 1, 1, 0, 0, 0)),
    "Deca":            (2, 2, 2, (0, 0, 0, 1, 0, 0, 0)),
    "Donu":            (2, 2, 2, (0, 0, 1, 1, 0, 0, 0)),
    "TimeEater":       (2, 2, 4, (0, 0, 1, 0, 0, 0, 0)),
    # --- Act 4 (Ending) ---
    "CorruptHeart":    (2, 3, 4, (0, 0, 1, 1, 1, 0, 0)),
    "SpireShield":     (1, 3, 3, (0, 0, 0, 0, 0, 0, 0)),
    "SpireSpear":      (1, 3, 3, (0, 0, 0, 1, 0, 0, 0)),
    # --- Bronze Automaton's minion ---
    "BronzeOrb":       (1, 1, 3, (0, 0, 0, 0, 0, 0, 0)),
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
_MONSTER_BEHAV_DIM = 7      # behavioral flags from MONSTER_DB
_MONSTER_BASE_DIM = 12      # existing: present, hp_ratio, max_hp, block, intent(5), dmg, hits, spare
_MONSTER_SLOT_DIM = _MONSTER_BASE_DIM + _MONSTER_ID_EMBED_DIM + _MONSTER_MOVE_HIST_DIM + _MONSTER_BEHAV_DIM  # 30

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
MAX_HAND = 10
MAX_MONSTERS = 5
MAX_POTIONS = 5
MAX_CHOICES = 40

PLAYER_STATE_DIM = 15
SCREEN_TYPE_DIM = 14
HAND_CARD_DIM = 16                            # was 10: +4 identity +1 exhausts +1 upgraded
HAND_DIM = MAX_HAND * HAND_CARD_DIM           # 160
MONSTER_DIM = MAX_MONSTERS * _MONSTER_SLOT_DIM  # 5 * 30 = 150
PLAYER_POWER_DIM = 20
MONSTER_POWER_DIM = MAX_MONSTERS * 8          # 40
CHOICE_DIM = 7
RELIC_DIM = RELIC_FEATURE_DIM                 # 25
POTION_SLOT_DIM = 8
POTION_DIM = MAX_POTIONS * POTION_SLOT_DIM    # 40
DECK_PROFILE_DIM = 20

MAX_MAP_CHOICES = 4
MAP_NODE_TYPES = 6                            # M, E, R, $, ?, T
MAP_LOOKAHEAD = 3
MAP_CHOICE_DIM = MAP_NODE_TYPES + 3           # one-hot type + elite/rest/combat density
MAP_GLOBAL_DIM = 3                            # n_choices, boss_avail, floors_remaining
MAP_DIM = MAX_MAP_CHOICES * MAP_CHOICE_DIM + MAP_GLOBAL_DIM  # 39

_MAP_SYMBOL_IDX = {"M": 0, "E": 1, "R": 2, "$": 3, "?": 4, "T": 5}


def _map_lookahead(start_node, game_map, depth: int = MAP_LOOKAHEAD) -> List[int]:
    """BFS from start_node for `depth` floors, counting reachable node types."""
    counts = [0] * MAP_NODE_TYPES
    if start_node is None or game_map is None:
        return counts
    if not hasattr(game_map, "get_node"):
        return counts
    real = game_map.get_node(start_node.x, start_node.y)
    if real is None:
        return counts
    frontier = list(real.children)
    for _ in range(depth):
        next_frontier = []
        seen_xy = set()
        for node in frontier:
            key = (node.x, node.y)
            if key in seen_xy:
                continue
            seen_xy.add(key)
            sym = getattr(node, "symbol", "?")
            idx = _MAP_SYMBOL_IDX.get(sym, 4)
            counts[idx] += 1
            next_frontier.extend(node.children)
        frontier = next_frontier
    return counts


OBS_SIZE = (
    PLAYER_STATE_DIM
    + SCREEN_TYPE_DIM
    + HAND_DIM
    + MONSTER_DIM
    + PLAYER_POWER_DIM
    + MONSTER_POWER_DIM
    + CHOICE_DIM
    + RELIC_DIM
    + POTION_DIM
    + DECK_PROFILE_DIM
    + MAP_DIM
)  # 530


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
    """Return 7-dim behavioral flags for a monster."""
    idx = _monster_id_index(monster_id)
    if idx < 0:
        return np.zeros(_MONSTER_BEHAV_DIM, dtype=np.float32)

    mid = _MONSTER_LIST[idx]
    mtype, act, n_moves, flags = MONSTER_DB[mid]
    return np.array(flags, dtype=np.float32)


# ---------------------------------------------------------------------------
# Card identity embedding
# ---------------------------------------------------------------------------
_CARD_ID_EMBED_DIM = 4

_CARD_ID_TO_IDX: Dict[str, int] = {}
for _i, _cid in enumerate(CARD_ID_LIST):
    _CARD_ID_TO_IDX[_cid.lower().replace(" ", "").replace("_", "")] = _i

_CARD_EMBED_CACHE: Dict[int, np.ndarray] = {}


def _card_embed(card_id: str) -> np.ndarray:
    """Return a 4-dim identity vector for a card."""
    key = card_id.lower().replace(" ", "").replace("_", "")
    idx = _CARD_ID_TO_IDX.get(key, -1)
    if idx in _CARD_EMBED_CACHE:
        return _CARD_EMBED_CACHE[idx]
    vec = np.zeros(_CARD_ID_EMBED_DIM, dtype=np.float32)
    if idx < 0:
        _CARD_EMBED_CACHE[idx] = vec
        return vec
    rng = np.random.RandomState(seed=idx + 1000)
    vec = rng.randn(_CARD_ID_EMBED_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    _CARD_EMBED_CACHE[idx] = vec
    return vec


# ---------------------------------------------------------------------------
# Relic ID normalisation for lookup
# ---------------------------------------------------------------------------
_RELIC_ID_NORM: Dict[str, str] = {}
for _rid in RELIC_DB:
    _RELIC_ID_NORM[
        _rid.lower().replace(" ", "").replace("_", "")
        .replace("-", "").replace("'", "")
    ] = _rid


def _relic_lookup(relic_id: str) -> str | None:
    """Return the canonical RELIC_DB key for a relic_id, or None."""
    key = (relic_id.lower().replace(" ", "").replace("_", "")
           .replace("-", "").replace("'", ""))
    return _RELIC_ID_NORM.get(key)


def _safe_int(val: Any, default: int = -1) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _encode_move_history(m: Any) -> np.ndarray:
    """Encode move_id, last_move_id, second_last_move_id as normalised floats."""
    vec = np.zeros(_MONSTER_MOVE_HIST_DIM, dtype=np.float32)
    mid = _safe_int(getattr(m, "move_id", -1))
    last = _safe_int(getattr(m, "last_move_id", None))
    second = _safe_int(getattr(m, "second_last_move_id", None))
    vec[0] = (mid + 1) / 10.0
    vec[1] = (last + 1) / 10.0 if last >= 0 else 0.0
    vec[2] = (second + 1) / 10.0 if second >= 0 else 0.0
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

    # === HAND CARDS (10 × 16 = 160) ===
    for i in range(MAX_HAND):
        base = offset + i * HAND_CARD_DIM
        if i < len(hand):
            card = hand[i]
            dmg, blk = _card_damage_block(card)
            ct = _card_type_str(card)
            cost = float(getattr(card, "cost", 0) or 0)
            cid = str(getattr(card, "card_id", "") or "")
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
            obs[base + 10:base + 10 + _CARD_ID_EMBED_DIM] = _card_embed(cid)
            obs[base + 14] = 1.0 if getattr(card, "exhausts", False) else 0.0
            obs[base + 15] = 1.0 if int(getattr(card, "upgrades", 0) or 0) > 0 else 0.0
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

    # === RELIC FEATURES (25) ===
    relics = list(getattr(gs, "relics", []) or [])
    relic_vec = np.zeros(RELIC_DIM, dtype=np.float32)
    relic_count = 0
    relic_value_sum = 0.0
    for r in relics:
        rid = str(getattr(r, "relic_id", "") or "")
        canon = _relic_lookup(rid)
        relic_count += 1
        if canon is not None:
            quality, effects = RELIC_DB[canon]
            relic_value_sum += quality
            for fidx, fval in effects.items():
                relic_vec[fidx] += fval
        else:
            relic_value_sum += 1.0
    relic_vec[R_RELIC_COUNT] = relic_count
    relic_vec[R_RELIC_VALUE] = relic_value_sum
    for fi in range(RELIC_DIM):
        relic_vec[fi] /= RELIC_FEATURE_NORMS[fi]
    obs[offset:offset + RELIC_DIM] = np.clip(relic_vec, -3.0, 3.0)
    offset += RELIC_DIM

    # === POTION SLOTS (5 × 8 = 40) ===
    for i in range(MAX_POTIONS):
        base = offset + i * POTION_SLOT_DIM
        if i < len(potions):
            p = potions[i]
            pid = str(getattr(p, "potion_id", "") or "")
            is_present = pid.lower() != "potion slot" and pid != ""
            obs[base + 0] = 1.0 if is_present else 0.0
            obs[base + 1] = 1.0 if getattr(p, "can_use", False) else 0.0
            obs[base + 2] = 1.0 if getattr(p, "requires_target", False) else 0.0
            effects = POTION_EFFECTS.get(pid, (0, 0, 0, 0, 0))
            obs[base + 3] = float(effects[0])  # deals_damage
            obs[base + 4] = float(effects[1])  # gives_block
            obs[base + 5] = float(effects[2])  # gives_strength
            obs[base + 6] = float(effects[3])  # gives_dex
            obs[base + 7] = float(effects[4])  # heals
    offset += POTION_DIM

    # === DECK PROFILE (20) ===
    n_deck = max(1, len(deck))
    total_cost = 0.0
    n_attacks = 0
    n_skills = 0
    n_powers = 0
    n_status_curse = 0
    n_exhaust = 0
    n_draw = 0
    n_aoe = 0
    n_multi = 0
    n_zero_cost = 0
    str_scale_sum = 0.0
    n_upgraded = 0
    quality_sum = 0.0
    key_cards = {"Barricade": 0, "Corruption": 0, "Demon Form": 0,
                 "Feel No Pain": 0, "Limit Break": 0, "Offering": 0}
    for c in deck:
        cid = str(getattr(c, "card_id", "") or "")
        ct = _card_type_str(c)
        cost = float(getattr(c, "cost", 0) or 0)
        total_cost += max(0.0, cost)
        if ct == "ATTACK":
            n_attacks += 1
        elif ct == "SKILL":
            n_skills += 1
        elif ct == "POWER":
            n_powers += 1
        if ct in ("STATUS", "CURSE"):
            n_status_curse += 1
        if getattr(c, "exhausts", False):
            n_exhaust += 1
        if cost <= 0 and ct in ("ATTACK", "SKILL"):
            n_zero_cost += 1
        if int(getattr(c, "upgrades", 0) or 0) > 0:
            n_upgraded += 1
        mech = CARD_MECHANICS.get(cid)
        if mech is not None:
            draws, aoe, multi, sscale, qual = mech
            n_draw += draws
            if aoe:
                n_aoe += 1
            if multi:
                n_multi += 1
            str_scale_sum += sscale
            quality_sum += qual
        if cid in key_cards:
            key_cards[cid] = 1

    obs[offset + 0] = len(deck) / 40.0
    obs[offset + 1] = (total_cost / n_deck) / 4.0
    obs[offset + 2] = n_attacks / n_deck
    obs[offset + 3] = n_skills / n_deck
    obs[offset + 4] = n_powers / n_deck
    obs[offset + 5] = n_status_curse / 10.0
    obs[offset + 6] = n_exhaust / 10.0
    obs[offset + 7] = n_draw / 10.0
    obs[offset + 8] = n_aoe / 5.0
    obs[offset + 9] = n_multi / 5.0
    obs[offset + 10] = n_zero_cost / 10.0
    obs[offset + 11] = str_scale_sum / 10.0
    obs[offset + 12] = n_upgraded / n_deck
    obs[offset + 13] = (quality_sum / n_deck) / 10.0
    obs[offset + 14] = float(key_cards["Barricade"])
    obs[offset + 15] = float(key_cards["Corruption"])
    obs[offset + 16] = float(key_cards["Demon Form"])
    obs[offset + 17] = float(key_cards["Feel No Pain"])
    obs[offset + 18] = float(key_cards["Limit Break"])
    obs[offset + 19] = float(key_cards["Offering"])
    offset += DECK_PROFILE_DIM

    # === MAP PATH ENCODING (39) ===
    scr = getattr(gs, "screen", None)
    if screen_name == "MAP" and scr is not None:
        game_map = getattr(gs, "map", None)
        next_nodes = list(getattr(scr, "next_nodes", []) or [])
        boss_avail = bool(getattr(scr, "boss_available", False))
        act = int(getattr(gs, "act", 0) or 0)
        floor = int(getattr(gs, "floor", 0) or 0)
        act_last_floor = {1: 17, 2: 34, 3: 51}.get(act, 51)
        floors_remaining = max(0, act_last_floor - floor)

        n_valid = min(len(next_nodes), MAX_MAP_CHOICES)
        for ci in range(n_valid):
            node = next_nodes[ci]
            base = offset + ci * MAP_CHOICE_DIM
            sym = getattr(node, "symbol", "?")
            type_idx = _MAP_SYMBOL_IDX.get(sym, 4)
            obs[base + type_idx] = 1.0
            counts = _map_lookahead(node, game_map)
            total = max(1, sum(counts))
            obs[base + MAP_NODE_TYPES + 0] = counts[1] / total  # elite density
            obs[base + MAP_NODE_TYPES + 1] = counts[2] / total  # rest density
            obs[base + MAP_NODE_TYPES + 2] = counts[0] / total  # combat density

        gbase = offset + MAX_MAP_CHOICES * MAP_CHOICE_DIM
        obs[gbase + 0] = n_valid / MAX_MAP_CHOICES
        obs[gbase + 1] = float(boss_avail)
        obs[gbase + 2] = floors_remaining / 17.0
    offset += MAP_DIM

    np.nan_to_num(obs, copy=False)
    return obs
