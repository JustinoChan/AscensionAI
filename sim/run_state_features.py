"""Compact run-state feature vector for the drafting value model V.

Pure + torch-free: maps a deck role-composition dict (as produced by
V2._deck_roles) plus run scalars to a fixed-length float32 vector. The eval
computes the role dict live and the scalars from gc; this module only
vectorizes, so it is unit-testable offline."""
import numpy as np

# Deck-composition keys mirror V2._deck_roles output (qsum intentionally omitted).
DECK_KEYS = ("size", "starters", "upgraded", "upg_tempo", "damage", "block",
             "aoe", "scaling", "draw_energy", "powers", "junk")
SCALAR_KEYS = ("act", "floor", "hp_frac", "gold", "relics")
FEATURE_KEYS = DECK_KEYS + SCALAR_KEYS
FEAT_DIM = len(FEATURE_KEYS)

# Rough normalizers so every feature lands ~[0,1] for a small MLP.
_NORM = {"size": 30.0, "starters": 10.0, "upgraded": 12.0, "upg_tempo": 10.0,
         "damage": 12.0, "block": 10.0, "aoe": 5.0, "scaling": 6.0,
         "draw_energy": 6.0, "powers": 6.0, "junk": 6.0,
         "act": 3.0, "floor": 56.0, "hp_frac": 1.0, "gold": 400.0, "relics": 12.0}


def vectorize(roles, scalars):
    """roles: dict with any of DECK_KEYS (missing -> 0). scalars: dict with
    SCALAR_KEYS. Returns float32 np.ndarray of length FEAT_DIM."""
    out = np.zeros(FEAT_DIM, np.float32)
    for j, k in enumerate(DECK_KEYS):
        out[j] = float(roles.get(k, 0)) / _NORM[k]
    for j, k in enumerate(SCALAR_KEYS):
        out[len(DECK_KEYS) + j] = float(scalars.get(k, 0)) / _NORM[k]
    return out


# Offered-card context block for the enriched drafting features (v3).
CARD_KEYS = ("rare", "uncommon", "common", "qual", "frontload",
             "is_attack", "is_skill", "is_power", "is_take")
FEAT2_DIM = FEAT_DIM + len(CARD_KEYS)


def vectorize2(roles, scalars, card):
    """Enriched candidate features: run-state vector + offered-card block.
    card: None for skip, else dict with type/rarity/qual/frontload (the raw
    log also stores name, unvectorized, so future feature sets can be built
    offline). Returns float32 np.ndarray of length FEAT2_DIM."""
    ext = np.zeros(len(CARD_KEYS), np.float32)
    if card is not None:
        r = str(card.get("rarity", "")).upper()
        ext[0] = 1.0 if r == "RARE" else 0.0
        ext[1] = 1.0 if r == "UNCOMMON" else 0.0
        ext[2] = 1.0 if r == "COMMON" else 0.0
        ext[3] = float(card.get("qual", 0)) / 5.0
        ext[4] = 1.0 if card.get("frontload") else 0.0
        t = str(card.get("type", "")).upper()
        ext[5] = 1.0 if t == "ATTACK" else 0.0
        ext[6] = 1.0 if t == "SKILL" else 0.0
        ext[7] = 1.0 if t == "POWER" else 0.0
        ext[8] = 1.0
    return np.concatenate([vectorize(roles, scalars), ext])
