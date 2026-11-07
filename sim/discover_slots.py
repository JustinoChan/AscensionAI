"""Identify the Strike/Defend starter slots in the obs deck region (obs[19:239],
2 slots per card: even=base, odd=upgraded, counts capped at 7) and cross-validate
the trainer-side deck potential against a direct gc.deck derivation. Runs in the
import context (stable for deck reads, unlike worker mode).
"""
import random
import numpy as np
from sim_worker import SafeStsEnv
import slaythespire  # noqa: F401

DECK0, DECK1 = 19, 239
PHI_CAP = 20.0


def counts_of(obs_raw):
    return np.asarray(obs_raw[DECK0:DECK1], dtype=np.int64)


def phi_from_counts(c, s_strike, s_defend):
    starters = c[s_strike] + c[s_strike + 1] + c[s_defend] + c[s_defend + 1]
    upgrades = int(c[1::2].sum())
    nonstarter = float(c.sum() - starters)
    return -0.3 * float(starters) + min(nonstarter, PHI_CAP) + 0.5 * upgrades


def phi_from_deck(deck):
    """Direct derivation (spec table), excluding the curse term -- curses are
    not observable in the obs encoding (alias to the fallback slot)."""
    nonstarter = 0.0
    phi = 0.0
    curses = 0
    for card in deck:
        if "CURSE" in str(card.type):
            curses += 1
            continue
        if card.is_starter_strike_or_defend:
            phi -= 0.3
        else:
            nonstarter += 1.0
        phi += 0.5 * int(card.upgrade_count)
    return phi + min(nonstarter, PHI_CAP), curses


env = SafeStsEnv(ascension=0)
rng = random.Random(99)
obs, info = env.reset(seed=31337)
c0 = counts_of(obs)
nz = [(int(i), int(c0[i])) for i in np.nonzero(c0)[0]]
print("reset deck slots (slot, count):", nz)
by_count = {cnt: slot for slot, cnt in nz}
s_strike = by_count.get(5)
s_defend = by_count.get(4)
print("strike slot:", s_strike, "defend slot:", s_defend)
assert s_strike is not None and s_defend is not None, "unexpected starter deck"
assert s_strike % 2 == 0 and s_defend % 2 == 0, "starter slots should be base (even)"

games = checked = mismatches = curse_steps = 0
while games < 60:
    mask = info["action_mask"]
    legal = [i for i in range(len(mask)) if mask[i]]
    obs, r, term, trunc, info = env.step(rng.choice(legal))
    if not env.in_combat:
        a = phi_from_counts(counts_of(obs), s_strike, s_defend)
        b, ncurse = phi_from_deck(env.gc.deck)
        if ncurse:
            curse_steps += 1
        elif abs(a - b) > 1e-6:
            mismatches += 1
            if mismatches <= 5:
                print("MISMATCH game %d: obs-phi %.2f vs deck-phi %.2f (deck %d cards)"
                      % (games, a, b, len(env.gc.deck)))
        checked += 1
    if term or trunc:
        games += 1
        obs, info = env.reset()

print("validated %d non-combat steps over %d games | mismatches %d | curse-steps skipped %d"
      % (checked, games, mismatches, curse_steps))
print("PASS" if mismatches == 0 else "FAIL")
