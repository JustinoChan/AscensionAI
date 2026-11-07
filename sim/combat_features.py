"""Shared, torch-free combat-feature block appended to the 1200-d obs.
combat_block(env) -> np.float32[35]: zeros when not in combat. Damage is
adjusted in Python from raw base damage + strength/weak/vulnerable (STS order:
(base+strength) -> *0.75 if attacker weak -> *1.5 if defender vulnerable).

Layout (K=35):
  per enemy e in 0..4 (base = e*6):
    [+0] is_attacking            [+1] total_incoming/40 (clip 0..2)
    [+2] num_hits/8              [+3] enemy_vulnerable/10
    [+4] enemy_weak/10           [+5] enemy_strength/10 (signed)
  [30] player_vulnerable/10  [31] player_weak/10  [32] player_strength/10
  [33] total_incoming_all/40 (clip 0..2)
  [34] net_incoming_after_block/40 (clip 0..2)
"""
import numpy as np

K = 35
_DMG_NORM = 40.0
_HIT_NORM = 8.0
_STATUS_NORM = 10.0
_CLIP = 2.0


def _adjusted_per_hit(base, attacker_strength, attacker_weak, defender_vuln):
    d = base + attacker_strength
    if attacker_weak > 0:
        d = int(d * 0.75)
    if defender_vuln > 0:
        d = int(d * 1.5)
    return max(0, d)


def combat_block(env):
    out = np.zeros(K, dtype=np.float32)
    if not getattr(env, "in_combat", False) or getattr(env, "bc", None) is None:
        return out
    try:
        bc = env.bc
        player = bc.player
        p_vuln = player.vulnerable()
        p_weak = player.weak()
        p_str = player.strength()
        p_block = player.block
        mons = bc.monsters.monsters
        total_raw = 0
        for e in range(min(5, len(mons))):
            m = mons[e]
            if not m.is_alive():
                continue
            base, hits = m.intent_base_damage(bc)
            e_str, e_weak, e_vuln = m.strength(), m.weak(), m.vulnerable()
            per_hit = _adjusted_per_hit(base, e_str, e_weak, p_vuln)
            total_e = per_hit * hits
            total_raw += total_e
            b = e * 6
            out[b + 0] = 1.0 if m.is_attacking() else 0.0
            out[b + 1] = min(total_e / _DMG_NORM, _CLIP)
            out[b + 2] = hits / _HIT_NORM
            out[b + 3] = e_vuln / _STATUS_NORM
            out[b + 4] = e_weak / _STATUS_NORM
            out[b + 5] = e_str / _STATUS_NORM
        out[30] = p_vuln / _STATUS_NORM
        out[31] = p_weak / _STATUS_NORM
        out[32] = p_str / _STATUS_NORM
        out[33] = min(total_raw / _DMG_NORM, _CLIP)
        out[34] = min(max(0, total_raw - p_block) / _DMG_NORM, _CLIP)
    except Exception:
        return np.zeros(K, dtype=np.float32)
    return out
