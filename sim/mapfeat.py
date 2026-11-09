"""Visible-map summary features for the out-of-combat candidate scorer (MACRO_MAPFEAT=1).

The current act's map is fully visible in-game (node symbols, edges; '?' stays unknown). It is
rebuilt from the live run seed exactly as the game builds it (Map::fromSeed(seed, asc, act, true));
nothing hidden is read. For a map position (x, y) -- or the act start (y = -1) -- summarise every
path from that node up to the boss:
  for each room type (ELITE, REST, SHOP, EVENT '?', TREASURE, MONSTER): min and max count over paths
  + floors left to the boss row + log2(number of distinct paths)
-> 14 numbers, scaled to ~[0, 1]. The scorer gets them for the position now AND after each option
(after != now only for map moves), i.e. 28 extra inputs.
"""
import functools, math

import slaythespire as S

TYPES = ("ELITE", "REST", "SHOP", "EVENT", "TREASURE", "MONSTER")
N = 2 * len(TYPES) + 2


@functools.lru_cache(maxsize=16)
def _graph(seed, act):
    m = S.SpireMap(int(seed), 0, int(act), True)
    room, edges = {}, {}
    for y in range(15):
        for x in range(7):
            r = str(m.get_room_type(x, y)).split(".")[-1]
            room[(x, y)] = r
            edges[(x, y)] = [x2 for x2 in (x - 1, x, x + 1) if 0 <= x2 <= 6 and y < 14 and m.has_edge(x, y, x2)]
    starts = [x for x in range(7) if m.has_edge(-1, -1, x)]
    return room, edges, starts


@functools.lru_cache(maxsize=4096)
def _dp(seed, act, x, y):
    """-> (min counts tuple, max counts tuple, paths) over paths from (x, y) to the top row (inclusive)."""
    room, edges, _ = _graph(seed, act)
    own = [1 if room[(x, y)] == t else 0 for t in TYPES]
    nxt = edges[(x, y)]
    if not nxt:
        return tuple(own), tuple(own), 1
    subs = [_dp(seed, act, x2, y + 1) for x2 in nxt]
    mn = tuple(own[i] + min(s[0][i] for s in subs) for i in range(len(TYPES)))
    mx = tuple(own[i] + max(s[1][i] for s in subs) for i in range(len(TYPES)))
    return mn, mx, sum(s[2] for s in subs)


def summary(gc, x, y):
    seed, act = int(gc.seed), int(gc.act)
    if act > 3:
        return [0.0] * N
    try:
        if y < 0:
            _, _, starts = _graph(seed, act)
            subs = [_dp(seed, act, sx, 0) for sx in starts]
            mn = [min(s[0][i] for s in subs) for i in range(len(TYPES))]
            mx = [max(s[1][i] for s in subs) for i in range(len(TYPES))]
            paths = sum(s[2] for s in subs)
        else:
            mn, mx, paths = _dp(seed, act, x, y)
        left = 15 - (y + 1)
        return [v / 15.0 for v in mn] + [v / 15.0 for v in mx] + [left / 15.0, math.log2(max(1, paths)) / 12.0]
    except Exception:
        return [0.0] * N
