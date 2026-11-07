from run_state_features import vectorize, FEAT_DIM, DECK_KEYS, SCALAR_KEYS

roles = {"size": 20, "starters": 8, "upgraded": 2, "upg_tempo": 1, "damage": 9,
         "block": 5, "aoe": 1, "scaling": 3, "draw_energy": 0, "powers": 3, "junk": 0}
scalars = {"act": 2, "floor": 17, "hp_frac": 0.5, "gold": 120, "relics": 5}
v = vectorize(roles, scalars)
assert v.shape == (FEAT_DIM,), v.shape
assert FEAT_DIM == len(DECK_KEYS) + len(SCALAR_KEYS)
assert abs(v[DECK_KEYS.index("size")] - 20 / 30.0) < 1e-6
assert abs(v[DECK_KEYS.index("starters")] - 8 / 10.0) < 1e-6
assert abs(v[len(DECK_KEYS) + SCALAR_KEYS.index("hp_frac")] - 0.5) < 1e-6
# missing role key -> 0, never crashes
assert abs(vectorize({}, {})[0]) < 1e-9
print("PASS FEAT_DIM=%d" % FEAT_DIM)
