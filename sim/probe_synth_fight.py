"""Make-or-break probe: construct a synthetic fight (chosen encounter, custom
deck, custom HP), play it to completion with random actions, and verify the
observation encodes the deck correctly WITHOUT importing gym_sts_lightspeed.
"""
import random
import sys

import numpy as np
import slaythespire as sts

attempt = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(attempt)

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1234 + attempt, 0)

print("encounter before:", gc.encounter)
try:
    gc.encounter = sts.MonsterEncounter.GREMLIN_NOB
    print("encounter set OK ->", gc.encounter)
except Exception as e:
    print("encounter SET FAILED:", e)
    sys.exit(1)

# Custom deck: add 2 cards, one upgraded
gc.obtain_card(sts.Card(sts.CardId.IRON_WAVE))
up = sts.Card(sts.CardId.SHRUG_IT_OFF)
up.upgrade()
gc.obtain_card(up)
gc.cur_hp = 60
gc.max_hp = 75
print("deck size:", len(gc.deck), "hp:", gc.cur_hp, "/", gc.max_hp)

bc = sts.BattleContext()
bc.init(gc)
mg = bc.monsters
print("monsters type:", type(mg).__name__, "members:",
      [n for n in dir(mg) if not n.startswith("_")])

steps = 0
while bc.outcome == sts.BattleOutcome.UNDECIDED and steps < 1000:
    acts = bc.get_possible_actions()
    if not acts:
        print("NO ACTIONS at step", steps)
        break
    random.choice(acts).execute(bc)
    steps += 1
print("outcome:", bc.outcome, "steps:", steps,
      "player hp:", bc.player.cur_hp, "turn:", bc.turn)

# Observation AFTER battle driving, no gym import anywhere in this process
nn = sts.getNNInterface()
obs = np.array(nn.getObservation(gc, bc), dtype=np.int32)
deck = obs[19:239]
nz = {i: int(v) for i, v in enumerate(deck) if v}
print("deck region nonzero slots:", nz)
print("EXPECT: strike base 132 -> 5, defend base 38 -> 4, bash 6 -> 1, "
      "iron wave base, shrug-it-off UPGRADED (odd slot)")
print("PASS" if nz.get(132) == 5 and nz.get(38) == 4 else "CHECK SLOTS")
