"""Live check of the intent feature: drive a Gremlin Nob fight and print the
enemy-0 intent slot (obs[448]) each step alongside turn/hp. Nob's script is
known: turn 1 BELLOW (buff), then SKULL_BASH/RUSH attacks -- so we can see
whether the slot (a) changes per turn, (b) leads or lags the move, (c) exceeds
the declared maximum of 20."""
import random
import numpy as np
import slaythespire as sts

INTENT0 = 448  # 9+10+220+178+5 (=422) + block,energy,drawn (3) + 20 powers -> 445 enemies; +3
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc, sts.MonsterEncounter.GREMLIN_NOB)
nn = sts.getNNInterface()
rng = random.Random(3)

last_turn = -1
for step in range(60):
    if bc.outcome != sts.BattleOutcome.UNDECIDED:
        print("battle over:", bc.outcome)
        break
    obs = np.asarray(nn.getObservation(gc, bc), dtype=np.int64)
    mon = bc.monsters.monsters[0]
    if bc.turn != last_turn:
        print("turn %2d | intent slot %4d | enemy hp %3d | player hp %3d block %2d"
              % (bc.turn, obs[INTENT0], mon.cur_hp, bc.player.cur_hp,
                 bc.player.block), flush=True)
        last_turn = bc.turn
    acts = bc.get_possible_actions()
    rng.choice(acts).execute(bc)

print("== declared maxima for enemy slots ==")
mx = np.asarray(nn.getObservationMaximums())
print("intent slot max:", mx[INTENT0], "| enemy block max:", mx[447],
      "| enemy hp max:", mx[445])
