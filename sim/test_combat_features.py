"""In-process: build a known Gremlin Nob fight, drive a few turns, and assert
combat_block is populated, in-range, and self-consistent with the bindings."""
import sys
sys.path.insert(0, ".")
import random
import numpy as np
import slaythespire as sts
from combat_features import combat_block, K


class Dummy:
    """Minimal env shim exposing in_combat + bc (what combat_block reads)."""
    in_combat = True

    def __init__(self, bc):
        self.bc = bc


class NoCombat:
    in_combat = False
    bc = None


# out-of-combat shim returns zeros
assert combat_block(NoCombat()).shape == (K,)
assert not combat_block(NoCombat()).any(), "should be zeros out of combat"

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
bc = sts.BattleContext()
bc.init(gc, sts.MonsterEncounter.GREMLIN_NOB)
env = Dummy(bc)

saw_damage = False
rng = random.Random(0)
for _ in range(40):
    if bc.outcome != sts.BattleOutcome.UNDECIDED:
        break
    blk = combat_block(env)
    assert blk.shape == (K,)
    assert np.isfinite(blk).all()
    assert (blk >= -5).all() and (blk <= 2.01).all(), "out of expected range"
    m = bc.monsters.monsters[0]
    base, hits = m.intent_base_damage(bc)
    if base > 0:
        saw_damage = True
        assert blk[1] > 0, "attack queued but incoming feature is 0"
        assert blk[0] == 1.0, "attack queued but is_attacking feature not set"
    acts = bc.get_possible_actions()
    rng.choice(acts).execute(bc)

assert saw_damage, "never observed an attack to validate damage features"
print("PASS: combat_block populated, in-range, consistent with bindings")
