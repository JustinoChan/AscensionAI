import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import gym_sts_lightspeed  # noqa
import slaythespire
from sim_worker_fight import FightEnv
env = FightEnv(ascension=0)
env.reset(seed=10007)
gc = env.gc
print("obtain_card:", hasattr(gc, 'obtain_card'), "remove_card:", hasattr(gc, 'remove_card'))
deck = list(gc.deck)
print("deck size:", len(deck))
c0 = deck[0]
print("card attrs:", [a for a in dir(c0) if not a.startswith('_')])
print("card0:", str(c0.id), str(c0.type), "upgraded", c0.upgraded)
print("module card-ish names:", [a for a in dir(slaythespire) if ('ard' in a or 'reate' in a or a == 'Card')])
# CardId + construction
try:
    cid = slaythespire.CardId.POMMEL_STRIKE
    print("CardId.POMMEL_STRIKE:", cid)
    try:
        card = slaythespire.Card(cid)
        print("Card(cid):", str(card.id), "upgraded", card.upgraded)
        try:
            card.upgrade()
            print("card.upgrade() -> upgraded", card.upgraded)
        except Exception as e:
            print("card.upgrade FAIL:", repr(e))
    except Exception as e:
        print("Card(cid) FAIL:", repr(e))
except Exception as e:
    print("CardId FAIL:", repr(e))
# mutate scratch deck
try:
    n0 = len(list(gc.deck))
    gc.remove_card(0)
    print("remove_card(0): %d -> %d" % (n0, len(list(gc.deck))))
except Exception as e:
    print("remove_card FAIL:", repr(e))
try:
    n1 = len(list(gc.deck))
    gc.obtain_card(slaythespire.Card(slaythespire.CardId.INFLAME))
    print("obtain_card(INFLAME): %d -> %d" % (n1, len(list(gc.deck))))
except Exception as e:
    print("obtain_card FAIL:", repr(e))
