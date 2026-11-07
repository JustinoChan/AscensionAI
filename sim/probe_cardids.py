import warnings, os
warnings.filterwarnings("ignore")
import slaythespire
ids = [x for x in dir(slaythespire.CardId) if not x.startswith("_")
       and x not in ("name", "value")]
print("CardId count:", len(ids))
print(ids)
print("\nCardType:", [x for x in dir(slaythespire.CardType) if not x.startswith("_")])
print("CardRarity:", [x for x in dir(slaythespire.CardRarity) if not x.startswith("_")])
# how does str(card.id) render?
import gym_sts_lightspeed  # noqa
from sim_worker_fight import FightEnv
import numpy as np
e = FightEnv(ascension=0)
e.reset(seed=10007)
try:
    c = e.gc.deck[0]
    print("sample card: repr=%r id=%r str_id=%s name=%s" % (
        c, c.id, str(c.id), str(c.id).split(".")[-1]))
except Exception as ex:
    print("deck read:", ex)
