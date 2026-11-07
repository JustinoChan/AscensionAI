"""Split bank_raw.jsonl into a stratified held-out eval slice and the train
bank. Eval: up to 16 BOSS + 16 ELITE + 32 MONSTER; EVENT fights excluded
from eval (and weighted 0 in training by fight_config.json)."""
import json
import random
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "bank_raw.jsonl"
entries = [json.loads(ln) for ln in open(src) if ln.strip()]
rng = random.Random(0)
rng.shuffle(entries)
by_room = {}
for e in entries:
    by_room.setdefault(e["room"], []).append(e)
eval_set = []
for room, quota in (("BOSS", 16), ("ELITE", 16), ("MONSTER", 32)):
    eval_set.extend(by_room.get(room, [])[:quota])
eval_ids = {id(e) for e in eval_set}
train = [e for e in entries if id(e) not in eval_ids]
with open("bank_eval.jsonl", "w") as f:
    for e in eval_set:
        f.write(json.dumps(e) + "\n")
with open("bank_train.jsonl", "w") as f:
    for e in train:
        f.write(json.dumps(e) + "\n")


def counts(rows):
    c = {}
    for e in rows:
        c[e["room"]] = c.get(e["room"], 0) + 1
    return c


print("eval :", len(eval_set), counts(eval_set))
print("train:", len(train), counts(train))
print("lost fights in train:", sum(1 for e in train if e["lost"]))
