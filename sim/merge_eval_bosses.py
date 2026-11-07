import json
existing = [json.loads(l) for l in open("bank_eval.jsonl") if l.strip()]
have_boss = sum(1 for e in existing if e.get("room") == "BOSS")
new = [json.loads(l) for l in open("harvest_bosses.jsonl") if l.strip()]
new_boss = [e for e in new if e.get("room") == "BOSS"]
seen = {(e["seed"], tuple(e["prefix"])) for e in existing}
add, want = [], max(0, 48 - have_boss)
for e in new_boss:
    k = (e["seed"], tuple(e["prefix"]))
    if k not in seen:
        seen.add(k); add.append(e)
    if len(add) >= want:
        break
with open("bank_eval.jsonl", "a") as f:
    for e in add:
        f.write(json.dumps(e) + "\n")
total_boss = have_boss + len(add)
print("had %d bosses, added %d, total %d" % (have_boss, len(add), total_boss))
assert total_boss >= 48, "not enough boss entries harvested; widen seed range"
