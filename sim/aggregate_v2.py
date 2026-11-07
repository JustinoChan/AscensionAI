import sys, json
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else "v2_decisions.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
n = len(rows)
def avg(k):
    return np.mean([r[k] for r in rows]) if rows else 0.0
print("==== %s (%d runs, mode=%s) ====" % (path, n, rows[0].get("mode", "?") if rows else "?"))
print("avg floor %.1f | wins %d" % (avg("floor"), sum(r["win"] for r in rows)))
taken = sum(r["cards_taken"] for r in rows); skipped = sum(r["cards_skipped"] for r in rows)
print("\n-- DRAFTING -- taken/run %.1f | removes/run %.2f | pick-rate %.0f%%"
      % (avg("cards_taken"), avg("removes"),
         100.0 * taken / max(1, taken + skipped)))
print("-- SHOPS -- shops/run %.2f | gold spent/run %.0f | shop cards/run %.2f | relics BOUGHT/run %.2f"
      % (avg("shops"), avg("gold_spent"), avg("shop_cards"), avg("shop_relics")))
cf = sum(r["campfires"] for r in rows); rest = sum(r["rests"] for r in rows); smith = sum(r["smiths"] for r in rows)
print("-- CAMPFIRES -- %d total | rests %d | smiths %d" % (cf, rest, smith))
print("-- elites/run %.2f | elites avoided/run %.2f | relics/run %.2f | end gold %.0f | end max_hp %.0f"
      % (avg("elites"), avg("elites_avoided"), avg("relics"), avg("gold"), avg("max_hp")))
def hp_per(room):
    tot = sum(r.get("hp_lost_" + room, 0) for r in rows)
    cnt = sum(r.get("n_" + room, 0) for r in rows)
    return (tot / cnt) if cnt else 0.0, cnt
cf_hp = sum(r.get("campfire_hp_sum", 0) for r in rows)
cf_cnt = sum(r.get("campfire_n", 0) for r in rows)
mhp, mc = hp_per("monster"); ehp, ec = hp_per("elite"); bhp, bc2 = hp_per("boss")
print("-- HP BLEED (avg HP lost per SURVIVED fight) -- monster %.1f (n%d) | elite %.1f (n%d) | boss %.1f (n%d)"
      % (mhp, mc, ehp, ec, bhp, bc2))
print("-- avg HP entering campfire: %.0f%% (n%d)" % (100.0 * cf_hp / max(1, cf_cnt), cf_cnt))
be_hp = sum(r.get("boss_entry_hp_sum", 0) for r in rows)
be_n = sum(r.get("boss_entry_n", 0) for r in rows)
print("-- avg HP entering boss: %.0f%% (n%d)" % (100.0 * be_hp / max(1, be_n), be_n))
dk = [r["final_deck"] for r in rows if r["final_deck"].get("size", -1) > 0]
if dk:
    print("-- FINAL DECK -- size %.1f | strikes %.1f | defends %.1f | attacks %.1f | powers %.1f | upgraded %.1f"
          % (np.mean([d["size"] for d in dk]), np.mean([d["strikes"] for d in dk]),
             np.mean([d["defends"] for d in dk]), np.mean([d["attacks"] for d in dk]),
             np.mean([d["powers"] for d in dk]), np.mean([d["upgraded"] for d in dk])))
losses = [r for r in rows if not r["win"]]
print("\n-- DEATHS by act --")
for a in (1, 2, 3):
    da = [r for r in losses if r.get("death_act") == a]
    if da:
        rooms = {x: sum(1 for r in da if r.get("death_room") == x) for x in set(r.get("death_room") for r in da)}
        print("  act %d: %2d  %s" % (a, len(da), rooms))
print("-- DEATHS by room --")
for rt in ("MONSTER", "ELITE", "BOSS"):
    dr = [r for r in losses if r.get("death_room") == rt]
    if dr:
        print("  %-7s %2d  avg enemy_hp_frac_left %.2f" % (rt, len(dr), np.mean([r["enemy_hp_frac"] for r in dr])))
a1b = [r for r in losses if r.get("death_act") == 1 and r.get("death_room") == "BOSS"]
dks = [r["final_deck"] for r in a1b if r["final_deck"].get("size", -1) > 0]
if dks:
    print("-- ACT1-BOSS deaths (%d) deck shape: size %.1f attacks %.1f strikes %.1f powers %.1f upgraded %.1f --"
          % (len(a1b), np.mean([d["size"] for d in dks]), np.mean([d["attacks"] for d in dks]),
             np.mean([d["strikes"] for d in dks]), np.mean([d["powers"] for d in dks]),
             np.mean([d["upgraded"] for d in dks])))
