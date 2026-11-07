"""Conditional loss taxonomy for the mainline_v3 150-seed eval.
Buckets every loss by death location, refining Act-2 monster deaths by whether
the run was outpaced (enemy still healthy) vs threw a winnable fight, and
summarises the Act-2-entry deck deficit for the dominant bucket.
Pure aggregation over the existing per-seed JSONL -- no re-run."""
import json, sys, statistics
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else 'v2_mainline_v3.jsonl'
ROWS = [json.loads(l) for l in open(PATH)]
WINS = sum(int(r.get('win', 0)) for r in ROWS)
LOSSES = [r for r in ROWS if not int(r.get('win', 0))]

# enemy HP fraction still alive at the death turn; > this = clearly outpaced.
OUTPACED = 0.25


def bucket(r):
    a = r.get('death_act')
    rm = (r.get('death_room') or '').upper()
    ehp = r.get('enemy_hp_frac', 0.0)
    if a == 1 and rm == 'BOSS':
        return 'A1_boss_underprepared'
    if a == 1 and rm == 'ELITE':
        return 'A1_elite_overrisk'
    if a == 1:
        return 'A1_other_%s' % rm
    if a == 2 and rm == 'MONSTER':
        return ('A2_monster_attrition_outpaced' if ehp > OUTPACED
                else 'A2_monster_thrown_winnable')
    if a == 2 and rm == 'ELITE':
        return 'A2_elite'
    if a == 2 and rm == 'BOSS':
        return 'A2_boss_scaling'
    if a == 2:
        return 'A2_other_%s' % rm
    if a == 3:
        return 'A3_%s' % rm.lower()
    return 'other_act%s_%s' % (a, rm)


counts = Counter(bucket(r) for r in LOSSES)
total = sum(counts.values())

out = []
out.append('LOSS TAXONOMY  (mainline_v3, 150 fixed seeds)')
out.append('runs %d  wins %d  losses %d' % (len(ROWS), WINS, len(LOSSES)))
out.append('')
for name, n in counts.most_common():
    out.append('%4d  %s' % (n, name))
out.append('')
out.append('sum check: %d == %d : %s' % (total, len(LOSSES), total == len(LOSSES)))

# Deck-deficit profile of the dominant Act-2 monster bucket (the lever target).
a2m = [r for r in LOSSES if r.get('death_act') == 2
       and (r.get('death_room') or '').upper() == 'MONSTER']
decks = [r.get('act2_deck') for r in a2m if r.get('act2_deck')]
if decks:
    def med(key):
        return round(statistics.median(d.get(key, 0) for d in decks), 1)
    out.append('')
    out.append('--- Act-2-monster losses: deck @ Act-2 entry (n=%d) ---' % len(decks))
    out.append('median size %s  starters %s  upgraded %s  upg_tempo %s'
               % (med('size'), med('starters'), med('upgraded'), med('upg_tempo')))
    out.append('median aoe %s  draw_energy %s  powers %s  scaling %s'
               % (med('aoe'), med('draw_energy'), med('powers'), med('scaling')))
    out.append('median enemy_hp_frac @ death %s'
               % round(statistics.median(r.get('enemy_hp_frac', 0) for r in a2m), 2))

text = '\n'.join(out)
print(text)
open('sim_logs/loss_taxonomy.txt', 'w').write(text + '\n')
