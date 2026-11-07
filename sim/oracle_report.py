import json


def _rows(arm):
    try:
        return [json.loads(l) for l in open(f'sim_logs/oracle_{arm}.jsonl')]
    except FileNotFoundError:
        return None


def wins(arm):
    r = _rows(arm)
    return None if r is None else sum(int(x.get('win', 0)) for x in r)


def floor(arm):
    r = _rows(arm)
    return None if r is None else sum(x.get('floor', 0) for x in r) / max(1, len(r))


base = wins('none')
arms = [('combat', 'combat (2-ply search)'),
        ('path', 'path/rest (best-of-8 routing)'),
        ('deck', 'deck (premium upgraded shell)'),
        ('combined', 'combined (deck + path)')]
L = ['ORACLE CEILING SWEEP  (150 fixed seeds; mainline_v3 baseline)',
     'baseline (none): %d wins  avg_floor %.2f' % (base, floor('none')), '']
for arm, label in arms:
    w = wins(arm)
    if w is None:
        L.append('%-30s  (not run)' % label)
    else:
        L.append('%-30s  %3d wins  floor %.2f  lift %+d'
                 % (label, w, floor(arm), w - base))
L += ['', '--- loss taxonomy ---', open('sim_logs/loss_taxonomy.txt').read(),
      'DECISION RULE',
      '  combat lift <= 0      -> deeper search is NOT the route to 100%.',
      '  path & deck lifts huge -> HP-routing and deck-quality are the levers.',
      '  combined ceiling       -> the fraction of ~100% reachable by stacking them.',
      '',
      'Oracles are UPPER BOUNDS: best-of-8 keeps the best of 8 random routes per',
      'seed; the deck oracle injects an ideal shell. A single deployable learned',
      'policy captures a FRACTION of each ceiling, not all of it. Per the design,',
      'the next build is a map-aware / multi-step value for routing (1-ply lookahead',
      'is too weak for pathing), with deck-quality as the stacked second lever.']
out = '\n'.join(L)
print(out)
open('sim_logs/oracle_ceiling_report.txt', 'w').write(out + '\n')
