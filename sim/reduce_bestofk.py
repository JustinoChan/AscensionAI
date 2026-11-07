import json, sys, glob
arm = sys.argv[1]
best = {}
for f in glob.glob(f'sim_logs/bok_parts_{arm}/*.json'):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    s = r['seed']
    key = (int(r.get('win', 0)), r.get('floor', 0))
    if s not in best or key > (int(best[s].get('win', 0)), best[s].get('floor', 0)):
        best[s] = r
rows = list(best.values())
open(f'sim_logs/oracle_{arm}.jsonl', 'w').write(
    ''.join(json.dumps(r) + '\n' for r in rows))
wins = sum(int(r.get('win', 0)) for r in rows)
floor = sum(r.get('floor', 0) for r in rows) / max(1, len(rows))
print(f'{arm}(best-of-K) n={len(rows)} wins={wins} avg_floor={floor:.2f}')
