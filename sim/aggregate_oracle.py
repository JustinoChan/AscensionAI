import json, sys
arm = sys.argv[1]
rows = [json.loads(l) for l in open(f'sim_logs/oracle_{arm}.jsonl')]
wins = sum(int(r.get('win', 0)) for r in rows)
floor = sum(r.get('floor', 0) for r in rows) / max(1, len(rows))
act2 = sum(int(r.get('act', 1) >= 2) for r in rows) / max(1, len(rows))
print(f"{arm:9s} n={len(rows):3d} wins={wins:3d} avg_floor={floor:.2f} act2={100*act2:.0f}%")
