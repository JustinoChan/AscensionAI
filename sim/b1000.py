"""Combined paired test over eval500 + eval500b: b1000.py ARM_A ARM_B (arm names without the eval500/eval500b prefix)."""
import sqlite3, math, sys
c = sqlite3.connect("sim_logs/results.db")
a, b = sys.argv[1], sys.argv[2]
d = lambda r: dict(c.execute("select seed,win from seeds where run=?", (r,)).fetchall())
x = y = wa = wb = n = 0
for p in ("eval500_", "eval500b_"):
    A, B = d(p + a), d(p + b)
    for s in A:
        if s in B:
            n += 1; wa += A[s]; wb += B[s]; x += A[s] and not B[s]; y += B[s] and not A[s]
k = x + y
pv = min(1, 2 * sum(math.comb(k, i) for i in range(min(x, y) + 1)) / 2 ** k) if k else 1
print("1000-seed paired %s vs %s: n=%d wins %d vs %d, flips %d/%d, McNemar p=%.4f" % (a, b, n, wa, wb, x, y, pv))
