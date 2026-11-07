"""How much of the teacher's within-decision edge is even representable with
the 16-d role features? Any picker is a function of the feature vector, so
candidates with identical features are indistinguishable: the best a
feature-based picker can do is pick the equivalence class with the highest
MEAN label. Reports that oracle ceiling on the same held-out group split used
by train_draft_v_rank.py."""
import json
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "draft_data.jsonl"
rows = [json.loads(l) for l in open(PATH)]
F = np.asarray([r["feat"] for r in rows], np.float32)
y = np.asarray([r["value"] for r in rows], np.float32)

groups, s0 = [], 0
for i in range(1, len(y) + 1):
    if i == len(y) or not np.array_equal(F[i, 11:], F[s0, 11:]):
        if i - s0 >= 2:
            groups.append((s0, i))
        s0 = i
rng = np.random.default_rng(0)
perm = rng.permutation(len(groups))
gva = [groups[k] for k in perm[int(0.85 * len(groups)):]]

orac = tpick = mpick = 0.0
n_alias = n_alias_diff = 0
top1_ceiling = 0.0
for a, b in gva:
    lab = y[a:b]
    classes = {}
    for j in range(a, b):
        classes.setdefault(F[j].tobytes(), []).append(y[j])
    if len(classes) < b - a:
        n_alias += 1
        if any(len(set(v)) > 1 for v in classes.values()):
            n_alias_diff += 1
    best_class = max(np.mean(v) for v in classes.values())
    orac += best_class
    tpick += lab.max()
    mpick += lab.mean()
    # a feature picker lands on the true argmax only if some argmax candidate's
    # class is the best-mean class and the picker wins the within-class tie
    for v in classes.values():
        if np.mean(v) == best_class:
            top1_ceiling += max(1.0 if x == lab.max() else 0.0 for x in v) / len(v)
            break

n = len(gva)
print("held-out groups", n)
print(f"groups w/ aliased candidates {n_alias}/{n}  (aliased AND different labels {n_alias_diff}/{n})")
print(f"feature-oracle edge captured  {(orac - mpick) / max(1e-9, tpick - mpick):.3f}")
print(f"feature-oracle top1 ceiling   {top1_ceiling / n:.3f}")
print(f"(values: oracle {orac/n:.1f}  teacher {tpick/n:.1f}  random {mpick/n:.1f})")
