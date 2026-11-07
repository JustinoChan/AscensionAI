"""Diagnose the Stage B failure: how does V rank candidates WITHIN a draft
decision, compared to the rollout teacher's labels?

Reconstructs decision groups from draft_data.jsonl (records were appended per
candidate in order; a group = consecutive records with identical scalar dims
11..15). Reports top-1 agreement, skip rates, take-type preferences, and value
captured, overall and on win-flip decisions.
"""
import json
import sys

import numpy as np

from np_draft_v import NpDraftV

PATH = sys.argv[1] if len(sys.argv) > 1 else "draft_data.jsonl"
NPZ = sys.argv[2] if len(sys.argv) > 2 else "draft_v.npz"

feats, vals = [], []
with open(PATH) as f:
    for line in f:
        r = json.loads(line)
        feats.append(r["feat"])
        vals.append(r["value"])
F = np.asarray(feats, np.float32)
V = np.asarray(vals, np.float32)
print("records", len(V))

# group boundaries: scalar block change
groups, start = [], 0
for i in range(1, len(V) + 1):
    if i == len(V) or not np.array_equal(F[i, 11:], F[start, 11:]):
        groups.append((start, i))
        start = i
groups = [(a, b) for a, b in groups if b - a >= 2]
print("groups(>=2 cands)", len(groups),
      "avg_size", round(np.mean([b - a for a, b in groups]), 2))

model = NpDraftV(NPZ)
pred = np.array([model.value(x) for x in F], np.float32)

# per-take role classification relative to the skip candidate (min size dim)
ROLE_DIMS = {4: "attack", 5: "block", 6: "aoe", 7: "scaling",
             8: "draw_energy", 9: "power", 10: "junk"}

agree = agree_rand = 0.0
n_var = n_flip = 0
agree_flip = agree_flip_rand = 0.0
v_pick_val, t_pick_val, mean_val = [], [], []
v_skip = t_skip = n_with_skip = 0
v_take_roles, t_take_roles = {}, {}
corrs = []
dv_take_minus_skip = []
for a, b in groups:
    lab = V[a:b]
    prd = pred[a:b]
    n = b - a
    t_best = set(np.flatnonzero(lab == lab.max()))
    v_best = int(np.argmax(prd))
    agree += 1.0 if v_best in t_best else 0.0
    agree_rand += len(t_best) / n
    spread = float(lab.max() - lab.min())
    if spread > 0:
        n_var += 1
        if len(set(lab)) > 1 and len(set(prd)) > 1:
            corrs.append(np.corrcoef(lab, prd)[0, 1])
    if spread >= 500:
        n_flip += 1
        agree_flip += 1.0 if v_best in t_best else 0.0
        agree_flip_rand += len(t_best) / n
    v_pick_val.append(float(lab[v_best]))
    t_pick_val.append(float(lab.max()))
    mean_val.append(float(lab.mean()))
    # skip candidate = unique min of size dim
    sizes = F[a:b, 0]
    smin = np.flatnonzero(sizes == sizes.min())
    if len(smin) == 1 and sizes.max() > sizes.min():
        n_with_skip += 1
        sk = int(smin[0])
        v_skip += 1 if v_best == sk else 0
        t_skip += 1 if sk in t_best and len(t_best) < n else 0
        dv_take_minus_skip.extend(
            float(prd[j] - prd[sk]) for j in range(n) if j != sk)
        def roles_of(j):
            d = F[a + j, :11] - F[a + sk, :11]
            return [nm for dim, nm in ROLE_DIMS.items() if d[dim] > 1e-6] or ["plain_attack_or_skill"]
        if v_best != sk:
            for nm in roles_of(v_best):
                v_take_roles[nm] = v_take_roles.get(nm, 0) + 1
        tb = [j for j in t_best if j != sk]
        if tb:
            for nm in roles_of(tb[0]):
                t_take_roles[nm] = t_take_roles.get(nm, 0) + 1

g = len(groups)
print(f"top1 agreement          {agree/g:.3f}  (random {agree_rand/g:.3f})")
print(f"groups with label var   {n_var}/{g}")
print(f"win-flip groups (>=500) {n_flip}/{g}  agreement {agree_flip/max(1,n_flip):.3f} (random {agree_flip_rand/max(1,n_flip):.3f})")
print(f"within-group corr(V,label) mean {np.mean(corrs):.3f} over {len(corrs)} groups")
print(f"value captured: V-pick {np.mean(v_pick_val):.1f}  teacher {np.mean(t_pick_val):.1f}  random {np.mean(mean_val):.1f}")
print(f"skip rate: V {v_skip}/{n_with_skip}={v_skip/max(1,n_with_skip):.3f}  teacher(strict-best) {t_skip}/{n_with_skip}={t_skip/max(1,n_with_skip):.3f}")
print(f"mean V(take)-V(skip) raw units: {np.mean(dv_take_minus_skip):.2f}")
print("V take-pick roles:      ", dict(sorted(v_take_roles.items(), key=lambda x: -x[1])))
print("teacher take-pick roles:", dict(sorted(t_take_roles.items(), key=lambda x: -x[1])))
