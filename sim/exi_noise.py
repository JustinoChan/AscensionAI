"""Label-noise ceiling: states labelled twice (EXI_DOUBLE=1) with independent sampled worlds. Reports how often the two
labellings agree on the best action, the mean regret of label-1's best under label-2, and the same for a net's choice."""
import glob, sys, json
import numpy as np
q1, q2, O, NL = [], [], [], []
for f in glob.glob(sys.argv[1] + "/seed_*.npz"):
    if ".tmp." in f:
        continue
    z = np.load(f)
    if "q2" not in z.files:
        continue
    k = z["n_legal"] >= 2
    q1.append(z["q"][k]); q2.append(z["q2"][k]); O.append(z["obs"][k]); NL.append(z["n_legal"][k])
q1, q2, O, NL = np.concatenate(q1), np.concatenate(q2), np.concatenate(O).astype(np.float32), np.concatenate(NL)
legal = np.arange(128)[None] < NL[:, None]
a1 = np.where(legal, np.nan_to_num(q1, nan=-1e9), -1e9).argmax(1); a2 = np.where(legal, np.nan_to_num(q2, nan=-1e9), -1e9).argmax(1)
qq2 = np.where(legal, np.nan_to_num(q2, nan=-1e9), -1e9)
res = {"states": int(len(q1)), "label_self_agree": float((a1 == a2).mean()),
       "label_regret_vs_2nd": float((qq2.max(1) - qq2[np.arange(len(a1)), a1]).mean())}
for net in sys.argv[2:]:
    z = np.load(net); L = [(z["pbody.%d.weight" % i], z["pbody.%d.bias" % i]) for i in (0, 2, 4)] + [(z["pi.weight"], z["pi.bias"])]
    h = O
    for li, (W, b) in enumerate(L):
        x = h @ W.T + b
        h = x if li == len(L) - 1 else 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    an = np.where(legal, h, -1e9).argmax(1)
    res[net] = {"agree_label2": float((an == a2).mean()), "regret_vs_2nd": float((qq2.max(1) - qq2[np.arange(len(an)), an]).mean())}
print(json.dumps(res, indent=1))
