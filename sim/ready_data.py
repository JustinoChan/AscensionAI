"""Boss-readiness dataset from PPO decision logs (honest: states are the public obs the policy saw).
For each logged game and each act a whose boss the run REACHED (floor >= boss floor), take up to PER states from that
act's out-of-combat decisions (always incl. the last one before the boss) with label = beat the act-a boss.
  .venv/bin/python ready_data.py OUT.npz DIR [DIR ...]
"""
import glob, sys, zlib
import numpy as np
from multiprocessing import Pool

BOSS_FLOOR = {1: 16, 2: 33, 3: 50}
PER = 4


def one(p):
    try:
        z = np.load(p)
        nc = z["ncand"].astype(np.int64)
        if len(nc) == 0:
            return None
        off = np.concatenate([[0], np.cumsum(nc)[:-1]])
        S = z["feats"][off][:, :1200]
        fl, ac, run = z["floor"], z["act"], z["run"]
        seed, rfloor, win = int(run[0]), int(run[1]), int(run[2])
        X, Y, A, F, L = [], [], [], [], []
        rng = np.random.default_rng(seed)
        for a, bf in BOSS_FLOOR.items():
            if rfloor < bf and not win:
                continue
            idx = np.where((ac == a) & (fl < bf))[0]
            if len(idx) == 0:
                continue
            last = idx[np.argmax(fl[idx])]
            pick = set(rng.choice(idx, min(PER, len(idx)), replace=False).tolist()) | {int(last)}
            y = int(win) if a == 3 else int(rfloor > bf)
            for i in sorted(pick):
                X.append(S[i]); Y.append(y); A.append(a); F.append(int(fl[i])); L.append(int(i == last))
        if not X:
            return None
        return np.array(X, np.float16), np.array(Y, np.int8), np.array(A, np.int8), np.array(F, np.int16), np.array(L, np.int8), np.full(len(X), seed, np.int64)
    except Exception:
        return None


if __name__ == "__main__":
    out, dirs = sys.argv[1], sys.argv[2:]
    paths = [p for d in dirs for p in glob.glob(d + "/it*/seed_*.npz")]
    print(len(paths), "games", flush=True)
    with Pool(3) as pool:
        R = [r for r in pool.imap_unordered(one, paths, chunksize=64) if r is not None]
    cols = [np.concatenate([r[k] for r in R]) for k in range(6)]
    np.savez(out, X=cols[0], y=cols[1], act=cols[2], floor=cols[3], last=cols[4], seed=cols[5])
    for a in (1, 2, 3):
        m = cols[2] == a
        print("act %d: %d states, %d last-before-boss, beat rate %.3f" % (a, m.sum(), (m & (cols[4] == 1)).sum(), cols[1][m & (cols[4] == 1)].mean()))
