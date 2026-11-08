"""Phase 3 step 3.2: learned out-of-combat policy (honest candidate scorer, see honest.py MACRO_POLICY).

Each decision = a set of legal options, each described by 2416 features
[obs1200(now), obs1200(after option) - obs1200(now), screen one-hot(16)] (after = determinized copy).
Scorer MLP 2416-256-256-1 (GELU-tanh); policy = softmax over the decision's options.

  bc   : imitate mainline_h1's heuristic choices from sim_logs/rl/bcdata (MACRO_FEAT_LOG runs)
         -> nets/rl/pol_bc.npz
  rl   : PPO on the run outcome. Each iteration plays GAMES games on fresh RL seeds
         (600000007 + i*13; disjoint from eval-150, training, holdout, eval500, bc seeds) with
         MACRO_POLICY_SAMPLE=1 and honest C++ K=RL_COMBAT_K combat. Reward = floor/50 + WIN_BONUS*win,
         advantage = batch-normalised reward (same for every decision of a game). Every EVAL_EVERY
         iterations: greedy policy on 100 fixed validation seeds. History -> nets/rl/train_history.json,
         progress bars -> sim_logs/honest/rl_macro.
Run with the training venv: .venv_train/bin/python rl_macro_train.py bc | rl [ITERS]   (rl is resumable)
"""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.expanduser("~/sts"))
OUT = os.environ.get("RL_OUT", "nets/rl")
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
GAMES = int(os.environ.get("RL_GAMES", "100"))
WORKERS = int(os.environ.get("RL_WORKERS", "10"))
COMBAT_K = os.environ.get("RL_COMBAT_K", "8")
WIN_BONUS = float(os.environ.get("RL_WIN_BONUS", "1.0"))
LR = float(os.environ.get("RL_LR", "1e-4"))
CLIP, PPO_EPOCHS, ENT = 0.2, 4, 0.01
EVAL_EVERY = int(os.environ.get("RL_EVAL_EVERY", "10"))
RL_SEED0, RL_STEP = 600000007, 13
VAL_SEEDS = [700000001 + i * 17 for i in range(int(os.environ.get("RL_VAL_N", "100")))]
D_IN = 2416
torch.manual_seed(0)


def scorer():
    return nn.Sequential(nn.Linear(D_IN, 256), nn.GELU(approximate="tanh"), nn.Linear(256, 256),
                         nn.GELU(approximate="tanh"), nn.Linear(256, 1))


def save_npz(m, path):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    np.savez(path + ".tmp.npz", **{"l0.weight": sd["0.weight"], "l0.bias": sd["0.bias"], "l1.weight": sd["2.weight"],
                                   "l1.bias": sd["2.bias"], "l2.weight": sd["4.weight"], "l2.bias": sd["4.bias"]})
    os.replace(path + ".tmp.npz", path)


def load_npz(m, path):
    z = np.load(path)
    for i, li in ((0, 0), (1, 2), (2, 4)):
        m[li].weight.data = torch.tensor(z["l%d.weight" % i]); m[li].bias.data = torch.tensor(z["l%d.bias" % i])
    return m


def load_logs(paths, need_logp=False):
    """-> list of games: (feats [Nc,2416] f16, ncand [D], action [D], logp [D], run [4], ss [D])"""
    G = []
    for p in paths:
        try:
            z = np.load(p)
            if len(z["action"]) == 0:
                continue
            G.append((z["feats"], z["ncand"].astype(np.int64), z["action"].astype(np.int64), z["logp"], z["run"], z["ss"]))
        except Exception:
            pass
    return G


def batch_logp(m, feats, ncand, actions):
    """feats: flat [sum ncand, D] tensor for a minibatch of decisions -> logp(action) per decision, entropy."""
    s = m(feats)[:, 0]
    D = len(ncand); W = int(ncand.max())
    idx = torch.arange(W, device=dev)[None, :].expand(D, W)
    valid = idx < ncand[:, None]
    off = torch.cumsum(ncand, 0) - ncand
    pos = (off[:, None] + idx).clamp(max=len(s) - 1)
    S = torch.where(valid, s[pos], torch.full_like(s[pos], -1e9))
    lp = F.log_softmax(S, 1)
    p = lp.exp()
    ent = -(p * lp.clamp(min=-30) * valid).sum(1)
    return lp.gather(1, actions[:, None])[:, 0], ent


def minibatches(G, dec_per_batch=512, extra=None):
    """yield (feats, ncand, actions, extra_vals) over all decisions of all games, shuffled by decision."""
    items = []  # (game idx, decision idx, cand offset)
    for gi, g in enumerate(G):
        off = np.concatenate([[0], np.cumsum(g[1])[:-1]])
        for di in range(len(g[2])):
            items.append((gi, di, off[di]))
    order = np.random.permutation(len(items))
    for i in range(0, len(order), dec_per_batch):
        sel = [items[j] for j in order[i:i + dec_per_batch]]
        fs = np.concatenate([G[gi][0][o:o + G[gi][1][di]] for gi, di, o in sel]).astype(np.float32)
        nc = np.array([G[gi][1][di] for gi, di, o in sel]); ac = np.array([G[gi][2][di] for gi, di, o in sel])
        ex = None if extra is None else np.array([extra(gi, di) for gi, di, o in sel], np.float32)
        yield (torch.tensor(fs, device=dev), torch.tensor(nc, device=dev), torch.tensor(ac, device=dev),
               None if ex is None else torch.tensor(ex, device=dev), [G[gi][5][di] for gi, di, o in sel])


# ------------------------------------------------------------------ BC
def bc():
    files = sorted(glob.glob(os.environ.get("BCDATA","sim_logs/rl/bcdata")+"/seed_*.npz"))
    G = load_logs(files)
    nval = max(1, len(G) // 10)
    tr, va = G[:-nval], G[-nval:]
    print("BC games train %d val %d, decisions %d / %d" % (len(tr), len(va), sum(len(g[2]) for g in tr), sum(len(g[2]) for g in va)), flush=True)
    m = scorer().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    hist = {"bc": []}

    def evaluate():
        m.eval(); ok, n, by = 0, 0, {}
        with torch.no_grad():
            for fs, nc, ac, _, ss in minibatches(va, 2048):
                s = m(fs)[:, 0]
                off = torch.cumsum(nc, 0) - nc
                for k in range(len(nc)):
                    pred = int(torch.argmax(s[off[k]:off[k] + nc[k]]))
                    hit = int(pred == int(ac[k])); ok += hit; n += 1
                    t = by.setdefault(ss[k], [0, 0]); t[0] += hit; t[1] += 1
        return ok / max(1, n), {k: round(v[0] / v[1], 3) for k, v in sorted(by.items())}
    best, best_state = -1, None
    for ep in range(12):
        m.train()
        for fs, nc, ac, _, _ in minibatches(tr, 512):
            lp, _ = batch_logp(m, fs, nc, ac)
            loss = -lp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        a, by = evaluate()
        hist["bc"].append({"epoch": ep, "loss": loss.item(), "val_acc": a, "by_screen": by})
        print("bc ep %d loss %.4f val acc %.4f %s" % (ep, loss.item(), a, by), flush=True)
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    save_npz(m, OUT + "/pol_bc.npz")
    json.dump(hist, open(OUT + "/bc_history.json", "w"), indent=1)
    print("saved %s/pol_bc.npz, best val acc %.4f" % (OUT, best))


# ------------------------------------------------------------------ RL
def play(policy, seeds, outdir, sample, tag):
    os.makedirs(outdir, exist_ok=True)
    prog = os.environ.get("RL_PROG", "sim_logs/honest/rl_macro")
    env = "HONEST=1 HONEST_CPP=1 HONEST_K=%s MACRO_POLICY=%s MACRO_POLICY_LOG=%s%s" % (
        COMBAT_K, policy, outdir, " MACRO_POLICY_SAMPLE=1" if sample else "")
    script = ("s=$1; [ -s {d}/seed_$s.npz ] && exit 0; env {env} nice -n 5 .venv/bin/python eval_fullrun_analytics_v2.py $s macro "
              "> {d}/game_$s.json 2>/dev/null && cp {d}/game_$s.json {p}/seed_{t}_$s.json; exit 0").format(d=outdir, env=env, p=prog, t=tag)
    p = subprocess.Popen(["xargs", "-P", str(WORKERS), "-n", "1", "bash", "-c", script, "_"], stdin=subprocess.PIPE, text=True)
    p.communicate("\n".join(map(str, seeds)) + "\n")
    return [q for q in (os.path.join(outdir, "seed_%d.npz" % s) for s in seeds) if os.path.exists(q)]


def evaluate_policy(policy, tag):
    d = os.environ.get("RL_TMP", "sim_logs/rl") + "/val_%s" % tag
    snap = OUT + "/val_%s.npz" % tag
    shutil.copy(policy, snap)
    G = load_logs(play(snap, VAL_SEEDS, d, False, "val_" + tag))
    fl = [int(g[4][1]) for g in G]; w = [int(g[4][2]) for g in G]
    rec = {"tag": tag, "n": len(fl), "floor": float(np.mean(fl)) if fl else None, "wins": int(sum(w)), "t": time.time()}
    print("VAL", json.dumps(rec), flush=True)
    return rec


def rl(iters):
    hist_p = OUT + "/train_history.json"
    hist = json.load(open(hist_p)) if os.path.exists(hist_p) else {"rl": [], "rl_val": []}
    start = len(hist["rl"])
    cur = OUT + "/pol_latest.npz"
    if not os.path.exists(cur):
        shutil.copy(OUT + "/pol_bc.npz", cur)
    m = load_npz(scorer(), cur).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    prog = os.environ.get("RL_PROG", "sim_logs/honest/rl_macro")
    os.makedirs(prog, exist_ok=True)
    json.dump({"name": "rl_macro", "description": "Phase 3 3.2: RL out-of-combat policy (PPO on run outcome, K=%s combat)" % COMBAT_K,
               "group": "rl", "total": iters * GAMES, "parallel": WORKERS, "env": "rl_macro_train.py",
               "seed_set": "%d + i*%d" % (RL_SEED0, RL_STEP), "started_at": time.time(), "log": "nets/rl/rl.log",
               "out_dir": prog, "no_ledger": True}, open(prog + "/RUN.json", "w"), indent=1)
    if not hist["rl_val"]:
        hist["rl_val"].append(evaluate_policy(OUT + "/pol_bc.npz", "bc"))
        json.dump(hist, open(hist_p, "w"))
    for it in range(start, iters):
        t0 = time.time()
        seeds = [RL_SEED0 + (it * GAMES + j) * RL_STEP for j in range(GAMES)]
        snap = OUT + "/pol_it%03d.npz" % it
        if not os.path.exists(snap):
            shutil.copy(cur, snap)
        G = load_logs(play(snap, seeds, os.environ.get("RL_TMP", "sim_logs/rl") + "/it%03d" % it, True, "it%03d" % it))
        if len(G) < GAMES // 2:
            print("iteration %d: only %d games, stopping" % (it, len(G)), flush=True); break
        R = np.array([g[4][1] / 50.0 + WIN_BONUS * g[4][2] for g in G])
        mu, sd = float(R.mean()), float(R.std() + 1e-6)
        adv = (R - mu) / sd
        m.train()
        for _ in range(PPO_EPOCHS):
            for fs, nc, ac, ex, _ in minibatches(G, 512, extra=lambda gi, di: (adv[gi], G[gi][3][di])):
                A, LP = ex[:, 0], ex[:, 1]
                lp, ent = batch_logp(m, fs, nc, ac)
                ratio = torch.exp(lp - LP)
                loss = -torch.min(ratio * A, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A).mean() - ENT * ent.mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        save_npz(m, cur)
        rec = {"iter": it, "games": (it + 1) * GAMES, "n": len(G), "mean_R": mu, "win_rate": float(np.mean([g[4][2] for g in G])),
               "floor": float(np.mean([g[4][1] for g in G])), "decisions": int(sum(len(g[2]) for g in G)),
               "entropy": float(ent.mean()), "secs": round(time.time() - t0, 1)}
        hist["rl"].append(rec)
        print(json.dumps(rec), flush=True)
        if (it + 1) % EVAL_EVERY == 0:
            hist["rl_val"].append(evaluate_policy(cur, "it%03d" % it))
        json.dump(hist, open(hist_p + ".tmp", "w")); os.replace(hist_p + ".tmp", hist_p)
        if it % 10:
            shutil.rmtree("sim_logs/rl/it%03d" % it, ignore_errors=True)
    open(prog + "/.end", "w").write(str(int(time.time())))


if __name__ == "__main__":
    if sys.argv[1] == "bc":
        bc()
    elif sys.argv[1] == "rl":
        rl(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
