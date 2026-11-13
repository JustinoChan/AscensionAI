"""Boss-fight combat RL learner (GPU). See boss_rl.py for the honest data collection.

  .venv_train/bin/python boss_rl_train.py [ITERS]
Each iteration: BOSS_WORKERS processes x BOSS_SEEDS_PER seeds of fresh training seeds (Act 2 and Act 3 bosses,
alternating), BOSS_EPS determinized episodes per boss start with the current combat net sampling; PPO update with
a GROUP-RELATIVE advantage (reward - mean reward of the same start, / its std). Every VAL_EVERY iterations:
GREEDY policy on fixed held-out validation seeds (separate seed range), BOSS_EPS determinized episodes each,
win rate + mean reward, compared with varN on the same starts.
Writes nets/boss1/{combat_latest.npz, combat_itNNN.npz, history.json}; varN layout, so honest search can use it
via HONEST_ROLLOUT_NET (off by default).
"""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.expanduser("~/sts"))
OUT = os.environ.get("BOSS_OUT", "nets/boss1"); os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get("BOSS_TMP", "sim_logs/boss1"); os.makedirs(TMP, exist_ok=True)
W = int(os.environ.get("BOSS_WORKERS", "4")); PER = int(os.environ.get("BOSS_SEEDS_PER", "4"))
EPS = int(os.environ.get("BOSS_EPS", "16")); VAL_EVERY = int(os.environ.get("BOSS_VAL_EVERY", "10"))
SEED0 = int(os.environ.get("BOSS_SEED0", "910000019")); VAL0, VAL_N = 960000011, int(os.environ.get("BOSS_VAL_N", "40"))
LR, CLIP, EPOCHS, ENT = float(os.environ.get("BOSS_LR", "3e-5")), 0.2, 4, 0.01
ENV = "MACRO_MAPFEAT=1 MACRO_POLICY=%s" % os.environ.get("BOSS_MACRO_POLICY", "nets/rl5/pol_gate.npz")
dev = "cuda"


def net():
    g = lambda: nn.GELU(approximate="tanh")
    return nn.Sequential(nn.Linear(1235, 512), g(), nn.Linear(512, 256), g(), nn.Linear(256, 256), g(), nn.Linear(256, 128))


def load(m, p):
    z = np.load(p)
    for i in (0, 2, 4):
        m[i].weight.data = torch.tensor(z["pbody.%d.weight" % i]); m[i].bias.data = torch.tensor(z["pbody.%d.bias" % i])
    m[6].weight.data = torch.tensor(z["pi.weight"]); m[6].bias.data = torch.tensor(z["pi.bias"])
    return m


def save(m, p):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    z = np.load("policy_varN.npz")
    d = {"pbody.0.weight": sd["0.weight"], "pbody.0.bias": sd["0.bias"], "pbody.2.weight": sd["2.weight"], "pbody.2.bias": sd["2.bias"],
         "pbody.4.weight": sd["4.weight"], "pbody.4.bias": sd["4.bias"], "pi.weight": sd["6.weight"], "pi.bias": sd["6.bias"]}
    for k in ("vbody.0.weight", "vbody.0.bias", "vbody.2.weight", "vbody.2.bias", "v.weight", "v.bias", "zmean", "zstd"):
        d[k] = z[k]
    np.savez(p + ".tmp.npz", **d); os.replace(p + ".tmp.npz", p)


def collect(pol, seed0, n_per, outdir, step, acts_split=True, greedy=False):
    os.makedirs(outdir, exist_ok=True)
    procs = []
    for w in range(W):
        acts = ("2" if w % 2 == 0 else "3") if acts_split else "2,3"
        env = dict(os.environ, BOSS_ACTS=acts, BOSS_SEEDSTEP=str(step * W), BOSS_GREEDY="1" if greedy else "0")
        for kv in ENV.split():
            k, _, v = kv.partition("="); env[k] = v
        procs.append(subprocess.Popen(["nice", "-n", "5", ".venv/bin/python", "boss_rl.py", "worker", pol, str(seed0 + w * step), str(n_per),
                                       str(EPS), outdir], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    G = []
    for f in sorted(glob.glob(outdir + "/seed_*.npz")):
        z = np.load(f)
        if len(z["action"]):
            G.append({k: z[k] for k in z.files})
    return G


def val(pol, tag):
    G = collect(pol, VAL0, VAL_N // W, "%s/val_%s" % (TMP, tag), 29, acts_split=False, greedy=True)
    r = np.concatenate([g["reward"] for g in G]) if G else np.zeros(0)
    rec = {"tag": tag, "starts": len(G), "episodes": int(len(r)), "win_rate": float((r >= 1).mean()) if len(r) else None,
           "mean_R": float(r.mean()) if len(r) else None, "t": time.time()}
    print("VAL", json.dumps(rec), flush=True)
    return rec


def main(iters):
    hp = OUT + "/history.json"
    h = json.load(open(hp)) if os.path.exists(hp) else {"it": [], "val": []}
    cur = OUT + "/combat_latest.npz"
    if not os.path.exists(cur):
        shutil.copy("policy_varN.npz", cur)
    m = load(net(), cur).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    if not h["val"]:
        h["val"].append(val("policy_varN.npz", "varN")); json.dump(h, open(hp, "w"))
    for it in range(len(h["it"]), iters):
        t0 = time.time()
        snap = OUT + "/combat_it%03d.npz" % it
        if not os.path.exists(snap):
            shutil.copy(cur, snap)
        G = collect(snap, SEED0 + it * W * PER * 17, PER, "%s/it%03d" % (TMP, it), 17)
        if not G:
            print("iteration %d: no boss starts" % it, flush=True); continue
        O, NL, A, LP, ADV = [], [], [], [], []
        for g in G:
            R = g["reward"]; adv_ep = (R - R.mean()) / (R.std() + 1e-6) if R.std() > 1e-6 else np.zeros_like(R)
            O.append(g["obs"]); NL.append(g["n_legal"]); A.append(g["action"]); LP.append(g["logp"]); ADV.append(adv_ep[g["ep"]])
        O = torch.tensor(np.concatenate(O).astype(np.float32), device=dev); NL = torch.tensor(np.concatenate(NL).astype(np.int64), device=dev)
        A = torch.tensor(np.concatenate(A).astype(np.int64), device=dev); LP = torch.tensor(np.concatenate(LP), device=dev)
        ADV = torch.tensor(np.concatenate(ADV).astype(np.float32), device=dev)
        idxs = torch.arange(128, device=dev)
        for _ in range(EPOCHS):
            for b in torch.randperm(len(A), device=dev).split(4096):
                lg = m(O[b]).masked_fill(idxs[None, :] >= NL[b][:, None], -1e9)
                lp_all = F.log_softmax(lg, 1); lp = lp_all.gather(1, A[b][:, None])[:, 0]
                ratio = torch.exp(lp - LP[b])
                ent = -(lp_all.exp() * lp_all.clamp(min=-30)).sum(1).mean()
                loss = -torch.min(ratio * ADV[b], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV[b]).mean() - ENT * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        save(m, cur)
        allR = np.concatenate([g["reward"] for g in G])
        rec = {"iter": it, "starts": len(G), "episodes": int(len(allR)), "win_rate": float((allR >= 1).mean()), "mean_R": float(allR.mean()),
               "entropy": float(ent), "secs": round(time.time() - t0, 1)}
        h["it"].append(rec); print(json.dumps(rec), flush=True)
        if (it + 1) % VAL_EVERY == 0:
            shutil.copy(cur, OUT + "/val_it%03d.npz" % it)
            h["val"].append(val(OUT + "/val_it%03d.npz" % it, "it%03d" % it))
        json.dump(h, open(hp + ".tmp", "w")); os.replace(hp + ".tmp", hp)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
