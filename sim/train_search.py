"""Expert-iteration distillation trainer. Drives N torch-free search workers to
generate per-decision targets, fits the separate-trunk ActorCritic (policy
cross-entropy to the search's improved policy + value MSE to max-Q, value
targets normalized via a rolling buffer), exports policy_live.npz each update so
workers track the improving policy, and logs greedy (no-search) vs search fight
win-rates. Bootstraps from sim_models_varJ.

Usage: python train_search.py [max_decisions] [n_workers] [eval_every] [tag]"""
import os
import sys
import csv
import json
import time
import struct
import pickle
import subprocess
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

from train_sim_mp import ActorCritic, OBS, ACT   # reuse net + dims (1235/128)

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
torch.manual_seed(0)
np.random.seed(0)

MAX_DEC = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
TAG = ("_" + sys.argv[4]) if len(sys.argv) > 4 else ""
MODELS = os.path.join(HERE, "sim_models" + TAG)
LOGS = os.path.join(HERE, "sim_logs")
CSV_PATH = os.path.join(LOGS, "sim_training" + TAG + ".csv")
WORKER = "sim_worker_search.py"
BANK = os.path.join(HERE, "bank_train.jsonl")
CFG = os.path.join(HERE, "fight_config.json")
LIVE = os.path.join(HERE, "policy_live.npz")
EVAL_TIERS = [json.loads(l)["room"] for l in open(
    os.path.join(HERE, "bank_eval.jsonl")) if l.strip()]
BATCH = 4096
EPOCHS, MB = 3, 256
VBUF_CAP = 40000


def _send(p, o):
    b = pickle.dumps(o, 4)
    p.stdin.write(struct.pack(">I", len(b)))
    p.stdin.write(b)
    p.stdin.flush()


def _recv(p):
    h = p.stdout.read(4)
    if len(h) < 4:
        return None
    n = struct.unpack(">I", h)[0]
    buf = b""
    while len(buf) < n:
        c = p.stdout.read(n - len(buf))
        if not c:
            return None
        buf += c
    return pickle.loads(buf)


def export_live(net, zmean=0.0, zstd=1.0):
    tmp = LIVE + ".tmp.npz"
    sd = net.state_dict()
    arrs = {k: v.detach().numpy() for k, v in sd.items()}
    arrs["zmean"] = np.float32(zmean)
    arrs["zstd"] = np.float32(zstd)
    np.savez(tmp, **arrs)
    os.replace(tmp, LIVE)


class SearchPool:
    def __init__(self, n):
        self.n = n
        self.ps = [None] * n
        self.launches = 0
        for i in range(n):
            self._launch(i)

    def _launch(self, i):
        self.launches += 1
        self.ps[i] = subprocess.Popen(
            [PY, WORKER, str(1000 + self.launches), BANK, CFG, LIVE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=HERE)

    def collect(self, k, beta):
        """Pipelined: send 'gen' to ALL workers, then read ALL replies, so the
        12 search workers run in parallel (sequential send/recv would idle 11 of
        12). Dead workers are relaunched and skipped this round. beta is the
        DAgger expert-mix probability passed through to each worker."""
        recs = []
        wins = []
        while len(recs) < k:
            alive = []
            for i, p in enumerate(self.ps):
                try:
                    _send(p, ("gen", beta))
                    alive.append(i)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                    self._launch(i)
            for i in alive:
                try:
                    out = _recv(self.ps[i])
                except Exception:
                    out = None
                if out is None:
                    try:
                        self.ps[i].kill()
                    except Exception:
                        pass
                    self._launch(i)
                else:
                    obs, pi, z, mask, done, oc = out
                    recs.append((obs, pi, z, mask))
                    if done:
                        wins.append(1 if oc == 1 else 0)
        return recs, wins

    def eval(self, kind):
        res = [0] * len(EVAL_TIERS)
        for idx in range(len(EVAL_TIERS)):
            p = self.ps[idx % self.n]
            for _try in range(3):
                try:
                    _send(p, (kind, idx))
                    out = _recv(p)
                except Exception:
                    out = None
                if out is None:
                    try:
                        self.ps[idx % self.n].kill()
                    except Exception:
                        pass
                    self._launch(idx % self.n)
                    p = self.ps[idx % self.n]
                    continue
                res[idx] = out[1]
                break
        by = {"MONSTER": [], "ELITE": [], "BOSS": []}
        for r, t in zip(res, EVAL_TIERS):
            by.setdefault(t, []).append(r)
        out = {"all": float(np.mean(res)) if res else 0.0}
        for t in ("MONSTER", "ELITE", "BOSS"):
            out[t] = float(np.mean(by[t])) if by[t] else 0.0
        return out

    def close(self):
        for p in self.ps:
            try:
                _send(p, ("close",))
            except Exception:
                pass


def log_csv(row):
    hdr = ["timestamp", "decisions", "updates", "steps_per_s",
           "train_fight_wr", "ev",
           "g_all", "g_monster", "g_elite", "g_boss",
           "s_all", "s_monster", "s_elite", "s_boss"]
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(hdr)
        w.writerow(row)


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    net = ActorCritic()
    bootL = os.path.join(HERE, "sim_models_varL", "ppo_sim_latest.pt")
    net.load_state_dict(torch.load(bootL, map_location="cpu")["model"])
    print("bootstrapped policy+value <- varL (keep Act-1, add Act-2/3)", flush=True)
    popt = torch.optim.Adam(
        list(net.pbody.parameters()) + list(net.pi.parameters()), lr=3e-4)
    vopt = torch.optim.Adam(
        list(net.vbody.parameters()) + list(net.v.parameters()), lr=1e-3)
    export_live(net)
    pool = SearchPool(N)
    buf_obs = np.zeros((0, OBS), np.float32)
    buf_z = np.zeros((0,), np.float32)
    recent_w = []
    dec = 0
    upd = 0
    t0 = time.time()
    next_m = EVAL_EVERY
    print("search-distill: N=%d batch=%d eval_every=%d tag=%s" % (
        N, BATCH, EVAL_EVERY, TAG), flush=True)

    while dec < MAX_DEC:
        beta = max(0.05, 0.5 * (1.0 - dec / MAX_DEC))
        recs, wins = pool.collect(BATCH, beta)
        dec += len(recs)
        recent_w += wins
        recent_w[:] = recent_w[-1000:]
        O = torch.as_tensor(np.stack([r[0] for r in recs]))
        PI = torch.as_tensor(np.stack([r[1] for r in recs]))
        Z = np.stack([r[2] for r in recs]).astype(np.float32)
        Mk = torch.as_tensor(np.stack([r[3] for r in recs]).astype(np.float32))
        buf_obs = np.concatenate([buf_obs, O.numpy()])[-VBUF_CAP:]
        buf_z = np.concatenate([buf_z, Z])[-VBUF_CAP:]
        zmean = float(buf_z.mean())
        zstd = float(max(buf_z.std(), 0.1))

        idx = np.arange(len(recs))
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, len(recs), MB):
                j = idx[s:s + MB]
                logits = net.pi(net.pbody(O[j]))
                logits = logits + (Mk[j] - 1.0) * 1e9
                logp = torch.log_softmax(logits, dim=1)
                ce = -(PI[j] * logp).sum(1).mean()
                popt.zero_grad()
                ce.backward()
                nn.utils.clip_grad_norm_(
                    list(net.pbody.parameters()) + list(net.pi.parameters()), 0.5)
                popt.step()

        bt_o = torch.as_tensor(buf_obs)
        bt_t = torch.as_tensor((buf_z - zmean) / zstd)
        nb = len(buf_z)
        bidx = np.arange(nb)
        for _ in range(EPOCHS):
            np.random.shuffle(bidx)
            for s in range(0, nb, 512):
                j = bidx[s:s + 512]
                vp = net.v(net.vbody(bt_o[j])).squeeze(-1)
                vl = ((vp - bt_t[j]) ** 2).mean()
                vopt.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(
                    list(net.vbody.parameters()) + list(net.v.parameters()), 0.5)
                vopt.step()
        with torch.no_grad():
            vp = (net.v(net.vbody(bt_o)).squeeze(-1) * zstd + zmean).numpy()
        ev = float(1.0 - np.var(buf_z - vp) / (np.var(buf_z) + 1e-8))

        export_live(net, zmean, zstd)
        upd += 1
        sps = round(dec / (time.time() - t0))
        twr = 100 * float(np.mean(recent_w)) if recent_w else 0.0
        print("upd %3d | dec %7d | steps/s %4d | train_wr %.0f%% | ev %.2f | "
              "beta %.2f | relaunch %d" % (
                  upd, dec, sps, twr, ev, beta, pool.launches - N),
              flush=True)

        if dec >= next_m:
            torch.save({"model": net.state_dict(), "obs_size": OBS,
                        "n_actions": ACT},
                       os.path.join(MODELS, "ppo_sim_%07d.pt" % dec))
            torch.save({"model": net.state_dict(), "obs_size": OBS,
                        "n_actions": ACT},
                       os.path.join(MODELS, "ppo_sim_latest.pt"))
            g = pool.eval("eval_greedy")
            sc = pool.eval("eval_search")
            print("  >> MILESTONE dec %d | GREEDY all %.0f%% mon %.0f%% eli %.0f%% "
                  "boss %.0f%% | SEARCH boss %.0f%%" % (
                      dec, 100 * g["all"], 100 * g["MONSTER"], 100 * g["ELITE"],
                      100 * g["BOSS"], 100 * sc["BOSS"]), flush=True)
            log_csv([datetime.utcnow().isoformat(), dec, upd, sps, round(twr, 1),
                     round(ev, 4),
                     round(g["all"], 4), round(g["MONSTER"], 4),
                     round(g["ELITE"], 4), round(g["BOSS"], 4),
                     round(sc["all"], 4), round(sc["MONSTER"], 4),
                     round(sc["ELITE"], 4), round(sc["BOSS"], 4)])
            next_m += EVAL_EVERY

    pool.close()
    auto = os.path.join(HERE, ".sim_autorun")
    if TAG and os.path.exists(auto) and TAG.lstrip("_") in open(auto).read():
        os.remove(auto)
    print("DONE %.0fs | %d decisions | %d updates" % (
        time.time() - t0, dec, upd), flush=True)


if __name__ == "__main__":
    main()
