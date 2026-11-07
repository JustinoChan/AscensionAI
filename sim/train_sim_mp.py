"""Production multiprocess PPO on STS-v0 with milestone checkpoints, greedy
fixed-seed evaluation, and CSV logging.

Trainer imports torch; env workers (sim_worker.py) import ONLY the slaythespire
extension -> sidesteps the in-process torch<->ext native conflict and uses many
cores. The fork has an ASLR-dependent ~50% worker-startup segfault (survivors are
stable); VecEnv handles it with retry-until-healthy + mid-run relaunch.

Every EVAL_EVERY games: save a checkpoint, run a greedy fixed-seed eval, append a
row to sim_logs/sim_training.csv. Runs until MAX_GAMES.

Usage: python train_sim_mp.py [max_games] [n_workers] [eval_every] [tag]
"""
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
from torch.distributions import Categorical

OBS, ACT = 1235, 128   # 1200 base obs + 35-d appended combat-feature block
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
torch.manual_seed(0)
np.random.seed(0)

MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
TAG = ("_" + sys.argv[4]) if len(sys.argv) > 4 else ""
# MODE: "shaped" = Variant C reward (worker fight/death terms + trainer deck
# PBRS); "bare" = obs-fix-only control (varB reward semantics, shaping off,
# diagnostics still logged). Bare mode uses its own worker file so the proven
# shaped worker stays byte-identical (the extension is layout-sensitive).
MODE = sys.argv[5] if len(sys.argv) > 5 else "shaped"
WORKER_FILE = "sim_worker_bare.py" if MODE == "bare" else "sim_worker.py"
# fightmix: the first N_RUN workers play full runs (sim_worker.py); the rest
# serve one-fight episodes from the harvested bank (sim_worker_fight.py).
# fightpure: ALL workers serve fights (N_RUN=0) -- removes the run-worker
# dilution that swamped the value signal in fightmix (pure-fight returns are
# ~97% learnable; mixed batches ~0.18). One PPO learner consumes the stream.
# Deck PBRS stays off automatically (PHI_W_RUN below is 0 for non-"shaped").
FIGHT_MODES = ("fightmix", "fightpure")
if MODE == "fightpure":
    N_RUN = 0
elif MODE == "fightmix":
    N_RUN = max(1, N // 3)
else:
    N_RUN = N
FIGHT_WORKER = "sim_worker_fight.py"
BANK_TRAIN = os.path.join(HERE, "bank_train.jsonl")
BANK_EVAL = os.path.join(HERE, "bank_eval.jsonl")
FIGHT_CONFIG = os.path.join(HERE, "fight_config.json")
EVAL_BANK_TIERS = []
if MODE in FIGHT_MODES:
    EVAL_BANK_TIERS = [json.loads(ln)["room"]
                       for ln in open(BANK_EVAL) if ln.strip()]
MODELS = os.path.join(HERE, "sim_models" + TAG)
LOGS = os.path.join(HERE, "sim_logs")
CSV_PATH = os.path.join(LOGS, "sim_training" + TAG + ".csv")
T = 128
EPOCHS, MB = 4, 256
GAMMA, LAM, CLIP, VF = 0.97, 0.95, 0.2, 0.5   # 0.97: ~33-step horizon, covers a fight
# Value-replay buffer (decoupled value learning): the value head is trained to
# near-convergence on a large rolling buffer of clean (obs, Monte-Carlo-return)
# pairs -- the offline setup that provably reaches R2 ~0.97 -- instead of 4 PPO
# mini-epochs (which plateaued at ev~0). Gives an informative advantage baseline.
VBUF_CAP, VALUE_EPOCHS, VMB, VLR = 20000, 4, 512, 1e-3
EVAL_MAX_EP_STEPS = 2000   # cap a greedy eval episode (early policies can loop)

# --- Variant A: exploration fix (vs baseline's entropy collapse at ~3.5k games)
REWARD_SCALE = 0.1          # env rewards are huge (floor*10, win 1000); rescale
ENT_INIT = 0.05             # baseline's fixed 0.01 was swamped by advantages
ENT_MIN, ENT_MAX = 0.005, 0.30
NENT_LOW, NENT_HIGH = 0.15, 0.30   # auto-tune band for normalized entropy (~0.2 target)
# fixed eval seed set (deterministic, reused at every milestone)
EVAL_SEEDS = [10007 + i * 7919 for i in range(48)]

# --- Variant C: deck-quality potential, computed TRAINER-side from the obs ---
# Per-step gc.deck reads in the worker destabilize the extension (~1 segfault
# per game: the deck binding copies the card vector on every access), so the
# potential is derived from the obs deck region instead -- zero native calls.
DECK0, DECK1 = 19, 239         # obs deck-count region, 2 slots/card (odd = upgraded)
CARD_MAX = 7.0                 # per-slot count cap in the obs encoding
STRIKE_SLOT, DEFEND_SLOT = 132, 38   # starter base slots (empirical; +1 = upgraded)
PHI_W, PHI_CAP = 10.0, 20.0    # reward weight per phi unit; non-starter sum cap
PHI_W_RUN = PHI_W if MODE == "shaped" else 0.0   # bare mode: shaping off
# Slots 0/1 are the encoder's fallback bucket (curses and unmapped cards alias
# there) -- zero-valued so curses earn nothing rather than +1.0 per copy.


def _deck_counts(obs2d):
    return np.rint(np.asarray(obs2d)[:, DECK0:DECK1] * CARD_MAX)


def phi_of(obs2d):
    """Deck score: starter Strike/Defend -0.3, other playable +1.0 (sum capped
    at PHI_CAP), +0.5 per upgrade; fallback slots zero-valued."""
    c = _deck_counts(obs2d)
    starters = (c[:, STRIKE_SLOT] + c[:, STRIKE_SLOT + 1]
                + c[:, DEFEND_SLOT] + c[:, DEFEND_SLOT + 1])
    fallback = c[:, 0] + c[:, 1]
    upgrades = c[:, 1::2].sum(axis=1) - c[:, 1]
    nonstarter = c.sum(axis=1) - starters - fallback
    return (-0.3 * starters + np.minimum(nonstarter, PHI_CAP)
            + 0.5 * upgrades).astype(np.float32)


def deck_sizes_of(obs2d):
    return _deck_counts(obs2d).sum(axis=1).astype(np.float32)


def _send(p, obj):
    b = pickle.dumps(obj, protocol=4)
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


class VecEnv:
    def __init__(self, n, base_seed=0, max_tries=60):
        self.n = n
        self.max_tries = max_tries
        self._seed = base_seed
        self.ps = [None] * n
        self.obs = np.zeros((n, OBS), np.float32)
        self.mask = np.zeros((n, ACT), np.int8)
        self.launches = 0
        for i in range(n):
            self._launch_healthy(i)

    def _spawn(self, i, seed):
        self.launches += 1
        if MODE in FIGHT_MODES and i >= N_RUN:
            cmd = [PY, FIGHT_WORKER, str(seed), BANK_TRAIN, FIGHT_CONFIG]
        else:
            cmd = [PY, WORKER_FILE, str(seed)]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=HERE)
        pack = _recv(p)
        if pack is None:
            try:
                p.kill()
            except Exception:
                pass
            return None
        return p, pack

    def _launch_healthy(self, i, seed=None):
        for _ in range(self.max_tries):
            if seed is None:
                self._seed += 100003
                s = self._seed
            else:
                s = seed
            res = self._spawn(i, s)
            if res is not None:
                p, pk = res
                self.ps[i] = p
                self.obs[i] = pk[0]
                self.mask[i] = pk[3]
                return
        raise RuntimeError("worker %d: no healthy launch in %d tries" % (i, self.max_tries))

    def _kill(self, i):
        try:
            self.ps[i].kill()
        except Exception:
            pass

    # ---- training step (lockstep, auto-reset on done) ----
    def step(self, actions):
        alive = [True] * self.n
        for i, (p, a) in enumerate(zip(self.ps, actions)):
            try:
                _send(p, ("step", int(a)))
            except Exception:
                alive[i] = False
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, np.float32)
        epfloors = []
        epoutcomes = []
        ndone = 0
        for i, p in enumerate(self.ps):
            out = _recv(p) if alive[i] else None
            if out is None:
                self._kill(i)
                self._launch_healthy(i)
                rew[i] = 0.0
                done[i] = 1.0
                continue
            o, r, d, m, fl, oc = out
            self.obs[i] = o
            self.mask[i] = m
            rew[i] = r
            done[i] = 1.0 if d else 0.0
            if d:
                ndone += 1
                if i < N_RUN:
                    epfloors.append(fl)
                else:
                    epoutcomes.append(oc)
        return self.obs, rew, done, self.mask, epfloors, ndone, epoutcomes

    # ---- eval primitives (per-worker, fixed seed, restart-on-death) ----
    def reset_seed(self, i, seed):
        try:
            _send(self.ps[i], ("reset_seed", int(seed)))
            out = _recv(self.ps[i])
        except Exception:
            out = None
        if out is not None:
            self.obs[i] = out[0]
            self.mask[i] = out[3]
            return
        self._kill(i)
        self._launch_healthy(i, seed=seed)  # fresh worker whose first reset == seed

    def eval_step(self, idxs, actions, seed_of):
        """Batched eval step for the active workers. Returns a list of
        (i, done, floor, outcome, died). On death, relaunches the worker on its
        eval seed (the caller decides whether to retry or give up on the seed)."""
        sent_ok = {}
        for i, a in zip(idxs, actions):
            try:
                _send(self.ps[i], ("step", int(a)))
                sent_ok[i] = True
            except Exception:
                sent_ok[i] = False
        out = []
        for i in idxs:
            res = _recv(self.ps[i]) if sent_ok[i] else None
            if res is None:
                self._kill(i)
                self._launch_healthy(i, seed=seed_of[i])
                out.append((i, False, 0, 0, True))
                continue
            o, r, d, m, fl, oc = res
            self.obs[i] = o
            self.mask[i] = m
            out.append((i, bool(d), int(fl), int(oc), False))
        return out

    def close(self):
        for p in self.ps:
            try:
                _send(p, ("close",))
            except Exception:
                pass
        for p in self.ps:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


class ActorCritic(nn.Module):
    # Separate policy and value trunks: prevents the value objective (dominant
    # gradient) from injecting noise into the policy's shared features -- a
    # diagnosed contributor to the dead-value-head failure.
    def __init__(self):
        super().__init__()
        self.pbody = nn.Sequential(nn.Linear(OBS, 512), nn.GELU(),
                                   nn.Linear(512, 256), nn.GELU(),
                                   nn.Linear(256, 256), nn.GELU())
        self.vbody = nn.Sequential(nn.Linear(OBS, 256), nn.GELU(),
                                   nn.Linear(256, 256), nn.GELU())
        self.pi = nn.Linear(256, ACT)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        return self.pi(self.pbody(x)), self.v(self.vbody(x)).squeeze(-1)


@torch.no_grad()
def evaluate(vec, net, seeds, max_attempts=2, workers=None):
    """Greedy fixed-seed eval. Robust to two fork failure modes:
    - policy loops forever on a seed       -> EVAL_MAX_EP_STEPS cap
    - seed deterministically segfaults the worker under greedy play
      (relaunch would replay to the same crash) -> max_attempts cap
    Capped seeds are recorded as (best_floor_reached, no-win)."""
    n = vec.n
    t_eval = time.time()
    launches0 = vec.launches
    results = []          # (floor, outcome, deck_size, phi)
    pending = list(seeds)
    cur = [None] * n
    steps = [0] * n
    best_floor = [0] * n  # survives deaths/retries; reset only on a new seed
    attempts = [0] * n
    active = [False] * n
    last_ds = [0] * n     # last seen deck size/phi, for capped/crashy fallbacks
    last_ph = [0.0] * n

    def assign(i):
        if pending:
            cur[i] = pending.pop()
            steps[i] = 0
            best_floor[i] = 0
            attempts[i] = 0
            vec.reset_seed(i, cur[i])
            last_ds[i] = float(deck_sizes_of(vec.obs[i:i + 1])[0])
            last_ph[i] = float(phi_of(vec.obs[i:i + 1])[0])
            active[i] = True
        else:
            active[i] = False

    for i in (range(n) if workers is None else workers):
        assign(i)
    while any(active):
        idxs = [i for i in range(n) if active[i]]
        ot = torch.as_tensor(vec.obs[idxs])
        mt = torch.as_tensor(vec.mask[idxs], dtype=torch.float32)
        logits, _ = net(ot)
        acts = torch.argmax(logits + (mt - 1.0) * 1e9, dim=1).numpy()
        seed_of = {i: cur[i] for i in idxs}
        for (i, done, floor, outcome, died) in vec.eval_step(idxs, acts, seed_of):
            if died:
                attempts[i] += 1
                steps[i] = 0
                if attempts[i] >= max_attempts:
                    # crashy seed: don't replay into the same segfault again
                    results.append((best_floor[i], 0, last_ds[i], last_ph[i]))
                    assign(i)
                continue
            steps[i] += 1
            best_floor[i] = max(best_floor[i], floor)
            if done:
                # vec.obs[i] is already the next episode's reset obs; use the
                # values stashed from the last in-episode step.
                results.append((floor, outcome, last_ds[i], last_ph[i]))
                assign(i)
            elif steps[i] >= EVAL_MAX_EP_STEPS:
                # greedy policy stuck in an action loop: cap and move on
                last_ds[i] = float(deck_sizes_of(vec.obs[i:i + 1])[0])
                last_ph[i] = float(phi_of(vec.obs[i:i + 1])[0])
                results.append((best_floor[i], 0, last_ds[i], last_ph[i]))
                assign(i)
            else:
                last_ds[i] = float(deck_sizes_of(vec.obs[i:i + 1])[0])
                last_ph[i] = float(phi_of(vec.obs[i:i + 1])[0])
    print("  eval: %d seeds in %.0fs (%d worker relaunches)" % (
        len(results), time.time() - t_eval, vec.launches - launches0), flush=True)
    floors = [f for f, _, _, _ in results]
    wins = sum(1 for _, o, _, _ in results if o == 1)
    nres = max(1, len(results))
    return {
        "n": len(results),
        "mean_floor": float(np.mean(floors)) if floors else 0.0,
        "max_floor": int(np.max(floors)) if floors else 0,
        "win_rate": wins / nres,
        "act2": float(np.mean([f >= 17 for f in floors])) if floors else 0.0,
        "act3": float(np.mean([f >= 34 for f in floors])) if floors else 0.0,
        "deck_size": float(np.mean([d for _, _, d, _ in results])) if results else 0.0,
        "phi": float(np.mean([p for _, _, _, p in results])) if results else 0.0,
    }


@torch.no_grad()
def evaluate_fights(vec, net, tiers, max_attempts=2):
    """Greedy eval over the held-out fight bank. Fight workers interpret
    reset_seed N as 'load eval-bank entry N'. Returns win rates by tier.
    Caveat: if a worker dies INSIDE reset_seed, the relaunched worker sits on
    a random entry until our explicit retry below re-sends reset_seed."""
    workers = list(range(N_RUN, N))
    t_eval = time.time()
    launches0 = vec.launches
    results = {}
    pending = list(range(len(tiers)))
    cur = [None] * vec.n
    steps = [0] * vec.n
    attempts = [0] * vec.n
    active = [False] * vec.n

    def assign(i):
        if pending:
            cur[i] = pending.pop()
            steps[i] = 0
            attempts[i] = 0
            vec.reset_seed(i, cur[i])
            active[i] = True
        else:
            active[i] = False

    for i in workers:
        assign(i)
    while any(active):
        idxs = [i for i in range(vec.n) if active[i]]
        ot = torch.as_tensor(vec.obs[idxs])
        mt = torch.as_tensor(vec.mask[idxs], dtype=torch.float32)
        logits, _ = net(ot)
        acts = torch.argmax(logits + (mt - 1.0) * 1e9, dim=1).numpy()
        seed_of = {i: cur[i] for i in idxs}
        for (i, done, floor, outcome, died) in vec.eval_step(idxs, acts, seed_of):
            if died:
                attempts[i] += 1
                steps[i] = 0
                if attempts[i] >= max_attempts:
                    results[cur[i]] = -1   # crashy entry counts as a loss
                    assign(i)
                else:
                    vec.reset_seed(i, cur[i])
                continue
            steps[i] += 1
            if done:
                results[cur[i]] = outcome
                assign(i)
            elif steps[i] >= 600:          # worker cap is 500; 600 = safety
                results[cur[i]] = 0
                assign(i)

    def rate(keys):
        vals = [results[k] for k in keys if k in results]
        return float(np.mean([v == 1 for v in vals])) if vals else 0.0

    out = {"all": rate(range(len(tiers)))}
    for tier in ("MONSTER", "ELITE", "BOSS"):
        out[tier] = rate([k for k, t in enumerate(tiers) if t == tier])
    print("  fight eval: %d entries in %.0fs (%d relaunches) | win all %.0f%% "
          "monster %.0f%% elite %.0f%% boss %.0f%%" % (
              len(results), time.time() - t_eval, vec.launches - launches0,
              100 * out["all"], 100 * out["MONSTER"], 100 * out["ELITE"],
              100 * out["BOSS"]), flush=True)
    return out


def save_ckpt(net, popt, vopt, games, updates, path):
    torch.save({"model": net.state_dict(), "popt": popt.state_dict(),
                "vopt": vopt.state_dict(), "games": games, "updates": updates,
                "obs_size": OBS, "n_actions": ACT,
                "net_arch": {"p": [512, 256, 256], "v": [256, 256]},
                "activation": "gelu"}, path)


def log_csv(row):
    header = ["timestamp", "games", "updates", "train_mean_floor", "eval_n",
              "eval_mean_floor", "eval_max_floor", "eval_win_rate", "eval_act2",
              "eval_act3", "steps_per_s", "relaunches", "norm_ent", "ent_coef",
              "train_deck_size", "eval_deck_size", "eval_phi",
              "expl_var", "fight_win_all", "fight_win_monster",
              "fight_win_elite", "fight_win_boss"]
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    print("launching %d workers (retry-until-healthy)..." % N, flush=True)
    vec = VecEnv(N)
    print("all %d workers healthy after %d launches" % (N, vec.launches), flush=True)
    phi_prev = phi_of(vec.obs)
    deck_prev = deck_sizes_of(vec.obs)
    net = ActorCritic()
    # decoupled optimizers: policy (PPO) and value (buffer regression) never
    # share gradients or learning rate.
    policy_opt = torch.optim.Adam(
        list(net.pbody.parameters()) + list(net.pi.parameters()), lr=3e-4)
    value_opt = torch.optim.Adam(
        list(net.vbody.parameters()) + list(net.v.parameters()), lr=VLR)
    ent_coef = ENT_INIT
    # value targets are normalized to unit scale (mean/std from the buffer).
    ret_mean, ret_std = 0.0, 1.0
    buf_obs = np.zeros((0, OBS), np.float32)   # rolling value-replay buffer
    buf_mc = np.zeros((0,), np.float32)
    recent = []
    recent_deck = []
    recent_fout = []      # recent fight outcomes (train win-rate signal)
    total_steps = 0
    total_games = 0
    upd = 0
    next_milestone = EVAL_EVERY
    # Resume after preemption/reboot: pick up from this tag's latest checkpoint.
    latest = os.path.join(MODELS, "ppo_sim_latest.pt")
    if os.path.exists(latest):
        try:
            ck = torch.load(latest, map_location="cpu")
            net.load_state_dict(ck["model"])
            if "popt" in ck:
                policy_opt.load_state_dict(ck["popt"])
            if "vopt" in ck:
                value_opt.load_state_dict(ck["vopt"])
            total_games = int(ck.get("games", 0))
            upd = int(ck.get("updates", 0))
            next_milestone = (total_games // EVAL_EVERY + 1) * EVAL_EVERY
            print("RESUMED from %s at %d games / %d updates" % (
                latest, total_games, upd), flush=True)
        except Exception as e:
            print("resume failed (%s); starting fresh" % e, flush=True)
    t0 = time.time()
    print("MAX_GAMES=%d N=%d EVAL_EVERY=%d T=%d batch=%d eval_seeds=%d "
          "reward_scale=%.2f ent_init=%.3f (auto-tune %.2f-%.2f norm-ent band) "
          "mode=%s worker=%s phi_w=%.1f n_run=%d n_fight=%d" % (
              MAX_GAMES, N, EVAL_EVERY, T, N * T, len(EVAL_SEEDS),
              REWARD_SCALE, ENT_INIT, NENT_LOW, NENT_HIGH,
              MODE, WORKER_FILE, PHI_W_RUN, N_RUN, N - N_RUN), flush=True)

    while total_games < MAX_GAMES:
        O, A, LP, V, R, D, M = [], [], [], [], [], [], []
        nent_sum, nent_n = 0.0, 0
        for _ in range(T):
            obs_t = torch.as_tensor(vec.obs)
            mask_t = torch.as_tensor(vec.mask, dtype=torch.float32)
            with torch.no_grad():
                logits, val = net(obs_t)
                dist = Categorical(logits=logits + (mask_t - 1.0) * 1e9)
                a = dist.sample()
                lp = dist.log_prob(a)
                # normalized entropy: per-sample entropy / ln(#legal actions)
                legal = mask_t.sum(dim=1).clamp(min=2)
                nent_sum += float((dist.entropy() / torch.log(legal)).mean())
                nent_n += 1
            O.append(obs_t); A.append(a); LP.append(lp)
            V.append(val * ret_std + ret_mean)   # denormalize to true value scale
            M.append(mask_t)
            _, r, d, _, epfl, ndone, epout = vec.step(a.numpy())
            # Deck-quality potential (PBRS form), trainer-side. On done steps
            # vec.obs is already the next episode's reset obs, so no term is
            # applied across the boundary (no terminal correction, per spec).
            phi_new = phi_of(vec.obs)
            deck_new = deck_sizes_of(vec.obs)
            cont = (d == 0.0)
            r = r + np.where(cont, PHI_W_RUN * (GAMMA * phi_new - phi_prev),
                             0.0).astype(np.float32)
            for i in range(N_RUN):
                if d[i] == 1.0:
                    recent_deck.append(float(deck_prev[i]))
            phi_prev, deck_prev = phi_new, deck_new
            R.append(torch.as_tensor(r) * REWARD_SCALE)
            D.append(torch.as_tensor(d))
            total_steps += N
            total_games += ndone
            recent += epfl
            recent_fout += epout
            recent[:] = recent[-500:]
            recent_deck[:] = recent_deck[-500:]
            recent_fout[:] = recent_fout[-1000:]

        Rs = torch.stack(R)
        Ds = torch.stack(D)
        Of = torch.cat(O); Af = torch.cat(A); LPf = torch.cat(LP); Mf = torch.cat(M)

        # ---- clean Monte-Carlo returns for COMPLETED episodes -> value buffer ----
        # reward-to-go within each episode (reset at done); the truncated tail at
        # rollout end (no done ahead) is excluded so every buffered target is a
        # clean MC return -- the signal the offline probe fit to R2 0.97.
        Rs_np = Rs.numpy(); Ds_np = Ds.numpy()
        mc = np.zeros((T, N), np.float32)
        comp = np.zeros((T, N), dtype=bool)
        run = np.zeros(N, np.float32); seen = np.zeros(N, dtype=bool)
        for t in reversed(range(T)):
            d_t = Ds_np[t] > 0
            seen = seen | d_t
            run = np.where(d_t, Rs_np[t], Rs_np[t] + GAMMA * run)
            mc[t] = run
            comp[t] = seen
        cflat = comp.reshape(-1)
        if cflat.any():
            buf_obs = np.concatenate([buf_obs, Of.numpy()[cflat]])[-VBUF_CAP:]
            buf_mc = np.concatenate([buf_mc, mc.reshape(-1)[cflat]])[-VBUF_CAP:]

        # ---- train the value head to near-convergence on the buffer ----
        ret_mean = float(buf_mc.mean()); ret_std = float(max(buf_mc.std(), 0.1))
        bt_obs = torch.as_tensor(buf_obs)
        bt_tgt = torch.as_tensor((buf_mc - ret_mean) / ret_std)
        vparams = list(net.vbody.parameters()) + list(net.v.parameters())
        nb = len(buf_mc); bidx = np.arange(nb)
        for _ in range(VALUE_EPOCHS):
            np.random.shuffle(bidx)
            for s in range(0, nb, VMB):
                j = bidx[s:s + VMB]
                vpred = net.v(net.vbody(bt_obs[j])).squeeze(-1)
                vl = ((vpred - bt_tgt[j]) ** 2).mean()
                value_opt.zero_grad(); vl.backward()
                nn.utils.clip_grad_norm_(vparams, 0.5)
                value_opt.step()

        # ---- advantages from the freshly-trained value (denormalized) ----
        with torch.no_grad():
            v_roll = (net.v(net.vbody(Of)).squeeze(-1) * ret_std + ret_mean).reshape(T, N)
            _, lastv = net(torch.as_tensor(vec.obs))
            last_v = lastv * ret_std + ret_mean
        Vs = torch.cat([v_roll, last_v.unsqueeze(0)])
        adv = torch.zeros(T, N); gae = torch.zeros(N)
        for t in reversed(range(T)):
            nonterm = 1.0 - Ds[t]
            delta = Rs[t] + GAMMA * Vs[t + 1] * nonterm - Vs[t]
            gae = delta + GAMMA * LAM * nonterm * gae
            adv[t] = gae
        ret = (adv + Vs[:-1]).reshape(-1)
        ev = float(1.0 - (ret - Vs[:-1].reshape(-1)).var() / (ret.var() + 1e-8))
        advf = adv.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)

        # ---- policy update (PPO clipped, POLICY params only) ----
        pparams = list(net.pbody.parameters()) + list(net.pi.parameters())
        idx = np.arange(T * N)
        ent_last = 0.0
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, T * N, MB):
                j = idx[s:s + MB]
                logits = net.pi(net.pbody(Of[j]))
                dist = Categorical(logits=logits + (Mf[j] - 1.0) * 1e9)
                lp = dist.log_prob(Af[j])
                ratio = torch.exp(lp - LPf[j])
                aj = advf[j]
                pg = -torch.min(ratio * aj, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * aj).mean()
                ent = dist.entropy().mean()
                loss = pg - ent_coef * ent
                policy_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pparams, 0.5)
                policy_opt.step()
                ent_last = float(ent)

        upd += 1
        # entropy auto-tune: hold sampled normalized entropy in the target band.
        # Gentle steps: the old x1.5/x0.7 overshot under the shaped reward's
        # larger advantage magnitudes (collapse to nent 0.02, then a 17x coef
        # spike whose transient cost varC ~2.5 eval floors at 35k games).
        norm_ent = nent_sum / max(1, nent_n)
        if norm_ent < NENT_LOW:
            ent_coef = min(ent_coef * 1.15, ENT_MAX)
        elif norm_ent > NENT_HIGH:
            ent_coef = max(ent_coef * 0.9, ENT_MIN)
        dt = time.time() - t0
        sps = round(total_steps / dt)
        tmf = float(np.mean(recent)) if recent else 0.0
        print("upd %3d | games %6d | steps/s %4d | train_floor %.2f | nent %.3f | coef %.3f | ev %.2f | vstd %.2f | relaunch %d" % (
            upd, total_games, sps, tmf, norm_ent, ent_coef, ev, ret_std, vec.launches - N), flush=True)

        if total_games >= next_milestone:
            ckpt = os.path.join(MODELS, "ppo_sim_%06d.pt" % total_games)
            save_ckpt(net, policy_opt, value_opt, total_games, upd, ckpt)
            save_ckpt(net, policy_opt, value_opt, total_games, upd, os.path.join(MODELS, "ppo_sim_latest.pt"))
            m = {"n": 0, "mean_floor": 0, "max_floor": 0, "win_rate": 0,
                 "act2": 0, "act3": 0, "deck_size": 0, "phi": 0}
            if N_RUN > 0:   # pure fight gym has no run workers -> skip full-run eval
                try:
                    m = evaluate(vec, net, EVAL_SEEDS, workers=list(range(N_RUN)))
                except Exception as e:
                    print("  EVAL ERROR: %s" % e, flush=True)
            tds = float(np.mean(recent_deck)) if recent_deck else 0.0
            print("  >> MILESTONE %d games | EVAL greedy n=%d floor %.2f (max %d) "
                  "win %.1f%% act2 %.0f%% act3 %.0f%% deck %.1f phi %.1f | ckpt %s" % (
                      total_games, m["n"], m["mean_floor"], m["max_floor"],
                      100 * m["win_rate"], 100 * m["act2"], 100 * m["act3"],
                      m["deck_size"], m["phi"], os.path.basename(ckpt)), flush=True)
            fw = {"all": 0.0, "MONSTER": 0.0, "ELITE": 0.0, "BOSS": 0.0}
            if MODE in FIGHT_MODES:
                try:
                    fw = evaluate_fights(vec, net, EVAL_BANK_TIERS)
                except Exception as e:
                    print("  FIGHT EVAL ERROR: %s" % e, flush=True)
                tfwr = (100 * float(np.mean([o == 1 for o in recent_fout]))
                        if recent_fout else 0.0)
                print("  train fight win-rate (sampling, last %d) = %.0f%%"
                      % (len(recent_fout), tfwr), flush=True)
            log_csv([datetime.utcnow().isoformat(), total_games, upd, round(tmf, 3),
                     m["n"], round(m["mean_floor"], 3), m["max_floor"],
                     round(m["win_rate"], 4), round(m["act2"], 4), round(m["act3"], 4),
                     sps, vec.launches - N, round(norm_ent, 4), round(ent_coef, 4),
                     round(tds, 2), round(m["deck_size"], 2), round(m["phi"], 2),
                     round(ev, 4), round(fw["all"], 4), round(fw["MONSTER"], 4),
                     round(fw["ELITE"], 4), round(fw["BOSS"], 4)])
            next_milestone += EVAL_EVERY

    save_ckpt(net, policy_opt, value_opt, total_games, upd, os.path.join(MODELS, "ppo_sim_final.pt"))
    print("\nDONE %.0fs | %d games | %d updates | total launches %d" % (
        time.time() - t0, total_games, upd, vec.launches), flush=True)
    # A finished run must not be relaunched by the cron watchdog: disarm the
    # autorun flag if it belongs to this tag. If a .sim_queue exists, promote
    # its first line into .sim_autorun so the watchdog starts the next queued
    # run on its next tick (preemption-safe sequencing for overnight batches).
    auto = os.path.join(HERE, ".sim_autorun")
    queue = os.path.join(HERE, ".sim_queue")
    try:
        if TAG and os.path.exists(auto) and TAG.lstrip("_") in open(auto).read():
            os.remove(auto)
            print("autorun disarmed (run complete)", flush=True)
            if os.path.exists(queue):
                lines = [ln.strip() for ln in open(queue).read().splitlines()
                         if ln.strip()]
                if lines:
                    with open(auto, "w") as f:
                        f.write(lines[0] + "\n")
                    rest = "\n".join(lines[1:])
                    with open(queue, "w") as f:
                        f.write(rest + "\n" if rest else "")
                    print("queued run armed: %s" % lines[0], flush=True)
    except Exception:
        pass
    vec.close()


if __name__ == "__main__":
    main()
