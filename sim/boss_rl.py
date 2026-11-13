"""Phase 3: boss-fight combat RL (honest).

The combat network (varN layout: obs1235 -> 512-256-256 -> 128 logits over the legal actions) is trained with
PPO directly on Act 2/3 boss fights:

  worker POLICY_OUT SEED0 N K_EP OUTDIR [COMBAT_NPZ]
      For each seed: play the run with the best out-of-combat policy (MACRO_POLICY env) and cheap honest K=8
      combat until the first decision of an Act 2 or Act 3 BOSS fight; snapshot that state; play K_EP episodes
      of the boss fight, each from a FRESHLY DETERMINIZED copy (draw order / RNG / hidden queues resampled, so
      outcomes cannot be memorised), sampling actions from the combat net's softmax over the legal actions using
      only the visible observation (obs1200 + combat block). Writes OUTDIR/seed_<s>.npz with
      obs, n_legal, action, logp, episode index, per-episode reward (win + HP left / max HP), boss name.
  The learner (boss_rl_train.py, GPU) does PPO with a GROUP-RELATIVE advantage: reward minus the mean reward
  of the episodes from the same start.

Honest by construction: the policy input is the same visible observation honest search uses; every episode is
a determinized copy of the start state; nothing reads the true future.
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("HONEST", "1")
os.environ.setdefault("HONEST_CPP", "1")
os.environ.setdefault("HONEST_K", "8")
import eval_fullrun_analytics_v2 as E
from honest import make_honest
from eval_fullrun_search import GAMMA, TOPK, ROLL_CAP
import slaythespire as S

GREEDY = os.environ.get("BOSS_GREEDY", "0") == "1"   # validation: greedy policy
TARGET_ACTS = [int(x) for x in os.environ.get("BOSS_ACTS", "2").split(",")]   # which act bosses to collect (first match)
_LOSS = S.GameOutcome.PLAYER_LOSS
_UND = S.GameOutcome.UNDECIDED


class _Done(Exception):
    pass


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class CombatPolicy:
    def __init__(self, npz):
        z = np.load(npz)
        self.L = [(z["pbody.%d.weight" % i], z["pbody.%d.bias" % i]) for i in (0, 2, 4)]
        self.pi = (z["pi.weight"], z["pi.bias"])

    def logits(self, obs):
        h = obs
        for W, b in self.L:
            h = _gelu(h @ W.T + b)
        return h @ self.pi[0].T + self.pi[1]


def run_seed(seed, pol, k_ep, rng_seed):
    ev = make_honest(E.V2, GAMMA, TOPK, ROLL_CAP)("policy_varN.npz", "policy_varE.npz", "macro")
    base = ev._search_action
    out = {}

    def hooked():
        e = ev.env
        gc = e.gc
        room = str(gc.cur_room).split(".")[-1]
        if not getattr(ev, "_in_rollout", False) and room == "BOSS" and int(gc.act) in TARGET_ACTS:
            snap = ev._hsnap()
            rng = np.random.default_rng(rng_seed)
            O, NL, A, LP, EP, R = [], [], [], [], [], []
            for ep in range(k_ep):
                det = int(rng.integers(1, 2 ** 62))
                ev._hload(snap, det)                      # fresh determinized world for every episode
                steps = 0
                e0 = max(1, ev.env._enemy_hp()); last_e = e0
                while ev.env.in_combat and ev.env.gc.outcome == _UND and steps < 1500:
                    n = min(len(ev.env.bc.get_possible_actions()), 128)
                    if n <= 0:
                        break
                    obs = ev._obs1235().astype(np.float32)
                    lg = pol.logits(obs)[:n].astype(np.float64)
                    p = np.exp(lg - lg.max()); p /= p.sum()
                    a = int(np.argmax(lg)) if GREEDY else int(rng.choice(n, p=p))
                    O.append(obs); NL.append(n); A.append(a); LP.append(float(np.log(p[a] + 1e-12))); EP.append(ep)
                    ev.env.step(a)
                    steps += 1
                    if ev.env.in_combat and ev.env.bc is not None:
                        last_e = ev.env._enemy_hp()
                g = ev.env.gc
                won = g.outcome != _LOSS and not ev.env.in_combat
                # win: 1 + HP left fraction; loss: 0.5 x fraction of total enemy HP removed (signal on all-loss starts)
                R.append(float(won) + (max(0, int(g.cur_hp)) / max(1, int(g.max_hp)) if won else 0.5 * (1 - last_e / e0)))
            out.update(obs=np.array(O, np.float16), n_legal=np.array(NL, np.int16), action=np.array(A, np.int16),
                       logp=np.array(LP, np.float32), ep=np.array(EP, np.int16), reward=np.array(R, np.float32),
                       boss=np.array([str(gc.encounter).split(".")[-1]]), act=np.array([int(gc.act)], np.int16),
                       hp_in=np.array([int(snap[1].player.cur_hp) if snap[1] is not None else 0], np.int16))
            raise _Done()
        return base()

    ev._search_action = hooked
    try:
        ev.play_v2(seed)
    except _Done:
        pass
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "worker":
        pol_path, seed0, n, k_ep, outdir = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        step = int(os.environ.get("BOSS_SEEDSTEP", "17"))
        os.makedirs(outdir, exist_ok=True)
        pol = CombatPolicy(pol_path)
        for i in range(n):
            s = seed0 + i * step
            p = os.path.join(outdir, "seed_%d.npz" % s)
            if os.path.exists(p):
                continue
            t0 = time.time()
            d = run_seed(s, pol, k_ep, s * 7919 + 1)
            if d:
                np.savez_compressed(p + ".tmp.npz", **d); os.replace(p + ".tmp.npz", p)
            else:
                open(p.replace(".npz", ".none"), "w").write("no act2/3 boss reached\n")
            print(json.dumps({"seed": s, "boss": (d.get("boss", [""])[0] if d else None), "eps": k_ep,
                              "mean_R": float(d["reward"].mean()) if d else None, "secs": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
