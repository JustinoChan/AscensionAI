"""Full-run eval: does GOOD COMBAT (1-ply search) break the ~14.7 full-run
plateau? Plays complete Ironclad runs over the fixed EVAL_SEEDS; non-combat
decisions (map/card/shop/event/rest) use the baseline full-run policy
(policy_varE.npz, 1200-d shared trunk); combat decisions either also use that
baseline (mode=baseline, reproduces the plateau as a harness check) or 1-ply
rollout search with the combat net (policy_varL.npz, 1235-d, mode=search).

The fight state at a combat decision is reconstructed by deterministic replay of
(seed + all actions taken so far) via execute-only _ff_step -- the same machinery
the search worker uses, generalized to full-run prefixes. Reports avg floor,
Act-2 reach %, and win %.

Torch-free (the slaythespire extension segfaults with torch in-process).

argv: [1] mode = baseline | search | verify
      [2] combat_npz (default policy_varL.npz)
      [3] vare_npz   (default policy_varE.npz)
      [4] n_seeds    (default 48)
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Pin BLAS to one thread (consistent with the search worker; keeps a candidate
# rollout cheap and avoids oversubscription if this is ever run in parallel).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import gym_sts_lightspeed  # noqa: F401
import slaythespire
from sim_worker_fight import FightEnv          # full StsEnv + dense combat reward
from combat_features import combat_block
from np_policy import NpNet                    # combat net (1235-d)

GAMMA = 0.97
TOPK = 6
ROLL_CAP = 200
MAX_GAME_STEPS = 3000
WIN_BONUS, LOSS_PENALTY = 50.0, -50.0
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
_UND = slaythespire.GameOutcome.UNDECIDED
_VIC = slaythespire.GameOutcome.PLAYER_VICTORY
EVAL_SEEDS = [10007 + i * 7919 for i in range(48)]


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpPolicy:
    """varE baseline full-run policy (shared trunk, 1200-d) -- argmax greedy."""
    def __init__(self, npz_path):
        w = np.load(npz_path)
        self.layers = [(w["body.%d.weight" % i], w["body.%d.bias" % i])
                       for i in (0, 2, 4)]
        self.pi = (w["pi.weight"], w["pi.bias"])

    def act(self, obs_norm, mask):
        h = obs_norm
        for W, b in self.layers:
            h = _gelu(h @ W.T + b)
        logits = h @ self.pi[0].T + self.pi[1]
        logits = np.where(np.asarray(mask) > 0, logits, -1e9)
        return int(np.argmax(logits))


class FullRunEval:
    def __init__(self, combat_npz, vare_npz, mode):
        self.mode = mode
        self.env = FightEnv(ascension=0)
        self.maxima = np.maximum(
            np.asarray(self.env.nn.getObservationMaximums(), np.float32), 1.0)
        self.net = NpNet(combat_npz)        # combat (1235-d)
        self.vare = NpPolicy(vare_npz)      # non-combat / baseline (1200-d)
        self.seed = None
        self.history = []

    # ---- observations ----
    def _obs1200(self):
        return np.asarray(self.env._get_observation(), np.float32) / self.maxima

    def _obs1235(self):
        return np.concatenate(
            [self._obs1200(), combat_block(self.env)]).astype(np.float32)

    def _mask(self):
        return np.asarray(self.env._get_info()["action_mask"], np.int8)

    # ---- execute-only replay (mirrors the search worker) ----
    def _ff_step(self, idx):
        env = self.env
        if env.in_combat and env.bc is not None:
            acts = env.bc.get_possible_actions()
        else:
            acts = env.gc.get_possible_actions()
        if not acts:
            return False
        action = acts[idx] if idx < len(acts) else acts[0]
        if env.in_combat:
            action.execute(env.bc)
            if env.bc.outcome != slaythespire.BattleOutcome.UNDECIDED:
                env.bc.exit_battle(env.gc)
                env.in_combat = False
                env.bc = None
        else:
            action.execute(env.gc)
            if env.gc.screen_state == slaythespire.ScreenState.BATTLE:
                env.in_combat = True
                env.bc = slaythespire.BattleContext()
                env.bc.init(env.gc)
        return True

    def _restore(self):
        """Replay seed + full action history to the current decision state."""
        try:
            self.env.reset(seed=int(self.seed))
        except Exception:
            return False
        for a in self.history:
            if not self._ff_step(int(a)) or self.env.gc.outcome != _UND:
                return False
        return True

    # ---- 1-ply rollout search (fight-end value) ----
    def _term(self):
        if not self.env.in_combat:
            return True, (LOSS_PENALTY if self.env.gc.outcome == _LOSS
                          else WIN_BONUS)
        return False, 0.0

    def _rollout_q(self, action):
        if not self._restore():
            return None
        self.env.last_hp = self.env.bc.player.cur_hp
        self.env._last_enemy_hp = self.env._enemy_hp()
        G, disc = 0.0, 1.0
        _, r, _, _, _ = self.env.step(int(action))
        done, tb = self._term()
        G += disc * (r + tb)
        disc *= GAMMA
        steps = 0
        while not done and steps < ROLL_CAP:
            a = self.net.greedy(self._obs1235(), self._mask())
            _, r, _, _, _ = self.env.step(a)
            done, tb = self._term()
            G += disc * (r + tb)
            disc *= GAMMA
            steps += 1
        return G

    def _search_action(self):
        mask = self._mask()
        legal = [i for i in range(128) if mask[i]]
        if len(legal) > TOPK:
            prior = self.net.logits(self._obs1235(), mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
        best, bq = legal[0], -1e18
        for a in legal:
            qa = self._rollout_q(a)
            if qa is not None and qa > bq:
                bq, best = qa, a
        if not self._restore():       # restore live state for the real step
            return None
        return best

    # ---- play one full run ----
    def play(self, seed):
        self.seed = seed
        self.history = []
        try:
            self.env.reset(seed=int(seed))
        except Exception:
            return None
        for _ in range(MAX_GAME_STEPS):
            if self.mode == "search" and self.env.in_combat:
                a = self._search_action()
                if a is None:
                    break
            else:
                a = self.vare.act(self._obs1200(), self._mask())
            _, r, term, trunc, _ = self.env.step(int(a))
            self.history.append(int(a))
            if term or trunc:
                break
        return {"seed": seed, "floor": int(self.env.gc.floor_num),
                "act": int(self.env.gc.act),
                "win": bool(self.env.gc.outcome == _VIC)}


def _hp(env):
    if env.in_combat and env.bc is not None:
        return int(env.bc.player.cur_hp)
    return int(env.gc.cur_hp)


def verify(ev, seeds, nsteps=120):
    """GATE: live greedy play vs _ff_step replay must produce the same
    (floor, in_combat, hp) sequence -- proves full-run replay is deterministic,
    which the search reconstruction depends on."""
    ok = 0
    for seed in seeds:
        ev.seed = seed
        ev.env.reset(seed=int(seed))
        live, hist = [], []
        for _ in range(nsteps):
            live.append((int(ev.env.gc.floor_num), bool(ev.env.in_combat),
                         _hp(ev.env)))
            a = ev.vare.act(ev._obs1200(), ev._mask())
            _, r, term, trunc, _ = ev.env.step(int(a))
            hist.append(int(a))
            if term or trunc:
                break
        ev.env.reset(seed=int(seed))
        rep = []
        good = True
        for a in hist:
            rep.append((int(ev.env.gc.floor_num), bool(ev.env.in_combat),
                        _hp(ev.env)))
            if not ev._ff_step(int(a)):
                good = False
                break
        match = good and rep == live[:len(rep)]
        print("seed %d: steps %d  replay_match=%s" % (seed, len(hist), match))
        if not match:
            for i, (a_, b_) in enumerate(zip(live, rep)):
                if a_ != b_:
                    print("   first mismatch @%d live=%s replay=%s" % (i, a_, b_))
                    break
        ok += int(match)
    print("VERIFY %d/%d seeds replay-deterministic" % (ok, len(seeds)))
    return ok == len(seeds)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    combat_npz = sys.argv[2] if len(sys.argv) > 2 else "policy_varL.npz"
    vare_npz = sys.argv[3] if len(sys.argv) > 3 else "policy_varE.npz"
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    seeds = EVAL_SEEDS[:n]
    ev = FullRunEval(combat_npz, vare_npz, mode)

    if mode == "verify":
        verify(ev, seeds[:5])
        return

    t0 = time.time()
    floors, acts, wins = [], [], []
    for i, seed in enumerate(seeds):
        res = ev.play(seed)
        if res is None:
            print("seed %d: FAILED to start" % seed, flush=True)
            continue
        floors.append(res["floor"])
        acts.append(res["act"])
        wins.append(int(res["win"]))
        print("[%2d/%2d] seed %d -> floor %2d act %d win %d  (%.0fs)" % (
            i + 1, len(seeds), seed, res["floor"], res["act"], res["win"],
            time.time() - t0), flush=True)

    fl = np.asarray(floors, np.float64)
    act2 = np.mean([a >= 2 for a in acts]) if acts else 0.0
    act3 = np.mean([a >= 3 for a in acts]) if acts else 0.0
    print("\n==== %s | %d runs | %.0fs ====" % (mode, len(floors),
                                                time.time() - t0))
    print("avg floor   : %.2f  (max %d)" % (fl.mean() if len(fl) else 0,
                                            int(fl.max()) if len(fl) else 0))
    print("Act-2 reach : %.1f%%" % (100 * act2))
    print("Act-3 reach : %.1f%%" % (100 * act3))
    print("win rate    : %.1f%% (%d/%d)" % (100 * np.mean(wins) if wins else 0,
                                            sum(wins), len(wins)))


if __name__ == "__main__":
    main()
