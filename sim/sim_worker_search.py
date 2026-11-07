"""Train-time-planning fight worker (torch-free). Plays bank fights; at each
decision runs 1-ply rollout search with a NumPy copy of the current net and
emits a distillation target. The fight state is defined by (bank entry + the
real in-combat actions taken so far); there is no state cloning, so every
candidate evaluation re-derives the state by replay (deterministic, verified).

argv: [1] sampling seed  [2] bank_train.jsonl  [3] fight_config.json
      [4] policy_live.npz

Protocol (parent->worker):
  ('gen',)             -> run search at the current decision, advance one real
                          action; reply a decision-target tuple
  ('eval_greedy', idx) -> play eval-bank entry idx greedily (NO search) to end;
                          reply ('EVAL', outcome:int)
  ('eval_search', idx) -> play eval-bank entry idx with search to end;
                          reply ('EVAL', outcome:int)
  ('close',)
Worker->parent for 'gen':
  (obs_f32[1235], pi_f32[128], z_f32, mask_i8[128], done_bool, outcome_int)
  done/outcome describe whether the real action just taken ENDED the fight.

Reloads policy_live.npz (mtime) at the start of every 'gen'/'eval'. Reward
matches the fight gym: per-step FightEnv._get_reward (HP delta + damage dealt)
plus terminal +50 win / -50 loss; discounted by GAMMA."""
import sys
import os
import json
import struct
import pickle
import warnings
warnings.filterwarnings("ignore")

# Pin BLAS to a single thread BEFORE numpy imports: 12 worker processes each
# spawning multi-threaded matmuls oversubscribes the cores and made one search
# decision take minutes. Single-threaded numpy + process-level parallelism is
# the right model here.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

_proto = sys.stdout.buffer
_inp = sys.stdin.buffer

import numpy as np
import slaythespire
from sim_worker_fight import FightEnv          # reuse env + dense reward
from combat_features import combat_block
from np_policy import NpNet

GAMMA = 0.97
TEMP = 0.5                 # softmax temperature on Q for the policy target
ROLL_CAP = 200
TOPK = 6                   # roll out only the top-K legal actions by policy prior
WIN_BONUS, LOSS_PENALTY = 50.0, -50.0
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
_UND = slaythespire.GameOutcome.UNDECIDED


def send(obj):
    b = pickle.dumps(obj, protocol=4)
    _proto.write(struct.pack(">I", len(b)))
    _proto.write(b)
    _proto.flush()


def recv():
    h = _inp.read(4)
    if len(h) < 4:
        return None
    n = struct.unpack(">I", h)[0]
    buf = b""
    while len(buf) < n:
        c = _inp.read(n - len(buf))
        if not c:
            return None
        buf += c
    return pickle.loads(buf)


def _shutdown():
    try:
        _proto.flush()
    except Exception:
        pass
    os._exit(0)


class Searcher:
    def __init__(self, bank_path, cfg_path, npz_path, seed):
        self.bank = [json.loads(l) for l in open(bank_path) if l.strip()]
        d = os.path.dirname(os.path.abspath(bank_path))
        self.eval_bank = [json.loads(l) for l in open(
            os.path.join(d, "bank_eval.jsonl")) if l.strip()]
        cfg = json.load(open(cfg_path))
        tw = cfg["tier_weights"]
        los = float(cfg["loss_oversample"])
        af = {2: 3.0, 3: 2.5}   # oversample Act-2/3 fights (combat-coverage gap)
        w = np.array([tw.get(e["room"], 0.0) * (los if e.get("lost") else 1.0)
                      * af.get(e.get("act", 1), 1.0)
                      for e in self.bank], dtype=np.float64)
        keep = w > 0
        self.bank = [e for e, k in zip(self.bank, keep) if k]
        self.probs = w[keep] / w[keep].sum()
        self.rng = np.random.default_rng(seed)
        self.env = FightEnv(ascension=0)
        self.maxima = np.maximum(
            np.asarray(self.env.nn.getObservationMaximums(), np.float32), 1.0)
        self.net = NpNet(npz_path)
        self.entry = None
        self.live = []

    def _obs(self):
        raw = np.asarray(self.env._get_observation(), np.float32) / self.maxima
        return np.concatenate([raw, combat_block(self.env)]).astype(np.float32)

    def _ff_step(self, idx):
        """Execute one recorded action WITHOUT computing obs/reward/info beyond
        the legal-action list (mirrors StsEnv.step's transition path). Fast."""
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
        """Fast-forward to the current decision state (bank prefix + live
        actions) via execute-only steps. Returns True if in combat. Then seed
        the reward trackers so the subsequent full-step rollout is correct."""
        if self.entry is None:
            return False
        try:
            self.env.reset(seed=int(self.entry["seed"]))
        except Exception:
            return False
        for a in self.entry["prefix"]:
            if not self._ff_step(int(a)) or self.env.gc.outcome != _UND:
                return False
        for a in self.live:
            if not self._ff_step(int(a)) or not self.env.in_combat:
                return False
        if not self.env.in_combat or self.env.bc is None:
            return False
        self.env.last_hp = self.env.bc.player.cur_hp
        self.env._last_enemy_hp = self.env._enemy_hp()
        return True

    def _new_fight(self):
        for _ in range(20):
            e = self.bank[int(self.rng.choice(len(self.bank), p=self.probs))]
            self.entry, self.live = e, []
            if self._restore():
                return True
        os._exit(1)            # parent relaunches

    def _term_bonus(self):
        if not self.env.in_combat:
            return True, (LOSS_PENALTY if self.env.gc.outcome == _LOSS else WIN_BONUS)
        return False, 0.0

    def _rollout_q(self, action):
        if not self._restore():
            return None
        G, disc = 0.0, 1.0
        _, r, term, _, _ = self.env.step(int(action))
        done, tb = self._term_bonus()
        G += disc * (r + tb)
        disc *= GAMMA
        steps = 0
        while not done and steps < ROLL_CAP:
            o = self._obs()
            a = self.net.greedy(o, self.env._get_info()["action_mask"])
            _, r, term, _, _ = self.env.step(a)
            done, tb = self._term_bonus()
            G += disc * (r + tb)
            disc *= GAMMA
            steps += 1
        return G

    def _q_value(self, action):
        """Value-bootstrapped 1-ply score: immediate dense reward plus the value
        head's estimate of the resulting position (terminal uses the literal
        +/-50). _restore already seeds last_hp/_last_enemy_hp for a correct
        reward, so the single env.step gives r(s, a)."""
        if not self._restore():
            return None
        _, r, _, _, _ = self.env.step(int(action))
        done, tb = self._term_bonus()
        if done:
            return r + tb
        return r + GAMMA * self.net.value_denorm(self._obs())

    def gen(self, beta):
        self.net.reload()
        if self.entry is None and not self._new_fight():
            return None
        if not self._restore():
            self._new_fight()
        obs = self._obs()
        mask = np.asarray(self.env._get_info()["action_mask"], np.int8)
        legal = [i for i in range(128) if mask[i]]
        if len(legal) > TOPK:
            prior = self.net.logits(obs, mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
        q = {}
        for a in legal:
            qa = self._rollout_q(a)
            if qa is not None:
                q[a] = qa
        if not q:
            self._new_fight()
            return self.gen(beta)
        legal = list(q.keys())
        qa = np.array([q[a] for a in legal], np.float64)
        ex = np.exp((qa - qa.max()) / TEMP)
        probs = ex / ex.sum()
        pi = np.zeros(128, np.float32)
        for a, p in zip(legal, probs):
            pi[a] = p
        z = float(qa.max())
        # DAgger: advance the real trajectory with the LEARNER (current greedy
        # policy) most of the time, so the buffer holds the states the deployed
        # greedy policy actually visits; with prob beta advance via the search
        # ("expert") sample to still reach deep boss states. Every state is
        # labeled with the search target (pi, z) regardless of who advances.
        if self.rng.random() < beta:
            real = int(self.rng.choice(legal, p=probs))
        else:
            real = int(self.net.greedy(obs, mask))
        if not self._restore():
            self._new_fight()
            return self.gen(beta)
        self.env.step(real)
        self.live.append(real)
        done, _ = self._term_bonus()
        outcome = 0
        if done:
            outcome = 1 if self.env.gc.outcome != _LOSS else -1
            self._new_fight()
        return (obs, pi, np.float32(z), mask, bool(done), int(outcome))

    def _play(self, idx, use_search):
        self.net.reload()
        e = self.eval_bank[int(idx) % len(self.eval_bank)]
        self.entry, self.live = e, []
        if not self._restore():
            return 0
        steps = 0
        while self.env.in_combat and steps < ROLL_CAP:
            if use_search:
                mask = self.env._get_info()["action_mask"]
                legal = [i for i in range(128) if mask[i]]
                if len(legal) > TOPK:
                    prior = self.net.logits(self._obs(), mask)
                    legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
                best, bq = legal[0], -1e18
                for a in legal:
                    qa = self._rollout_q(a)
                    if qa is not None and qa > bq:
                        bq, best = qa, a
                if not self._restore():
                    break
                a = best
            else:
                a = self.net.greedy(self._obs(), self.env._get_info()["action_mask"])
            self.env.step(a)
            self.live.append(a)
            steps += 1
        return 1 if (not self.env.in_combat and self.env.gc.outcome != _LOSS) else 0


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    s = Searcher(sys.argv[2], sys.argv[3], sys.argv[4], seed)
    while True:
        msg = recv()
        if msg is None or msg[0] == "close":
            _shutdown()
        if msg[0] == "gen":
            rec = s.gen(msg[1] if len(msg) > 1 else 0.0)
            if rec is None:
                _shutdown()
            send(rec)
        elif msg[0] == "eval_greedy":
            send(("EVAL", s._play(msg[1], False)))
        elif msg[0] == "eval_search":
            send(("EVAL", s._play(msg[1], True)))
