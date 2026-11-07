"""Fight-gym env worker: each episode is ONE fight, rebuilt from a harvested
bank entry (real deck/relics/potions/HP) by deterministic action replay, and
optionally re-rolled to a different same-tier encounter via the two-arg
BattleContext.init binding.

argv: [1] sampling seed  [2] bank_train.jsonl path  [3] fight_config.json path
(bank_eval.jsonl is loaded from the train bank's directory)

Protocol (parent->worker):
    ('step', action_int) | ('reset',) | ('reset_seed', eval_entry_idx) | ('close',)
Worker->parent (6-field, identical to sim_worker.py):
    (obs_f32[1200], reward_f32, done_bool, mask_i8[128], floor_int, outcome_int)
outcome: +1 fight won, -1 player died, 0 truncated at the step cap.
reset_seed N loads held-out eval entry N % len(eval_bank), never re-rolled.

Reward: per-step combat HP delta x1.0; terminal +50 win / -50 loss / 0 cap.

NEVER call slaythespire.getNNInterface() before StsEnv is constructed in this
process (corrupts the card-encode map; see sim_worker.py). maxima comes from
env.nn after construction.
"""
import sys
import os
import json
import struct
import pickle
import warnings
warnings.filterwarnings("ignore")

_proto = sys.stdout.buffer
_inp = sys.stdin.buffer

import numpy as np
import gym_sts_lightspeed  # noqa: F401
from gym_sts_lightspeed.envs.sts_env import StsEnv
import slaythespire
from combat_features import combat_block

OBS = 1200
MAX_FIGHT_STEPS = 500
WIN_BONUS, LOSS_PENALTY = 50.0, -50.0
_VICTORY = slaythespire.GameOutcome.PLAYER_VICTORY
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
_UNDECIDED = slaythespire.GameOutcome.UNDECIDED
maxima = None  # set in __main__ from env.nn AFTER env construction


class FightEnv(StsEnv):
    _last = None
    _last_enemy_hp = 0

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last = o
            return o
        except Exception:
            return self._last if self._last is not None else np.zeros(OBS, np.int32)

    def _enemy_hp(self):
        if not self.in_combat or self.bc is None:
            return self._last_enemy_hp
        try:
            return sum(m.cur_hp for m in self.bc.monsters.monsters if m.is_alive())
        except Exception:
            return self._last_enemy_hp

    def _get_reward(self):
        # DENSE combat reward: per-step player HP delta (defense) + damage dealt
        # to enemies (offense). Gives per-action credit and low-variance returns
        # so the value head can learn and the policy can improve combat play
        # (sparse terminal-only reward gave ev~0 / frozen win rates).
        if self.in_combat and self.bc:
            cur = self.bc.player.cur_hp
        else:
            cur = self.gc.cur_hp
        r = (cur - self.last_hp) * 1.0
        self.last_hp = cur
        eh = self._enemy_hp()
        r += max(0, self._last_enemy_hp - eh) * 0.5   # reward damage dealt
        self._last_enemy_hp = eh
        return r

    def reset_to_entry(self, entry, encounter=None):
        """Rebuild the harvested fight-entry state by replaying the recorded
        prefix through the env's own step() (exactness over speed), then
        optionally re-roll the encounter. Returns (obs, info) or None."""
        try:
            super().reset(seed=int(entry["seed"]))
            for a in entry["prefix"]:
                self.step(int(a))
                if self.gc.outcome != _UNDECIDED:
                    return None
            if not self.in_combat or self.bc is None:
                return None
            if encounter is not None:
                # init takes const GameContext& -> safe to discard the first
                # bc and re-init a fresh one against the chosen encounter
                bc = slaythespire.BattleContext()
                bc.init(self.gc, encounter)
                self.bc = bc
            self.last_hp = self.bc.player.cur_hp
            self._last_enemy_hp = self._enemy_hp()
            obs = self._get_observation()
            info = self._get_info()
            return obs, info
        except Exception:
            return None


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


def pack(obs, info, reward, done, floor, outcome):
    o = (np.asarray(obs, dtype=np.float32) / maxima).astype(np.float32)
    o = np.concatenate([o, combat_block(env)]).astype(np.float32)
    m = np.asarray(info["action_mask"], dtype=np.int8)
    return (o, np.float32(reward), bool(done), m, int(floor), int(outcome))


def _shutdown():
    try:
        _proto.flush()
    except Exception:
        pass
    os._exit(0)


def _load_jsonl(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


if __name__ == "__main__":
    samp_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    bank_path = sys.argv[2]
    cfg = json.load(open(sys.argv[3]))
    bank = _load_jsonl(bank_path)
    eval_bank = _load_jsonl(os.path.join(
        os.path.dirname(os.path.abspath(bank_path)), "bank_eval.jsonl"))
    rng = np.random.default_rng(samp_seed)

    tw = cfg["tier_weights"]
    los = float(cfg["loss_oversample"])
    weights = np.array([tw.get(e["room"], 0.0) * (los if e["lost"] else 1.0)
                        for e in bank], dtype=np.float64)
    keep = weights > 0
    bank = [e for e, k in zip(bank, keep) if k]
    weights = weights[keep]
    probs = weights / weights.sum()
    pools = {k: [getattr(slaythespire.MonsterEncounter, n) for n in v]
             for k, v in cfg["reroll_pools"].items()}
    reroll_p = float(cfg["encounter_reroll_prob"])

    env = FightEnv(ascension=0)
    maxima = np.maximum(
        np.asarray(env.nn.getObservationMaximums(), dtype=np.float32), 1.0)

    def reset_any():
        for _ in range(10):
            e = bank[int(rng.choice(len(bank), p=probs))]
            enc = None
            if e["room"] in pools and rng.random() < reroll_p:
                pool = pools[e["room"]]
                enc = pool[int(rng.integers(len(pool)))]
            out = env.reset_to_entry(e, enc)
            if out is not None:
                return out
        os._exit(1)   # parent's retry-until-healthy relaunches us

    obs, info = reset_any()
    send(pack(obs, info, 0.0, False, env.gc.floor_num, 0))
    fight_steps = 0

    while True:
        msg = recv()
        if msg is None or msg[0] == "close":
            _shutdown()
        if msg[0] == "reset":
            obs, info = reset_any()
            fight_steps = 0
            send(pack(obs, info, 0.0, False, env.gc.floor_num, 0))
            continue
        if msg[0] == "reset_seed":
            e = eval_bank[int(msg[1]) % len(eval_bank)]
            out = env.reset_to_entry(e, None)
            if out is None:
                out = reset_any()   # broken eval entry: degrade, don't die
            obs, info = out
            fight_steps = 0
            send(pack(obs, info, 0.0, False, env.gc.floor_num, 0))
            continue
        # step
        obs, r, term, trunc, info = env.step(int(msg[1]))
        fight_steps += 1
        done = False
        outcome = 0
        if term or not env.in_combat:
            done = True
            if env.gc.outcome == _LOSS:
                outcome = -1
                r += LOSS_PENALTY
            elif not env.in_combat:
                outcome = 1
                r += WIN_BONUS
            # else: env wedged mid-combat -> truncation-like, outcome 0
        elif fight_steps >= MAX_FIGHT_STEPS:
            done = True
        floor = int(env.gc.floor_num)
        if done:
            obs, info = reset_any()
            fight_steps = 0
        send(pack(obs, info, float(r), done, floor, outcome))
