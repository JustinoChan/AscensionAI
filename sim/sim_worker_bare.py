"""BARE-mode env worker for the obs-fix-only control run: identical to
sim_worker.py except _get_reward is NOT overridden, so the reward is the
fork's original (floor_delta*10 + hp_delta + win 1000 / flat -100 loss) --
exact varB semantics on top of the FIXED observation encoding. The trainer
must run with mode=bare (shaping off) when using this worker.

Protocol (parent->worker):
    ('step', action_int) | ('reset',) | ('reset_seed', seed) | ('close',)
Worker->parent:
    (obs_f32[1200], reward_f32, done_bool, mask_i8[128], floor_int, outcome_int)

NEVER call slaythespire.getNNInterface() before StsEnv is constructed in this
process: with gym_sts_lightspeed already imported, an early call corrupts the
card-encode map -- the deck/hand/discard/draw obs regions collapse into the
fallback slot -- and the same corruption intermittently segfaults. The only
safe pattern is to take maxima from the env's own instance (env.nn) after
construction.
"""
import sys
import os
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
_VICTORY = slaythespire.GameOutcome.PLAYER_VICTORY
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
maxima = None  # set in __main__ from env.nn AFTER env construction (see above)


class SafeStsEnv(StsEnv):
    _last = None

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last = o
            return o
        except Exception:
            return self._last if self._last is not None else np.zeros(OBS, np.int32)


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


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = SafeStsEnv(ascension=0)
    maxima = np.maximum(
        np.asarray(env.nn.getObservationMaximums(), dtype=np.float32), 1.0)
    obs, info = env.reset(seed=seed)
    send(pack(obs, info, 0.0, False, info.get("floor", 0), 0))

    while True:
        msg = recv()
        if msg is None or msg[0] == "close":
            _shutdown()
        if msg[0] == "reset":
            obs, info = env.reset()
            send(pack(obs, info, 0.0, False, info.get("floor", 0), 0))
            continue
        if msg[0] == "reset_seed":
            obs, info = env.reset(seed=int(msg[1]))
            send(pack(obs, info, 0.0, False, info.get("floor", 0), 0))
            continue
        # step
        obs, r, term, trunc, info = env.step(int(msg[1]))
        done = bool(term or trunc)
        floor = int(env.gc.floor_num)
        outcome = 0
        if done:
            o = env.gc.outcome
            outcome = 1 if o == _VICTORY else (-1 if o == _LOSS else 0)
            obs, info = env.reset()
        send(pack(obs, info, float(r), done, floor, outcome))
