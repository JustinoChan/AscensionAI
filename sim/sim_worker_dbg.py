"""Env worker: imports ONLY the slaythespire extension (NO torch). Speaks a
length-prefixed pickle protocol over its real stdout pipe. Auto-resets on done.

Protocol (parent->worker):
    ('step', action_int) | ('reset',) | ('reset_seed', seed) | ('close',)
Worker->parent:
    (obs_f32[1200], reward_f32, done_bool, mask_i8[128], floor_int, outcome_int)
outcome: 0 ongoing/undecided at the moment, +1 victory, -1 loss (set on the step
that ends an episode, for the just-finished game). obs is pre-normalized.

Variant C worker-side reward terms (spec: docs/2026-06-11-sim-reward-overhaul-
design.md on the local machine):
  kept:    floor_delta*10, hp_delta*1, win +1000 (invalid-action -1 in step())
  changed: loss penalty -100/act (was flat -100)
  added:   fight-outcome bonus on victory (elite +30, boss +60; room type
           captured at fight entry -- scalar reads only)
The deck-quality potential is computed TRAINER-side from the obs deck region
(obs[19:239]): per-step gc.deck reads in this process destabilize the extension
(~1 segfault/game; the deck binding copies the card vector on every access).
Keep the worker's native surface at varB level -- scalar reads only.

The extension is also heap-layout sensitive to WHEN getNNInterface() is first
called: worker mode needs it at this import-sequence position, while importing
processes (tests) must not trigger it at all. __name__ gates both placements.
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

OBS = 1200
ELITE_BONUS = 30.0
BOSS_BONUS = 60.0
_VICTORY = slaythespire.GameOutcome.PLAYER_VICTORY
_LOSS = slaythespire.GameOutcome.PLAYER_LOSS
if __name__ == "__main__":
    maxima = np.maximum(
        np.asarray(slaythespire.getNNInterface().getObservationMaximums(),
                   dtype=np.float32), 1.0)
else:
    maxima = None


class SafeStsEnv(StsEnv):
    _last = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._was_in_combat = bool(self.in_combat)
        self._fight_room = ""
        return obs, info

    def _get_observation(self):
        try:
            o = super()._get_observation()
            self._last = o
            return o
        except Exception:
            return self._last if self._last is not None else np.zeros(OBS, np.int32)

    def _get_reward(self):
        reward = 0.0
        current_floor = self.gc.floor_num
        reward += (current_floor - self.last_floor) * 10.0
        self.last_floor = current_floor
        if self.in_combat and self.bc:
            current_hp = self.bc.player.cur_hp
        else:
            current_hp = self.gc.cur_hp
        reward += (current_hp - self.last_hp) * 1.0
        self.last_hp = current_hp
        if self.gc.outcome == _VICTORY:
            reward += 1000.0
        elif self.gc.outcome == _LOSS:
            reward -= 100.0 / max(1, int(self.gc.act))

        if self.in_combat and not self._was_in_combat:
            self._fight_room = str(self.gc.cur_room)
        elif self._was_in_combat and not self.in_combat:
            if self.gc.outcome != _LOSS:
                if "ELITE" in self._fight_room:
                    reward += ELITE_BONUS
                elif "BOSS" in self._fight_room:
                    reward += BOSS_BONUS
            self._fight_room = ""
        self._was_in_combat = bool(self.in_combat)
        return reward


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
    obs, info = env.reset(seed=seed)
    _c = np.asarray(obs[19:239])
    _nz = np.nonzero(_c)[0]
    print("DBG raw deck slots:", [(int(i), int(_c[i])) for i in _nz],
          file=sys.stderr, flush=True)
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
