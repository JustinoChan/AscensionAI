"""Collect a fixed dataset of the proven rollout-search teacher's decisions:
(obs_1235, teacher_argmax_action, action_mask, room_tier). Advances on the
teacher's own choices (teacher state distribution). Torch-free.

argv: [1] N samples  [2] out.npz  [3] sampling seed"""
import sys, warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sim_worker_search import Searcher, TOPK

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
out = sys.argv[2] if len(sys.argv) > 2 else "teacher_data.npz"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
s = Searcher("bank_train.jsonl", "fight_config.json", "policy_varL.npz", seed)

obs_l, act_l, mask_l, room_l = [], [], [], []
s._new_fight()
while len(obs_l) < N:
    if not s._restore():
        s._new_fight(); continue
    obs = s._obs()
    mask = np.asarray(s.env._get_info()["action_mask"], np.int8)
    legal = [i for i in range(128) if mask[i]]
    if not legal:
        s._new_fight(); continue
    cand = legal
    if len(legal) > TOPK:
        prior = s.net.logits(obs, mask)
        cand = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
    best, bq = cand[0], -1e18
    for a in cand:
        qa = s._rollout_q(a)
        if qa is not None and qa > bq:
            bq, best = qa, a
    if not s._restore():
        s._new_fight(); continue
    obs_l.append(obs.astype(np.float16))
    act_l.append(best)
    mask_l.append(mask)
    room_l.append(s.entry.get("room", "?"))
    s.env.step(best); s.live.append(best)
    done, _ = s._term_bonus()
    if done:
        s._new_fight()
    if len(obs_l) % 500 == 0:
        print("%s: %d" % (out, len(obs_l)), flush=True)

np.savez(out,
         obs=np.asarray(obs_l, np.float16),
         act=np.asarray(act_l, np.int16),
         mask=np.asarray(mask_l, np.int8),
         room=np.asarray(room_l))
print("DONE %s %d samples" % (out, len(obs_l)), flush=True)
