import sys, numpy as np
sys.path.insert(0, ".")
from sim_worker_search import Searcher, GAMMA
s = Searcher("bank_train.jsonl", "fight_config.json", "policy_varL.npz", 7)
# advance to a real combat decision
assert s._new_fight()
assert s._restore()
mask = np.asarray(s.env._get_info()["action_mask"], np.int8)
legal = [i for i in range(128) if mask[i]]
a = legal[0]
# manual recompute of _q_value(a)
s._restore(); _, r, _, _, _ = s.env.step(int(a))
done, tb = s._term_bonus()
manual = (r + tb) if done else (r + GAMMA * s.net.value_denorm(s._obs()))
got = s._q_value(a)
print("manual=%.5f got=%.5f done=%s" % (manual, got, done))
assert abs(manual - got) < 1e-4, "q_value mismatch"
# end-to-end gen() yields finite, valid targets
rec = s.gen(0.5)
obs, pi, z, m, d, oc = rec
assert obs.shape == (1235,) and pi.shape == (128,)
assert np.isfinite(z) and abs(pi.sum() - 1.0) < 1e-3
print("gen ok: z=%.3f pi_sum=%.4f" % (float(z), float(pi.sum())))
# sanity: value-bootstrap search not worse than greedy on 8 eval bosses
ev = [e for e in s.eval_bank if e.get("room") == "BOSS"][:8]
gw = sw = 0
for _e in ev:
    j = s.eval_bank.index(_e)
    gw += s._play(j, False)
    sw += s._play(j, True)
print("8-boss (policy_varL.npz has no zmean/zstd -> value on wrong scale, "
      "INFORMATIONAL only): greedy %d/8 search %d/8" % (gw, sw))
print("VALUE-TEACHER GATE PASS (mechanics: manual==got, gen valid). "
      "Teacher quality is validated in the Task 3 smoke with real zmean/zstd.")
