"""Diagnostic for the combat/search v2 phase. Re-runs the mainline_v2 agent
(macro mode, varN combat) but instruments the 1-ply combat search to record,
for every LOSING Act-2 monster fight, the per-decision top-2 Q-gap, top-3 Q
values, player HP/block, incoming damage, and a lethal-this-turn flag.

This discriminates whether the ~58 Act-2 monster deaths are:
  (a) close decisions the 1-ply rollout mis-ranks  -> deeper/averaged search helps
  (b) baked-in lethal pressure (we ENTER the death turn already dead)  -> search can't help
  (c) confident-but-wrong picks (large gap, not lethal) -> value/teacher error

Read-only w.r.t. the deployed harness: subclasses V2 and overrides ONLY the
search action to log; the argmax it returns is identical to mainline_v2, so the
trajectory (and therefore every death) is exactly the real agent's.

argv: [1] seed   (mode is always macro; toggles via env, same as mainline_v2)"""
import sys
import json
import os
import warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: F401,E402
import slaythespire  # noqa: F401,E402
from eval_fullrun_search import TOPK  # noqa: E402
from eval_fullrun_analytics_v2 import V2  # noqa: E402
from combat_features import combat_block  # noqa: E402


class V2Diag(V2):
    def _log_decision(self, qs):
        """Record the live combat decision (after _restore) into self._slog."""
        bc = self.env.bc
        if bc is None or not qs:
            return
        try:
            cb = combat_block(self.env)              # [33]=incoming/40 [34]=net/40
            vals = sorted((qa for _, qa in qs), reverse=True)
            gap = (vals[0] - vals[1]) if len(vals) >= 2 else None
            ms = [m for m in bc.monsters.monsters if m.is_alive()]
            php = int(bc.player.cur_hp)
            net = float(cb[34] * 40.0)               # incoming after current block
            self._slog.append(dict(
                floor=int(self.env.gc.floor_num),
                act=int(self.env.gc.act),
                room=str(self.env.gc.cur_room).split(".")[-1],
                php=php, mhp=int(self.env.gc.max_hp), pblk=int(bc.player.block),
                ehp=int(sum(m.cur_hp for m in ms)), nalive=len(ms),
                inc=round(float(cb[33] * 40.0), 1), net=round(net, 1),
                lethal=int(net >= php), ncand=len(qs),
                gap=(round(float(gap), 3) if gap is not None else None),
                top3=[round(float(v), 2) for v in vals[:3]]))
        except Exception:
            pass

    def _search_action(self):
        # mirrors FullRunEval._search_action exactly (same argmax), plus logging
        mask = self._mask()
        legal = [i for i in range(128) if mask[i]]
        if len(legal) > TOPK:
            prior = self.net.logits(self._obs1235(), mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
        qs = []
        for a in legal:
            qa = self._rollout_q(a)
            if qa is not None:
                qs.append((a, qa))
        if not self._restore():            # restore live state for the real step
            return None
        self._log_decision(qs)
        if not qs:
            return legal[0]
        best, bq = qs[0]                   # first-max wins (same tie-break as base)
        for a, qa in qs[1:]:
            if qa > bq:
                bq, best = qa, a
        return best

    def play_diag(self, seed):
        self._slog = []
        out = self.play_v2(seed)
        dfight = []
        if (not out["win"] and out.get("death_act") == 2
                and "MONSTER" in str(out.get("death_room", ""))):
            if self._slog:                 # death fight = trailing same-floor records
                df = self._slog[-1]["floor"]
                for rec in reversed(self._slog):
                    if rec["floor"] == df:
                        dfight.append(rec)
                    else:
                        break
                dfight.reverse()
        out["death_fight"] = dfight
        return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    combat_npz = os.environ.get("COMBAT_NPZ", "policy_varN.npz")
    ev = V2Diag(combat_npz, "policy_varE.npz", "macro")
    print(json.dumps(ev.play_diag(seed)), flush=True)
