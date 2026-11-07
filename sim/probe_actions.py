"""Phase 1: action-semantics probe. (a) Locate + verify the obs deck region
(size and Strike/Defend slots) so the override can read deck state from the obs
(segfault-safe) instead of gc.deck. (b) For each non-combat screen type, execute
each legal action in a scratch replay and label it by macro-state delta
(skip / take-card / buy-card / remove / relic / potion / rest / smith / path)."""
import warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import slaythespire
from eval_fullrun_search import FullRunEval, MAX_GAME_STEPS

DECK_LO, DECK_HI = 19, 239   # candidate deck region (to verify)


class Probe(FullRunEval):
    def _raw(self):
        return np.asarray(self.env._get_observation(), np.float32)

    def _deck_from_obs(self):
        r = self._raw()[DECK_LO:DECK_HI]
        size = float(r.sum())
        upg = float(r[1::2].sum())   # odd slots = upgraded (per prior finding)
        return size, upg

    def _macro(self):
        gc = self.env.gc
        size, upg = self._deck_from_obs()
        return {"deck": size, "upg": upg, "gold": int(gc.gold),
                "relics": len(gc.relics), "hp": int(gc.cur_hp),
                "mhp": int(gc.max_hp)}

    def _label(self, a, b, ss):
        dd = b["deck"] - a["deck"]
        if dd >= 1:
            return "TAKE/BUY_CARD(+%.0f)" % dd
        if dd <= -1:
            return "REMOVE_CARD(%.0f)" % dd
        if b["relics"] > a["relics"]:
            return "GAIN/BUY_RELIC"
        if b["upg"] > a["upg"]:
            return "SMITH(upgrade)"
        if b["hp"] > a["hp"]:
            return "REST/HEAL(+%d)" % (b["hp"] - a["hp"])
        if b["gold"] < a["gold"]:
            return "SPEND_GOLD(-%d)[potion?]" % (a["gold"] - b["gold"])
        if b["hp"] < a["hp"]:
            return "LOSE_HP(-%d)" % (a["hp"] - b["hp"])
        return "SKIP/PROCEED/PATH"

    def run(self, seed):
        self.seed = seed
        self.history = []
        self.env.reset(seed=int(seed))
        # --- (a) verify deck region at the start (deck = 5 Strike,4 Defend,1 Bash) ---
        try:
            gc_size = len(self.env.gc.deck)
        except Exception:
            gc_size = -1
        obs_size, _ = self._deck_from_obs()
        print("START: gc.deck=%d  obs_region_sum[%d:%d]=%.0f" % (
            gc_size, DECK_LO, DECK_HI, obs_size))
        raw = self._raw()
        nz = [(i, int(raw[i])) for i in range(0, 320) if raw[i] > 0]
        print("nonzero obs[0:320] (locate Strike=5/Defend=4/Bash=1):", nz)

        seen = set()
        steps_logged = 0
        for _ in range(MAX_GAME_STEPS):
            ss = str(self.env.gc.screen_state).split(".")[-1]
            if (not self.env.in_combat and ss not in seen
                    and ss not in ("INVALID",)):
                seen.add(ss)
                acts = self.env.gc.get_possible_actions()
                print("\n--- %s (floor %d room %s gold %d deck %.0f hp %d/%d) %d acts ---"
                      % (ss, self.env.gc.floor_num,
                         str(self.env.gc.cur_room).split(".")[-1], self.env.gc.gold,
                         self._deck_from_obs()[0], self.env.gc.cur_hp,
                         self.env.gc.max_hp, len(acts)))
                if ss in ("REWARDS", "CARD_SELECT", "SHOP_ROOM"):
                    try:
                        cr = self.env.gc.get_card_reward()
                        print("   get_card_reward():", [
                            (str(c.id).split(".")[-1], str(c.rarity).split(".")[-1],
                             str(c.type).split(".")[-1]) for c in cr])
                    except Exception as ex:
                        print("   get_card_reward() ->", ex)
                base = self._macro()
                for i in range(min(len(acts), 12)):
                    if not self._restore():
                        break
                    a0 = self._macro()
                    try:
                        self._ff_step(i)
                        a1 = self._macro()
                        print("   act[%d] bits=%d -> %s" % (
                            i, acts[i].bits, self._label(a0, a1, ss)))
                    except Exception as ex:
                        print("   act[%d] -> ERR %s" % (i, ex))
                self._restore()
                steps_logged += 1
            # advance with varE macro / search combat
            if self.env.in_combat:
                a = self._search_action()
                if a is None:
                    break
            else:
                a = self.vare.act(self._obs1200(), self._mask())
            _, r, term, trunc, _ = self.env.step(int(a))
            self.history.append(int(a))
            if term or trunc or steps_logged >= 12:
                break


if __name__ == "__main__":
    Probe("policy_varL.npz", "policy_varE.npz", "search").run(10007)
