"""Mainline failure analytics: run the full-run search-combat agent over the
fixed eval seeds and log per-run death diagnostics, then print a breakdown of
WHERE and TO WHAT runs die (now that combat is strong). Reuses FullRunEval.

argv: [1] out.csv (default fullrun_deaths.csv)  [2] n_seeds (default 48)
"""
import sys
import csv
import warnings
import os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import slaythespire
from eval_fullrun_search import FullRunEval, MAX_GAME_STEPS, EVAL_SEEDS

_VIC = slaythespire.GameOutcome.PLAYER_VICTORY


class AnalyticsEval(FullRunEval):
    def _snapshot(self):
        bc = self.env.bc
        if bc is None:
            return None
        try:
            ms = [m for m in bc.monsters.monsters if m.is_alive()]
            return {"n": len(ms),
                    "ehp": int(sum(m.cur_hp for m in ms)),
                    "emax": int(sum(m.max_hp for m in ms)),
                    "room": str(self.env.gc.cur_room).split(".")[-1],
                    "floor": int(self.env.gc.floor_num),
                    "act": int(self.env.gc.act)}
        except Exception:
            return None

    def play_diag(self, seed):
        self.seed = seed
        self.history = []
        try:
            self.env.reset(seed=int(seed))
        except Exception:
            return None
        last_combat = None
        for _ in range(MAX_GAME_STEPS):
            if self.env.in_combat:
                snap = self._snapshot()
                if snap is not None:
                    last_combat = snap
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
        won = self.env.gc.outcome == _VIC
        d = {"seed": seed, "win": int(won),
             "floor": int(self.env.gc.floor_num), "act": int(self.env.gc.act)}
        try:
            d["max_hp"] = int(self.env.gc.max_hp)
        except Exception:
            d["max_hp"] = -1
        try:
            d["gold"] = int(self.env.gc.gold)
        except Exception:
            d["gold"] = -1
        try:
            d["deck"] = len(self.env.gc.deck)
        except Exception:
            d["deck"] = -1
        if not won and last_combat is not None:
            frac = (last_combat["ehp"] / last_combat["emax"]
                    if last_combat["emax"] > 0 else 1.0)
            d.update({"death_room": last_combat["room"],
                      "death_floor": last_combat["floor"],
                      "n_enemies": last_combat["n"],
                      "enemy_hp_left": last_combat["ehp"],
                      "enemy_hp_frac": round(frac, 3)})
        else:
            d.update({"death_room": "WON" if won else "?", "death_floor": d["floor"],
                      "n_enemies": 0, "enemy_hp_left": 0, "enemy_hp_frac": 0.0})
        return d


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fullrun_deaths.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    ev = AnalyticsEval("policy_varL.npz", "policy_varE.npz", "search")
    rows = []
    cols = ["seed", "win", "floor", "act", "max_hp", "gold", "deck",
            "death_room", "death_floor", "n_enemies", "enemy_hp_left",
            "enemy_hp_frac"]
    for i, seed in enumerate(EVAL_SEEDS[:n]):
        d = ev.play_diag(seed)
        if d is None:
            print("seed %d FAILED" % seed, flush=True)
            continue
        rows.append(d)
        print("[%2d/%2d] seed %d win=%d floor=%2d act=%d deck=%d hp=%d "
              "death=%s@%s ehp_frac=%.2f" % (
                  i + 1, n, seed, d["win"], d["floor"], d["act"], d["deck"],
                  d["max_hp"], d["death_room"], d["death_floor"],
                  d["enemy_hp_frac"]), flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- breakdown ----
    losses = [r for r in rows if not r["win"]]
    wins = [r for r in rows if r["win"]]
    print("\n==== SUMMARY (%d runs) ====" % len(rows))
    print("wins: %d  avg floor: %.1f" % (len(wins),
          np.mean([r["floor"] for r in rows]) if rows else 0))
    print("-- deaths by act --")
    for a in (1, 2, 3):
        da = [r for r in losses if r["death_floor"] and r["act"] == a]
        if da:
            print("  act %d: %2d deaths  rooms: %s" % (
                a, len(da),
                {x: sum(1 for r in da if r["death_room"] == x)
                 for x in set(r["death_room"] for r in da)}))
    print("-- deaths by room type --")
    for rt in ("MONSTER", "ELITE", "BOSS"):
        dr = [r for r in losses if r["death_room"] == rt]
        if dr:
            print("  %-7s: %2d deaths  avg enemy_hp_frac_left=%.2f (0=almost won, "
                  "1=crushed)  avg death_floor=%.0f" % (
                      rt, len(dr), np.mean([r["enemy_hp_frac"] for r in dr]),
                      np.mean([r["death_floor"] for r in dr])))
    print("-- deck/hp/gold: wins vs losses --")
    for label, grp in (("win", wins), ("loss", losses)):
        if grp:
            print("  %-4s: deck=%.1f max_hp=%.0f gold=%.0f" % (
                label, np.mean([r["deck"] for r in grp]),
                np.mean([r["max_hp"] for r in grp]),
                np.mean([r["gold"] for r in grp])))


if __name__ == "__main__":
    main()
