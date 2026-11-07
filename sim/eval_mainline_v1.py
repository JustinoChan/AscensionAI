"""v2 decision-log + death stats for ONE seed, with an optional NARROW macro
override (mode=macro). Combat = search. Non-combat = baseline varE, except in
macro mode where card-reward take/skip, card removal, and shop card-buys are
overridden by simple heuristics (everything else still defers to varE).

Decisions are attributed by macro-state DELTAS across screen transitions; deck
state is read from the obs deck region (obs[19:239]; Strike=151, Defend=57),
NOT gc.deck (segfault-safe). One process per seed.

argv: [1] seed  [2] mode = search | macro  (default search)"""
import sys, json, warnings, os
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import slaythespire
from eval_fullrun_search import FullRunEval, MAX_GAME_STEPS
from card_roles import classify

_VIC = slaythespire.GameOutcome.PLAYER_VICTORY
DECK_LO, DECK_HI = 19, 239
STRIKE_SLOT, DEFEND_SLOT = 151, 57
DECK_CAP = 22          # raise pick threshold sharply at/above this
SHOP_REMOVE_GOLD = 75  # rough min gold to attempt a shop remove
MIN_PLAYABLE = 7       # non-basic playable cards before strict hygiene kicks in
# Mainline_v1 defaults: elite-gating OFF (no-op), relic-buying OFF (neutral),
# role-scorer ON (deployed win). Combat net = policy_varN.npz (set in __main__).
GATE_ELITES = os.environ.get("GATE_ELITES", "0") == "1"  # toggle the path filter
RELIC_BUY = os.environ.get("RELIC_BUY", "0") == "1"      # toggle shop relic buying
ROLE_SCORE = os.environ.get("ROLE_SCORE", "1") == "1"    # toggle AoE/draw gap-filler
SMITH = os.environ.get("SMITH", "0") == "1"              # conservative boss-aware smith


class V2(FullRunEval):
    _last_deck = 10

    # ---- obs-derived deck stats (no gc.deck reads) ----
    def _deck_stats(self):
        r = np.asarray(self.env._get_observation(), np.float32)
        return (float(r[DECK_LO:DECK_HI].sum()),
                float(r[STRIKE_SLOT] + r[STRIKE_SLOT + 1]),
                float(r[DEFEND_SLOT] + r[DEFEND_SLOT + 1]))

    def _snap(self):
        d, _s, _de = self._deck_stats()
        self._last_deck = d
        gc = self.env.gc
        return {"deck": d, "gold": int(gc.gold), "relics": len(gc.relics),
                "hp": int(gc.cur_hp), "mhp": int(gc.max_hp),
                "upg": float(np.asarray(self.env._get_observation(), np.float32)[DECK_LO + 1:DECK_HI:2].sum())}

    def _deck_comp(self):
        try:
            deck = list(self.env.gc.deck)
            return {"size": len(deck),
                    "strikes": sum(1 for c in deck if c.is_strikeCard),
                    "defends": sum(1 for c in deck if "DEFEND" in str(c.id).upper()),
                    "attacks": sum(1 for c in deck if "ATTACK" in str(c.type).upper()),
                    "powers": sum(1 for c in deck if "POWER" in str(c.type).upper()),
                    "upgraded": sum(1 for c in deck if c.upgraded)}
        except Exception:
            d, s, de = self._deck_stats()
            return {"size": int(d), "strikes": int(s), "defends": int(de),
                    "attacks": -1, "powers": -1, "upgraded": -1}

    def _deck_roles(self):
        """Role composition of the current deck (for the Act-2-entry diagnostic)."""
        try:
            deck = list(self.env.gc.deck)
        except Exception:
            return None
        c = dict(size=len(deck), starters=0, upgraded=0, damage=0, block=0,
                 aoe=0, scaling=0, draw_energy=0, powers=0, junk=0, qsum=0.0)
        for card in deck:
            name = str(card.id).split(".")[-1]
            ctype = str(card.type).split(".")[-1]
            roles = classify(name, ctype)
            if card.upgraded:
                c["upgraded"] += 1
            if "starter" in roles:
                c["starters"] += 1
            if ctype == "ATTACK":
                c["damage"] += 1
            if "block" in roles:
                c["block"] += 1
            if "aoe" in roles:
                c["aoe"] += 1
            if "scaling" in roles:
                c["scaling"] += 1
            if ("draw" in roles) or ("energy" in roles):
                c["draw_energy"] += 1
            if ctype == "POWER":
                c["powers"] += 1
            if "junk" in roles:
                c["junk"] += 1
            c["qsum"] += self._qual(card)
        return c

    def _deck_role_counts(self):
        """(aoe_count, draw/energy_count) of the current deck. Fail-safe (9, 9)
        so a read failure simply disables the gap-filler for that decision."""
        try:
            deck = list(self.env.gc.deck)
        except Exception:
            return (9, 9)
        aoe = dr = 0
        for card in deck:
            rs = classify(str(card.id).split(".")[-1], str(card.type).split(".")[-1])
            if "aoe" in rs:
                aoe += 1
            if ("draw" in rs) or ("energy" in rs):
                dr += 1
        return (aoe, dr)

    # ---- combat snapshot for death stats ----
    def _csnap(self):
        bc = self.env.bc
        if bc is None:
            return None
        try:
            ms = [m for m in bc.monsters.monsters if m.is_alive()]
            return {"n": len(ms), "ehp": int(sum(m.cur_hp for m in ms)),
                    "emax": int(sum(m.max_hp for m in ms)),
                    "room": str(self.env.gc.cur_room).split(".")[-1],
                    "floor": int(self.env.gc.floor_num),
                    "act": int(self.env.gc.act)}
        except Exception:
            return None

    # ---- macro override ----
    def _upg(self):
        r = np.asarray(self.env._get_observation(), np.float32)
        return float(r[DECK_LO + 1:DECK_HI:2].sum())

    def _offered(self):
        try:
            return list(self.env.gc.get_card_reward())
        except Exception:
            return []

    def _qual(self, c):
        if c is None:
            return 0
        t = str(c.type).upper(); r = str(c.rarity).upper()
        if "CURSE" in t or "STATUS" in t:
            return -1
        s = 0
        if "POWER" in t:
            s += 2
        if "RARE" in r:
            s += 3
        elif "UNCOMMON" in r:
            s += 2
        elif "COMMON" in r:
            s += 1
        return s

    def _macro_action(self):
        gc = self.env.gc
        ss = str(gc.screen_state).split(".")[-1]
        handled = ("REWARDS", "CARD_SELECT", "SHOP_ROOM", "MAP_SCREEN")
        if SMITH:
            handled = handled + ("REST_ROOM",)
        if ss not in handled:
            return self.vare.act(self._obs1200(), self._mask())
        acts = gc.get_possible_actions()
        n = len(acts)
        if n <= 1:
            return 0
        bd, bs, bdef = self._deck_stats()
        bgold = int(gc.gold); brel = len(gc.relics)
        bhp = int(gc.cur_hp); bu = self._upg()
        labels = []
        for i in range(min(n, 12)):
            if not self._restore():
                return self.vare.act(self._obs1200(), self._mask())
            try:
                self._ff_step(i)
                d, s, de = self._deck_stats()
                labels.append({"i": i, "dd": d - bd, "sd": s - bs, "ded": de - bdef,
                               "rd": len(self.env.gc.relics) - brel,
                               "hp_d": int(self.env.gc.cur_hp) - bhp,
                               "ud": self._upg() - bu,
                               "to": str(self.env.gc.screen_state).split(".")[-1],
                               "room": str(self.env.gc.cur_room).split(".")[-1]})
            except Exception:
                pass
        self._restore()
        if not labels:
            return self.vare.act(self._obs1200(), self._mask())
        return self._choose(ss, labels, bd, bs, bdef, bgold, int(self.env.gc.act))

    def _choose(self, ss, labels, deck, strike, defend, gold, act):
        # "strict" lean hygiene only once the deck is developed (enough non-basic
        # playable cards) OR past the Act-1 boss. Before that, prioritize early
        # damage/block density so a thin deck doesn't fizzle at the Act-1 boss.
        nonbasic = deck - strike - defend
        strict = (act >= 2) or (nonbasic >= MIN_PLAYABLE)
        if ss == "MAP_SCREEN":
            return self._choose_path(labels, deck, strike, defend, act)
        # Conservative boss-aware smith: only when nearly full HP and not right
        # before an act boss (where the bar is even higher). Otherwise rest.
        if SMITH and ss == "REST_ROOM":
            gc = self.env.gc
            hp_frac = gc.cur_hp / max(1, gc.max_hp)
            fl = int(gc.floor_num)
            near_boss = (14 <= fl <= 16) or (31 <= fl <= 34) or (47 <= fl <= 51)
            bar = 0.95 if near_boss else 0.85
            smith = [L for L in labels if L["to"] == "CARD_SELECT" and L["dd"] == 0]
            rest = [L for L in labels if L["hp_d"] > 0]
            if hp_frac >= bar and smith:
                return smith[0]["i"]
            if rest:
                return rest[0]["i"]
            return self.vare.act(self._obs1200(), self._mask())
        # Upgrade-select (smith picker): complete the upgrade we initiated.
        if SMITH:
            upg = [L for L in labels if L.get("ud", 0) >= 1 and L["dd"] == 0]
            if upg:
                return upg[0]["i"]
        removes = [L for L in labels if L["dd"] <= -1]
        takes = sorted([L for L in labels if L["dd"] >= 1], key=lambda x: x["i"])
        noops = [L for L in labels if L["dd"] == 0 and L["rd"] == 0]
        leave = [L for L in noops if L["to"] == "MAP_SCREEN"]
        skip = leave or noops          # prefer leaving to the map; else any no-op
        # Removal selection (a CARD_SELECT showing deck-1 options): Strike, then
        # Defend; but only once strict (don't thin early damage); never remove a
        # good (non-basic) card.
        if removes:
            if strict:
                for key in ("sd", "ded"):
                    c = [L for L in removes if L[key] <= -1]
                    if c:
                        return c[0]["i"]
            return (skip or removes)[0]["i"]
        # Shop: never buy cards (bloat). Use the remove service if strict +
        # affordable, else LEAVE explicitly (must not defer to varE, which buys).
        if ss == "SHOP_ROOM":
            # Shop v2: ONE remove first (preserve the deck-hygiene lever that won
            # the first runs; cap 1/shop), then convert gold into power by buying
            # an affordable relic, then leave. Never buy filler cards.
            fl = int(self.env.gc.floor_num)
            opener = [L for L in noops if L["to"] == "CARD_SELECT"]
            if (strict and opener and (strike + defend) >= 1
                    and gold >= SHOP_REMOVE_GOLD and self._shop_removed_floor != fl):
                self._shop_removed_floor = fl
                return opener[0]["i"]
            if RELIC_BUY:
                relic_buys = [L for L in labels if L["rd"] >= 1]
                if relic_buys:
                    return relic_buys[0]["i"]
            if skip:
                return skip[0]["i"]
            return self.vare.act(self._obs1200(), self._mask())
        # Card reward: strict -> premium (uncommon+/power) or a non-junk common
        # while small. ROLE-AWARE gap-filler (narrow): if the deck is MISSING AoE
        # or draw/energy, take an offered card that fills it (even a common);
        # otherwise fall through to the strict rule -- never relax drafting
        # globally (don't re-bloat).
        if takes:
            offered = self._offered()
            need_aoe = need_draw = False
            # only once the deck is developed / entering Act 2 (strict) -- don't
            # over-draft weak gap-fillers in fragile early Act 1.
            if ROLE_SCORE and strict and ss in ("REWARDS", "CARD_SELECT"):
                aoe_have, draw_have = self._deck_role_counts()
                need_aoe = aoe_have <= 1
                need_draw = draw_have == 0
            best_i, best_q, best_fills = None, -9, False
            for k, L in enumerate(takes):
                c = offered[k] if k < len(offered) else None
                q = self._qual(c)
                fills = False
                if c is not None and (need_aoe or need_draw):
                    rs = classify(str(c.id).split(".")[-1], str(c.type).split(".")[-1])
                    if need_aoe and "aoe" in rs:
                        fills = True
                    if need_draw and (("draw" in rs) or ("energy" in rs)):
                        fills = True
                if (1 if fills else 0, q) > (1 if best_fills else 0, best_q):
                    best_i, best_q, best_fills = L["i"], q, fills
            if best_fills:
                take = best_q >= 1
            elif strict:
                take = best_q >= 2 or (deck < 16 and best_q >= 1)
            else:
                take = best_q >= 1
            if take and best_i is not None:
                return best_i
            if skip:
                return skip[0]["i"]
        return self.vare.act(self._obs1200(), self._mask())

    def _choose_path(self, labels, deck, strike, defend, act):
        """Risk-priced elite/path filter (NOT an elite ban). Each map option is
        labeled by the room it leads to (scratch-step). Only when an elite is
        unaffordable AND a safer node exists do we reroute; otherwise defer to
        the baseline policy so relic-bearing elites are still taken when the run
        can pay for them."""
        if not GATE_ELITES:
            return self.vare.act(self._obs1200(), self._mask())
        gc = self.env.gc
        hp_frac = gc.cur_hp / max(1, gc.max_hp)
        r = np.asarray(self.env._get_observation(), np.float32)
        upg = float(r[DECK_LO + 1:DECK_HI:2].sum())
        nonbasic = deck - strike - defend
        elite = [L for L in labels if "ELITE" in L.get("room", "")]
        nonelite = [L for L in labels if "ELITE" not in L.get("room", "")]
        if not elite or not nonelite:
            return self.vare.act(self._obs1200(), self._mask())
        if act <= 1:
            afford = hp_frac >= 0.70 or nonbasic >= 7
        else:
            afford = hp_frac >= 0.75 and nonbasic >= 8 and upg >= 2
        if afford:
            return self.vare.act(self._obs1200(), self._mask())
        order = {"REST": 0, "REST_ROOM": 0, "SHOP": 1, "SHOP_ROOM": 1,
                 "TREASURE": 2, "EVENT": 3, "EVENT_SCREEN": 3, "MONSTER": 4}
        nonelite.sort(key=lambda L: order.get(L.get("room", ""), 5))
        self._avoided += 1
        return nonelite[0]["i"]

    # ---- play one run with logging + (optional) override ----
    def play_v2(self, seed):
        self.seed = seed
        self.history = []
        self._avoided = 0
        self._shop_removed_floor = -1
        self.env.reset(seed=int(seed))
        macro = self.mode == "macro"
        m = dict(cards_taken=0, cards_skipped=0, card_screens=0, removes=0,
                 shops=0, gold_spent=0, shop_cards=0, shop_relics=0,
                 shop_skip_rich=0, campfires=0, rests=0, smiths=0, elites=0,
                 event_hp_gain=0, event_hp_loss=0,
                 hp_lost_monster=0, hp_lost_elite=0, hp_lost_boss=0,
                 n_monster=0, n_elite=0, n_boss=0,
                 campfire_hp_sum=0.0, campfire_n=0,
                 boss_entry_hp_sum=0.0, boss_entry_n=0)
        prev_ss = str(self.env.gc.screen_state).split(".")[-1]
        entry = self._snap()
        last_combat = None
        battle_room = None
        battle_entry_hp = 0
        battle_act = 1
        battle_steps = 0
        act2_roles = None
        fb = {}

        def _bucket(a, rm):
            k = "%d_%s" % (a, rm)
            if k not in fb:
                fb[k] = [0, 0, 0.0, 0, 0.0]  # survived, died, hp_lost, turns, ehp_sum
            return fb[k]

        def attribute(ss, a, b):
            dd = b["deck"] - a["deck"]
            if ss in ("REWARDS", "CARD_SELECT"):
                if ss == "CARD_SELECT":
                    m["card_screens"] += 1
                if dd > 0:
                    m["cards_taken"] += dd
                elif ss == "CARD_SELECT" and dd == 0:
                    m["cards_skipped"] += 1
                if dd < 0:
                    m["removes"] += -dd
            elif ss == "SHOP_ROOM":
                m["shops"] += 1
                spent = max(0, a["gold"] - b["gold"])
                m["gold_spent"] += spent
                if dd > 0:
                    m["shop_cards"] += dd
                if dd < 0:
                    m["removes"] += -dd
                if b["relics"] > a["relics"]:
                    m["shop_relics"] += b["relics"] - a["relics"]
                if a["gold"] >= 200 and spent < 50:
                    m["shop_skip_rich"] += 1
            elif ss == "REST_ROOM":
                m["campfires"] += 1
                if b["hp"] > a["hp"]:
                    m["rests"] += 1
                elif b["upg"] > a["upg"]:
                    m["smiths"] += 1
            elif ss == "EVENT_SCREEN":
                if dd < 0:
                    m["removes"] += -dd
                if b["hp"] > a["hp"]:
                    m["event_hp_gain"] += b["hp"] - a["hp"]
                elif b["hp"] < a["hp"]:
                    m["event_hp_loss"] += a["hp"] - b["hp"]

        for _ in range(MAX_GAME_STEPS):
            cur_ss = str(self.env.gc.screen_state).split(".")[-1]
            if cur_ss != prev_ss:
                ex = self._snap()
                attribute(prev_ss, entry, ex)
                if cur_ss == "BATTLE":
                    battle_room = str(self.env.gc.cur_room).split(".")[-1]
                    battle_entry_hp = ex["hp"]
                    battle_act = int(self.env.gc.act)
                    battle_steps = 0
                    if "ELITE" in battle_room:
                        m["elites"] += 1
                    if "BOSS" in battle_room:
                        m["boss_entry_hp_sum"] += ex["hp"] / max(1, ex["mhp"])
                        m["boss_entry_n"] += 1
                if prev_ss == "BATTLE" and battle_room is not None:
                    loss = max(0, battle_entry_hp - ex["hp"])
                    if "ELITE" in battle_room:
                        m["hp_lost_elite"] += loss; m["n_elite"] += 1
                    elif "BOSS" in battle_room:
                        m["hp_lost_boss"] += loss; m["n_boss"] += 1
                    elif "MONSTER" in battle_room:
                        m["hp_lost_monster"] += loss; m["n_monster"] += 1
                    b = _bucket(battle_act, battle_room)
                    b[0] += 1; b[2] += loss; b[3] += battle_steps
                    battle_room = None
                if cur_ss == "REST_ROOM":
                    m["campfire_hp_sum"] += ex["hp"] / max(1, ex["mhp"])
                    m["campfire_n"] += 1
                prev_ss = cur_ss
                entry = ex
            if act2_roles is None and int(self.env.gc.act) >= 2:
                act2_roles = self._deck_roles()
            if self.env.in_combat:
                battle_steps += 1
                cs = self._csnap()
                if cs is not None:
                    last_combat = cs
                a = self._search_action()
                if a is None:
                    break
            elif macro:
                a = self._macro_action()
            else:
                a = self.vare.act(self._obs1200(), self._mask())
            _, r, term, trunc, _ = self.env.step(int(a))
            self.history.append(int(a))
            if term or trunc:
                break
        attribute(prev_ss, entry, self._snap())
        gc = self.env.gc
        won = gc.outcome == _VIC
        out = {"seed": seed, "mode": self.mode, "win": int(won),
               "floor": int(gc.floor_num), "act": int(gc.act),
               "max_hp": int(gc.max_hp), "gold": int(gc.gold),
               "relics": len(gc.relics), "elites_avoided": self._avoided}
        out.update(m)
        out["final_deck"] = self._deck_comp()
        if not won and last_combat is not None:
            frac = (last_combat["ehp"] / last_combat["emax"]
                    if last_combat["emax"] > 0 else 1.0)
            db = _bucket(last_combat["act"], last_combat["room"])
            db[1] += 1; db[4] += frac
            out.update({"death_room": last_combat["room"],
                        "death_act": last_combat["act"],
                        "enemy_hp_frac": round(frac, 3)})
        else:
            out.update({"death_room": "WON" if won else "?",
                        "death_act": int(gc.act), "enemy_hp_frac": 0.0})
        out["fb"] = fb
        out["act2_deck"] = act2_roles
        return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "search"
    combat_npz = os.environ.get("COMBAT_NPZ", "policy_varN.npz")  # mainline; varL=rollback
    ev = V2(combat_npz, "policy_varE.npz", mode)
    print(json.dumps(ev.play_v2(seed)), flush=True)
