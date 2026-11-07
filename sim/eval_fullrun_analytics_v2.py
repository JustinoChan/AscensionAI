"""v2 decision-log + death stats for ONE seed, with an optional NARROW macro
override (mode=macro). Combat = search. Non-combat = baseline varE, except in
macro mode where card-reward take/skip, card removal, and shop card-buys are
overridden by simple heuristics (everything else still defers to varE).

Decisions are attributed by macro-state DELTAS across screen transitions; deck
state is read from the obs deck region (obs[19:239]; Strike=151, Defend=57),
NOT gc.deck (segfault-safe). One process per seed.

argv: [1] seed  [2] mode = search | macro  (default search)"""
import sys, json, warnings, os, random
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import slaythespire
from eval_fullrun_search import FullRunEval, MAX_GAME_STEPS, GAMMA, TOPK
from card_roles import classify, FRONTLOAD
from run_state_features import vectorize, vectorize2

_VIC = slaythespire.GameOutcome.PLAYER_VICTORY
_UND = slaythespire.GameOutcome.UNDECIDED
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
SMITH = os.environ.get("SMITH", "1") == "1"              # conservative boss-aware smith (deployed)
SMITH_TARGET = os.environ.get("SMITH_TARGET", "1") == "1"  # tempo-prioritized upgrade target (deployed mainline_v3; =0 reproduces v2)
# Oracle ceiling sweep (diagnostic). Default none = byte-identical mainline_v3.
ORACLE = os.environ.get("ORACLE", "none")        # none|combat|path|deck|combined
P2_TOPK = int(os.environ.get("P2_TOPK", "3"))    # 2nd-ply width for the combat oracle
DRAFT_SEARCH = os.environ.get("DRAFT_SEARCH", "0") == "1"  # Stage A rollout teacher
DRAFT_LOG = os.environ.get("DRAFT_LOG", "0") == "1"        # emit (feat,value) records
V_DRAFT = os.environ.get("V_DRAFT", "0") == "1"            # Stage B: deploy learned V
QLAND = os.environ.get("QLAND", "0") == "1"                # combat Q-landscape log
QLAND_DIR = os.environ.get("QLAND_DIR", "sim_logs/qland")
COMBAT_NET = os.environ.get("COMBAT_NET", "")              # student Q-net replaces live search


def _oracle(name):
    # combined stacks the two POSITIVE levers (deck + path); combat 2-ply did not
    # help wins and is slow, so it is excluded from combined.
    return ORACLE == name or (ORACLE == "combined" and name in ("path", "deck"))


# Strong Ironclad shell for the deck oracle (upper bound on deck quality).
PREMIUM_SHELL = ["POMMEL_STRIKE", "TWIN_STRIKE", "SHRUG_IT_OFF", "SHRUG_IT_OFF",
                 "INFLAME", "FEEL_NO_PAIN", "WHIRLWIND", "UPPERCUT",
                 "DISARM", "BATTLE_TRANCE", "IMPERVIOUS", "DEMON_FORM"]


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
        c = dict(size=len(deck), starters=0, upgraded=0, upg_tempo=0, damage=0,
                 block=0, aoe=0, scaling=0, draw_energy=0, powers=0, junk=0,
                 qsum=0.0)
        for card in deck:
            name = str(card.id).split(".")[-1]
            ctype = str(card.type).split(".")[-1]
            roles = classify(name, ctype)
            if card.upgraded:
                c["upgraded"] += 1
                if (name not in ("STRIKE_RED", "DEFEND_RED")
                        and (ctype == "ATTACK" or "block" in roles
                             or "aoe" in roles)):
                    c["upg_tempo"] += 1
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

    # ---- deck oracle: inject a premium upgraded shell at Act-2 entry ----
    def _inject_premium_deck(self):
        gc = self.env.gc
        try:
            deck = list(gc.deck)                     # single read (segfault-safe)
            basics = [i for i, c in enumerate(deck)
                      if c.is_starter_strike_or_defend]
            for i in sorted(basics, reverse=True):   # descending keeps indices valid
                gc.remove_card(i)
            for nm in PREMIUM_SHELL:
                cid = getattr(slaythespire.CardId, nm, None)
                if cid is None:
                    continue
                card = slaythespire.Card(cid)
                try:
                    card.upgrade()
                except Exception:
                    pass
                gc.obtain_card(card)
        except Exception:
            pass

    # The search reconstructs every state via reset+replay, which would wipe the
    # injected deck; re-inject at the same point (first Act-2 step) during replay
    # so the reconstructed trajectory matches the live one.
    def _restore(self):
        if not _oracle("deck"):
            return super()._restore()
        try:
            self.env.reset(seed=int(self.seed))
        except Exception:
            return False
        injected = False
        for a in self.history:
            if not injected and int(self.env.gc.act) >= 2:
                self._inject_premium_deck()
                injected = True
            if not self._ff_step(int(a)) or self.env.gc.outcome != _UND:
                return False
        if not injected and int(self.env.gc.act) >= 2:
            self._inject_premium_deck()
        return True

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

    # ---- combat oracle: 2-ply search (ceiling probe; ORACLE=combat) ----
    def _search_action(self):
        if not _oracle("combat"):
            if COMBAT_NET and not self._in_rollout:
                return self._net_combat_action()
            if QLAND and not self._in_rollout:
                return self._qland_search_action()
            return super()._search_action()        # 1-ply (deployed mainline)
        mask = self._mask()
        legal = [i for i in range(128) if mask[i]]
        if len(legal) > TOPK:
            prior = self.net.logits(self._obs1235(), mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
        saved = list(self.history)
        best, bq = legal[0], -1e18
        for a in legal:
            if not self._restore():
                self.history = saved
                continue
            self.env.last_hp = self.env.bc.player.cur_hp
            self.env._last_enemy_hp = self.env._enemy_hp()
            _, r, _, _, _ = self.env.step(int(a))
            done, tb = self._term()
            if done:
                qa = r + tb
            else:
                mask2 = self._mask()
                legal2 = [i for i in range(128) if mask2[i]]
                if len(legal2) > P2_TOPK:
                    prior2 = self.net.logits(self._obs1235(), mask2)
                    legal2 = sorted(legal2, key=lambda x: prior2[x],
                                    reverse=True)[:P2_TOPK]
                self.history = saved + [int(a)]
                bestb = -1e18
                for b in legal2:
                    qb = self._rollout_q(b)
                    if qb is not None and qb > bestb:
                        bestb = qb
                self.history = saved
                qa = (r + GAMMA * bestb) if bestb > -1e17 else (r + tb)
            if qa > bq:
                bq, best = qa, a
        self.history = saved
        if not self._restore():
            return None
        return int(best)

    # ---- combat Q-landscape harvest (QLAND) + student probe (COMBAT_NET) ----
    def _qland_search_action(self):
        """Base 1-ply search replicated verbatim + one (obs, candidates, Q)
        record per live decision. MUST stay decision-identical to
        FullRunEval._search_action (gate: identical run JSON)."""
        mask = self._mask()
        legal = [i for i in range(128) if mask[i]]
        obs = self._obs1235()
        if len(legal) > TOPK:
            prior = self.net.logits(obs, mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
        best, bq, qs = legal[0], -1e18, []
        for a in legal:
            qa = self._rollout_q(a)
            qs.append(float("nan") if qa is None else float(qa))
            if qa is not None and qa > bq:
                bq, best = qa, a
        if not self._restore():
            return None
        if any(q == q for q in qs):                    # >=1 non-NaN
            self._qland_records.append(
                (obs.astype(np.float16), list(legal), qs,
                 int(self.env.gc.floor_num)))
        return best

    def _net_combat_action(self):
        """Probe: student Q-net re-ranks the same TOPK-by-prior candidates the
        teacher searches. Zero rollouts. Restricting argmax to those candidates
        keeps the student on its training distribution."""
        if getattr(self, "qnet", None) is None:
            from np_qnet import NpQNet
            self.qnet = NpQNet(COMBAT_NET)
        mask = self._mask()
        legal = [i for i in range(128) if mask[i]]
        obs = self._obs1235()
        if len(legal) > TOPK:
            prior = self.net.logits(obs, mask)
            legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
        q = self.qnet.q(obs)
        return int(max(legal, key=lambda a: float(q[a])))

    def _qland_save(self):
        if not self._qland_records:
            return
        n, K = len(self._qland_records), TOPK
        obs = np.stack([r[0] for r in self._qland_records])
        cands = np.full((n, K), -1, np.int16)
        qs = np.full((n, K), np.nan, np.float32)
        for i, (_, c, q, _) in enumerate(self._qland_records):
            cands[i, :len(c)] = c
            qs[i, :len(q)] = q
        floor = np.asarray([r[3] for r in self._qland_records], np.int16)
        os.makedirs(QLAND_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(QLAND_DIR, "q_%d.npz" % self.seed),
                            obs=obs, cands=cands, qs=qs, floor=floor)

    # ---- macro override ----
    def _upg(self):
        r = np.asarray(self.env._get_observation(), np.float32)
        return float(r[DECK_LO + 1:DECK_HI:2].sum())

    # ---- smith target scoring (upgrade-efficiency lever) ----
    def _deck_snap(self):
        try:
            return [(str(c.id).split(".")[-1], str(c.type).split(".")[-1],
                     bool(c.upgraded)) for c in self.env.gc.deck]
        except Exception:
            return None

    def _smith_score(self, name, ctype):
        """Act-2 tempo value of upgrading this card (higher = better): premium
        attack > other attack > strong block > AoE > draw/energy > scaling/power
        > starter/junk filler."""
        if name in ("STRIKE_RED", "DEFEND_RED"):
            return 0
        roles = classify(name, ctype)
        if "junk" in roles:
            return 0
        if name in FRONTLOAD:
            return 6
        if ctype == "ATTACK":
            return 5
        if "block" in roles:
            return 4
        if "aoe" in roles:
            return 3
        if ("draw" in roles) or ("energy" in roles):
            return 2
        if "scaling" in roles or ctype == "POWER":
            return 1
        return 0

    def _smith_best(self):
        """At a smith upgrade screen, pick the action that upgrades the
        highest-tempo card. Identifies each action's card by diffing gc.deck
        across a scratch-step; falls back to action 0 on any read failure."""
        gc = self.env.gc
        n = len(gc.get_possible_actions())
        before = self._deck_snap()
        if not before or n == 0:
            return 0
        bmult = {}
        for nm, tp, up in before:
            if not up:
                bmult[(nm, tp)] = bmult.get((nm, tp), 0) + 1
        best_i, best_s = None, -1
        for i in range(min(n, 30)):
            if not self._restore():
                break
            try:
                self._ff_step(i)
            except Exception:
                continue
            after = self._deck_snap()
            if not after:
                continue
            amult = {}
            for nm, tp, up in after:
                if not up:
                    amult[(nm, tp)] = amult.get((nm, tp), 0) + 1
            card = next((k for k in bmult if amult.get(k, 0) < bmult[k]), None)
            if card is None:
                continue
            s = self._smith_score(card[0], card[1])
            if s > best_s:
                best_s, best_i = s, i
                if best_s >= 6:           # premium attack; nothing scores higher
                    break
        self._restore()
        return best_i if best_i is not None else 0

    def _offered(self):
        try:
            return list(self.env.gc.get_card_reward())
        except Exception:
            return []

    # ---- drafting teacher: 1-ply rollout over offered cards (DRAFT_SEARCH) ----
    def _rollout_run(self, hist, first_a, cap=4000):
        """From the live decision point, apply first_a then follow the FIXED base
        policy (search combat + mainline_v3 macro) to run-end. Returns
        (floor, win). self._in_rollout guards nested draft decisions so the tail
        uses the base heuristic, not the teacher."""
        self.history = list(hist)
        if not self._restore():
            return (0, 0)
        self.env.step(int(first_a))
        self.history.append(int(first_a))
        steps = 0
        while steps < cap and self.env.gc.outcome == _UND:
            if self.env.in_combat:
                a = self._search_action()
            else:
                a = self._macro_action()
            if a is None:
                break
            self.env.step(int(a))
            self.history.append(int(a))
            steps += 1
        gc = self.env.gc
        return (int(gc.floor_num), int(gc.outcome == _VIC))

    def _post_roles(self, base_roles, offered_card):
        """Post-pick deck roles by composition (no extra gc.deck reads). Take =
        base roles + the offered card's roles; skip (None) = base unchanged."""
        roles = dict(base_roles)
        if offered_card is not None:
            nm = str(offered_card.id).split(".")[-1]
            tp = str(offered_card.type).split(".")[-1]
            rs = classify(nm, tp)
            roles["size"] = roles.get("size", 0) + 1
            if tp == "ATTACK":
                roles["damage"] = roles.get("damage", 0) + 1
            if "block" in rs:
                roles["block"] = roles.get("block", 0) + 1
            if "aoe" in rs:
                roles["aoe"] = roles.get("aoe", 0) + 1
            if "scaling" in rs:
                roles["scaling"] = roles.get("scaling", 0) + 1
            if ("draw" in rs) or ("energy" in rs):
                roles["draw_energy"] = roles.get("draw_energy", 0) + 1
            if tp == "POWER":
                roles["powers"] = roles.get("powers", 0) + 1
            if "junk" in rs:
                roles["junk"] = roles.get("junk", 0) + 1
            if "starter" in rs:
                roles["starters"] = roles.get("starters", 0) + 1
        return roles

    def _candidate_features(self, base_roles, scalars, offered_card):
        """Post-pick run-state features (16-d), train/deploy matched."""
        return vectorize(self._post_roles(base_roles, offered_card), scalars)

    def _card_info(self, c):
        """Raw offered-card context: logged as-is so any future feature set
        can be built offline without re-running the teacher."""
        if c is None:
            return None
        nm = str(c.id).split(".")[-1]
        tp = str(c.type).split(".")[-1]
        return {"name": nm, "type": tp,
                "rarity": str(c.rarity).split(".")[-1],
                "qual": self._qual(c), "frontload": nm in FRONTLOAD}

    def _candidate_features2(self, base_roles, scalars, offered_card):
        """Enriched features (25-d): run-state + offered-card block."""
        return vectorize2(self._post_roles(base_roles, offered_card), scalars,
                          self._card_info(offered_card))

    def _draft_search(self, ss, labels):
        """Replace the card-reward decision: roll each candidate (take_k / skip)
        to run-end, pick argmax(floor + 1000*win). Logs (post-pick feat, value)
        per candidate when DRAFT_LOG."""
        takes = sorted([L for L in labels if L["dd"] >= 1], key=lambda x: x["i"])
        noops = [L for L in labels if L["dd"] == 0 and L["rd"] == 0]
        leave = [L for L in noops if L["to"] == "MAP_SCREEN"]
        skip = leave or noops
        if not takes and not skip:
            return None
        offered = self._offered()
        base_roles = self._deck_roles() or {}
        gc = self.env.gc
        scalars = {"act": int(gc.act), "floor": int(gc.floor_num),
                   "hp_frac": gc.cur_hp / max(1, gc.max_hp),
                   "gold": int(gc.gold), "relics": len(gc.relics)}
        cands = []                                  # (action_index, offered_card_or_None)
        for k, L in enumerate(takes):
            cands.append((L["i"], offered[k] if k < len(offered) else None))
        if skip:
            cands.append((skip[0]["i"], None))
        live = list(self.history)
        srf = self._shop_removed_floor
        self._in_rollout = True
        best_i, best_v, recs = cands[0][0], -1e18, []
        for ci, card in cands:
            floor, win = self._rollout_run(live, ci)
            v = floor + 1000.0 * win
            feat = self._candidate_features(base_roles, scalars, card)
            recs.append({"feat": [float(x) for x in feat], "value": v,
                         "roles": self._post_roles(base_roles, card),
                         "scalars": scalars, "card": self._card_info(card)})
            if v > best_v:
                best_v, best_i = v, ci
        self._in_rollout = False
        self.history = live
        self._shop_removed_floor = srf
        self._restore()
        if DRAFT_LOG:
            self._draft_records.extend(recs)
        return best_i

    def _draft_v_choose(self, ss, labels):
        """Deploy: pick argmax_candidate V(post-pick run-state). No rollouts."""
        if getattr(self, "draftv", None) is None:
            from np_draft_v import NpDraftV
            self.draftv = NpDraftV(os.environ.get("DRAFT_V_NPZ", "draft_v.npz"))
        takes = sorted([L for L in labels if L["dd"] >= 1], key=lambda x: x["i"])
        noops = [L for L in labels if L["dd"] == 0 and L["rd"] == 0]
        leave = [L for L in noops if L["to"] == "MAP_SCREEN"]
        skip = leave or noops
        if not takes and not skip:
            return None
        offered = self._offered()
        base_roles = self._deck_roles() or {}
        gc = self.env.gc
        scalars = {"act": int(gc.act), "floor": int(gc.floor_num),
                   "hp_frac": gc.cur_hp / max(1, gc.max_hp),
                   "gold": int(gc.gold), "relics": len(gc.relics)}
        cands = []
        for k, L in enumerate(takes):
            cands.append((L["i"], offered[k] if k < len(offered) else None))
        if skip:
            cands.append((skip[0]["i"], None))
        best_i, best_v = cands[0][0], -1e18
        enriched = self.draftv.W[0].shape[0] == 25   # npz input dim picks the feature set
        for ci, card in cands:
            x = (self._candidate_features2(base_roles, scalars, card) if enriched
                 else self._candidate_features(base_roles, scalars, card))
            v = self.draftv.value(x)
            if v > best_v:
                best_v, best_i = v, ci
        return best_i

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
        if (DRAFT_SEARCH and not self._in_rollout
                and ss in ("REWARDS", "CARD_SELECT")
                and any(L["dd"] >= 1 for L in labels)):
            r = self._draft_search(ss, labels)
            if r is not None:
                return r
        if (V_DRAFT and not self._in_rollout
                and ss in ("REWARDS", "CARD_SELECT")
                and any(L["dd"] >= 1 for L in labels)):
            r = self._draft_v_choose(ss, labels)
            if r is not None:
                return r
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
                return self._smith_best() if SMITH_TARGET else upg[0]["i"]
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

    def _oracle_path(self, labels):
        """Path/rest oracle (ORACLE=path): variant 0 = baseline route; variants
        >=1 sample an HP-aware weighted route. Best-of-K over variants upper-
        bounds what better routing/resting could buy."""
        pv = int(os.environ.get("PATH_VARIANT", "0"))
        if pv == 0 or not labels:
            return self.vare.act(self._obs1200(), self._mask())
        gc = self.env.gc
        hp = gc.cur_hp / max(1, gc.max_hp)
        rng = random.Random(
            (int(self.seed) * 7349 + pv * 911 + int(gc.floor_num)) & 0x7fffffff)

        def w(L):
            rm = L.get("room", "")
            if "REST" in rm:
                return 6.0 if hp < 0.7 else 2.0
            if "SHOP" in rm:
                return 3.0
            if "TREASURE" in rm:
                return 3.0
            if "EVENT" in rm:
                return 2.0
            if "ELITE" in rm:
                return 2.5 if hp > 0.8 else 0.3
            return 1.0                      # MONSTER / unknown
        ws = [w(L) for L in labels]
        x = rng.random() * (sum(ws) or 1.0)
        for L, wi in zip(labels, ws):
            x -= wi
            if x <= 0:
                return L["i"]
        return labels[0]["i"]

    def _choose_path(self, labels, deck, strike, defend, act):
        """Risk-priced elite/path filter (NOT an elite ban). Each map option is
        labeled by the room it leads to (scratch-step). Only when an elite is
        unaffordable AND a safer node exists do we reroute; otherwise defer to
        the baseline policy so relic-bearing elites are still taken when the run
        can pay for them."""
        if _oracle("path"):
            return self._oracle_path(labels)
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
        self._in_rollout = False
        self._draft_records = []
        self._qland_records = []
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
                if _oracle("deck"):
                    self._inject_premium_deck()
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
        if DRAFT_LOG:
            out["draft_records"] = self._draft_records
        if QLAND:
            self._qland_save()
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
    if os.environ.get("HONEST", "0") == "1":
        # search may SAMPLE hidden info, never read it (see honest.py)
        import time as _t
        from honest import make_honest
        from eval_fullrun_search import GAMMA as _G, TOPK as _TK, ROLL_CAP as _RC
        ev = make_honest(V2, _G, _TK, _RC)(combat_npz, "policy_varE.npz", mode)
        if os.environ.get("TRACE", "0") == "1":
            import run_trace as _trace_mod
            _trace_mod.install(ev)
        _t0 = _t.time()
        out = ev.play_v2(seed)
        if os.environ.get("TRACE", "0") == "1":
            out["trace"] = ev.trace
        out["secs"] = round(_t.time() - _t0, 2)
        if hasattr(ev, "potion_audit"):
            out["potion_audit"] = ev.potion_audit
        if hasattr(ev, "macro_log"):
            out["macro_log"] = ev.macro_log
        print(json.dumps(out), flush=True)
    else:
        ev = V2(combat_npz, "policy_varE.npz", mode)
        if os.environ.get("TRACE", "0") == "1":    # passive decision-level trace (run_trace.py)
            import run_trace as _trace_mod
            _trace_mod.install(ev)
            out = ev.play_v2(seed)
            out["trace"] = ev.trace
            print(json.dumps(out), flush=True)
        else:
            print(json.dumps(ev.play_v2(seed)), flush=True)
