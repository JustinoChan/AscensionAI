"""HONEST search: search may SAMPLE hidden information, never read it.

The deployed search (eval_fullrun_search.FullRunEval._rollout_q and
eval_fullrun_analytics_v2.V2._rollout_run) rebuilds every rollout by replaying
the TRUE seed, so a rollout knows the real draw order, enemy move rolls and
every future reward / encounter. This mixin replaces that with:

  * an exact C++ copy of the live state (GameContext.copy / BattleContext.copy)
    wherever the old code only needed to *reconstruct the present*
    (_restore), and
  * K fresh copies, each run through determinize(s) (all hidden RNGs, the
    unseen monster/elite queues, relic-pool order and the draw pile resampled;
    visible state unchanged), wherever the old code *looked into the future*
    (combat 1-ply rollouts, draft-teacher run rollouts).

Toggles (all default off; HONEST unset => this module is never imported):
  HONEST=1           enable
  HONEST_K=<n>       determinized rollouts per combat candidate (default 4)
  HONEST_DRAFT_K=<n> determinized run rollouts per draft candidate (default HONEST_K)
  HONEST_TAIL_K=<n>  combat-search K inside draft-teacher run rollouts (default 1)
  HONEST_MACRO=1     also label macro/smith 1-step lookaheads on a determinized
                     copy (the deployed macro peeks one step ahead, e.g. what an
                     unknown map node turns into). Off by default so the arms
                     isolate the combat search.
Common random numbers: within one decision every candidate is scored on the
same K sampled worlds (variance reduction). Sample seeds come from
(run seed, decision counter, k) and are expanded through splitmix in C++; they
share no state with the game's own RNG streams.
"""
import os
import slaythespire

HONEST_K = int(os.environ.get("HONEST_K", "4"))
HONEST_DRAFT_K = int(os.environ.get("HONEST_DRAFT_K", str(HONEST_K)))
HONEST_TAIL_K = int(os.environ.get("HONEST_TAIL_K", "1"))
HONEST_MACRO = os.environ.get("HONEST_MACRO", "0") == "1"
HONEST_CPP = os.environ.get("HONEST_CPP", "0") == "1"   # native batched rollouts (identical decisions, ~30x faster)
# combat engine: "1ply" (default; K-sample 1-ply rollout search) or "mcts" (ensemble of
# K BattleScumSearcher2 trees, each on its own determinized root, MCTS_SIMS sims per tree)
HONEST_ENGINE = os.environ.get("HONEST_ENGINE", "1ply")
MCTS_SIMS = int(os.environ.get("MCTS_SIMS", "200"))
# per-act engine, e.g. HONEST_ENGINE_BY_ACT=1ply,mcts,mcts (overrides HONEST_ENGINE)
HONEST_ENGINE_BY_ACT = [x.strip() for x in os.environ.get("HONEST_ENGINE_BY_ACT", "").split(",") if x.strip()]
# K for the MCTS engine when hybrid (so 1-ply K and MCTS trees can differ)
MCTS_TREES = int(os.environ.get("MCTS_TREES", "0"))
# adaptive budgets by room type (Phase 2 step 4); unset -> same as hallway
def _by_room(prefix, cast=int):
    d = {}
    for room, key in (("ELITE", "ELITE"), ("BOSS", "BOSS"), ("MONSTER", "MONSTER")):
        v = os.environ.get("%s_%s" % (prefix, key))
        if v:
            d[room] = cast(v)
    return d
HONEST_K_BY_ROOM = _by_room("HONEST_K")
POTION_SAVE = os.environ.get("POTION_SAVE", "0") == "1"
POTION_SAVE_HP = float(os.environ.get("POTION_SAVE_HP", "0.25"))
MCTS_SIMS_BY_ROOM = _by_room("MCTS_SIMS")

_UND = slaythespire.GameOutcome.UNDECIDED
_VIC = slaythespire.GameOutcome.PLAYER_VICTORY
_M64 = (1 << 64) - 1


def make_honest(base, GAMMA, TOPK, ROLL_CAP):
    class Honest(base):
        # ---------------- state snapshots (exact C++ copies) ----------------
        def _hsnap(self):
            e = self.env
            return (e.gc.copy(),
                    e.bc.copy() if (e.in_combat and e.bc is not None) else None,
                    bool(e.in_combat), e.last_hp, e.last_floor,
                    getattr(e, "_last_enemy_hp", 0), getattr(e, "_last", None))

        def _hload(self, snap, det=None):
            gc, bc, inc, lhp, lfl, leh, last = snap
            e = self.env
            e.gc = gc.copy()
            e.bc = bc.copy() if bc is not None else None
            e.in_combat = inc and e.bc is not None
            if det is not None:
                e.gc.determinize(det & _M64)
                if e.bc is not None:
                    e.bc.determinize((det * 0x9E3779B97F4A7C15 + 1) & _M64)
            e.last_hp, e.last_floor, e._last_enemy_hp, e._last = lhp, lfl, leh, last
            e._get_info()                      # refresh available_actions
            return True

        def _hdet(self, k):
            self._hctr = getattr(self, "_hctr", 0) + 1
            return ((int(self.seed) * 0x100000001B3) ^ (self._hctr * 0x9E3779B1)
                    ^ (k * 0x85EBCA77) ^ 0xC0FFEE) & _M64

        # _restore() = "rebuild the present". Honest: exact copy of the state at
        # entry of the innermost decision method (never a replay of the seed).
        def _restore(self):
            st = self.__dict__.get("_hstack")
            if not st:
                return super()._restore()
            snap, det = st[-1]
            return self._hload(snap, det)

        def _hwrapped(self, name, use_det, *a, **kw):
            st = self.__dict__.setdefault("_hstack", [])
            det = self._hdet(0) if (use_det and HONEST_MACRO) else None
            st.append((self._hsnap(), det))
            try:
                out = getattr(base, name)(self, *a, **kw)
            finally:
                snap, _ = st.pop()
            if det is not None:      # leave the live (un-determinized) state in place
                self._hload(snap)
            return out

        # live progress heartbeat for the control panel (HONEST_HB=<out dir>)
        def _hb(self):
            d = os.environ.get("HONEST_HB")
            if not d or getattr(self, "_in_rollout", False) or self.__dict__.get("_hstack"):
                return
            fl = int(self.env.gc.floor_num)
            if fl != getattr(self, "_hb_floor", -1):
                self._hb_floor = fl
                try:
                    p = os.path.join(d, "seed_%d.prog" % int(self.seed))
                    with open(p + ".t", "w") as f:
                        f.write("%d %d\n" % (fl, int(__import__("time").time())))
                    os.replace(p + ".t", p)
                except OSError:
                    pass

        def _macro_action(self):
            self._hb()
            return self._hwrapped("_macro_action", True)

        def _smith_best(self):
            return self._hwrapped("_smith_best", True)

        def _cnet(self):
            c = self.__dict__.get("_cnet_obj")
            if c is None:
                n = self.net
                c = slaythespire.CombatNet([w for w, _ in n.pbody] + [n.pi[0]],
                                           [b for _, b in n.pbody] + [n.pi[1]],
                                           self.maxima.astype("float32"))
                self._cnet_obj = c
            return c

        # ---------------- honest 1-ply combat search ----------------
        def _hrollout_q(self, action):
            e = self.env
            e.last_hp = e.bc.player.cur_hp
            e._last_enemy_hp = e._enemy_hp()
            G, disc = 0.0, 1.0
            _, r, _, _, _ = e.step(int(action))
            done, tb = self._term()
            G += disc * (r + tb)
            disc *= GAMMA
            steps = 0
            while not done and steps < ROLL_CAP:
                a = self.net.greedy(self._obs1235(), self._mask())
                _, r, _, _, _ = e.step(a)
                done, tb = self._term()
                G += disc * (r + tb)
                disc *= GAMMA
                steps += 1
            return G

        def _room(self):
            return str(self.env.gc.cur_room).split(".")[-1]

        # POTION_SAVE=1: keep potions for elites/bosses. In hallway/event fights the
        # root may not drink a potion unless HP <= POTION_SAVE_HP*max or the visible
        # incoming damage is lethal through current block.
        def _potion_blocked(self):
            if not POTION_SAVE or self._room() in ("ELITE", "BOSS"):
                return False
            bc = self.env.bc
            hp, mx = bc.player.cur_hp, max(1, self.env.gc.max_hp)
            if hp <= POTION_SAVE_HP * mx:
                return False
            try:
                from combat_features import combat_block
                incoming = float(combat_block(self.env)[33]) * 40.0
                if incoming >= 79.9 or incoming >= hp + bc.player.block:
                    return False
            except Exception:
                pass
            return True

        def _is_potion(self, i, acts=None):
            acts = acts if acts is not None else self.env.bc.get_possible_actions()
            return i < len(acts) and ((acts[i].bits >> 29) & 0xF) == 1

        def _mcts_action(self, K, sims):
            e = self.env
            dets = [self._hdet(k) for k in range(K)]
            stats = slaythespire.mcts_root(e.bc, dets, int(sims), False)
            acts = e.bc.get_possible_actions()
            idx = {a.bits: i for i, a in enumerate(acts)}
            block_pot = self._potion_blocked()
            best, bkey = 0, None
            for bits, visits, vsum in stats:
                if bits not in idx:
                    continue
                if block_pot and ((bits >> 29) & 0xF) == 1:
                    continue
                key = (visits, vsum / max(1, visits))
                if bkey is None or key > bkey:
                    bkey, best = key, idx[bits]
            return int(best)

        def _search_action(self):
            self._hb()
            room = self._room()
            if getattr(self, "_in_rollout", False):
                K = HONEST_TAIL_K
            else:
                K = HONEST_K_BY_ROOM.get(room, HONEST_K)
            engine = HONEST_ENGINE
            if HONEST_ENGINE_BY_ACT:
                act = int(self.env.gc.act)
                engine = HONEST_ENGINE_BY_ACT[min(act, len(HONEST_ENGINE_BY_ACT)) - 1]
            if engine == "mcts" and not getattr(self, "_in_rollout", False):
                return self._mcts_action(MCTS_TREES or K, MCTS_SIMS_BY_ROOM.get(room, MCTS_SIMS))
            mask = self._mask()
            legal = [i for i in range(128) if mask[i]]
            if POTION_SAVE and not getattr(self, "_in_rollout", False) and self._potion_blocked():
                acts = self.env.bc.get_possible_actions()
                keep = [i for i in legal if not self._is_potion(i, acts)]
                if keep:
                    legal = keep
            if len(legal) > TOPK:
                prior = self.net.logits(self._obs1235(), mask)
                legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:TOPK]
            dets = [self._hdet(k) for k in range(K)]
            if HONEST_CPP:
                # native batched rollouts: same worlds, same policy, same score
                e = self.env
                last = getattr(e, "_last", None)
                scores, _ = slaythespire.rollout_qb(e.gc, e.bc, legal, dets, self._cnet(), 0,
                                                    GAMMA, ROLL_CAP, False,
                                                    [] if last is None else [int(v) for v in last])
                best, bq = legal[0], -1e18
                for a, row in zip(legal, scores):
                    tot = 0.0
                    for v in row:
                        tot += v
                    qa = tot / K
                    if qa > bq:
                        bq, best = qa, a
                return int(best)
            live = self._hsnap()
            best, bq = legal[0], -1e18
            for a in legal:
                tot = 0.0
                for d in dets:
                    self._hload(live, d)
                    tot += self._hrollout_q(a)
                qa = tot / K
                if qa > bq:
                    bq, best = qa, a
            self._hload(live)
            return int(best)

        # ---------------- honest draft teacher (DRAFT_SEARCH=1) ----------------
        def _hrollout_run(self, live, first_a, det, cap=4000):
            self._hload(live, det)
            self.env.step(int(first_a))
            steps = 0
            while steps < cap and self.env.gc.outcome == _UND:
                if self.env.in_combat:
                    a = self._search_action()
                else:
                    a = self._macro_action()
                if a is None:
                    break
                self.env.step(int(a))
                steps += 1
            gc = self.env.gc
            return (int(gc.floor_num), int(gc.outcome == _VIC))

        def _draft_search(self, ss, labels):
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
            live = self._hsnap()
            hist = list(self.history)
            srf = self._shop_removed_floor
            dets = [self._hdet(k) for k in range(HONEST_DRAFT_K)]
            stack = self.__dict__.get("_hstack", [])
            saved_stack = list(stack)
            self._in_rollout = True
            best_i, best_v, recs = cands[0][0], -1e18, []
            for ci, card in cands:
                v = 0.0
                for d in dets:
                    self._shop_removed_floor = srf
                    floor, win = self._hrollout_run(live, ci, d)
                    v += floor + 1000.0 * win
                v /= len(dets)
                feat = self._candidate_features(base_roles, scalars, card)
                recs.append({"feat": [float(x) for x in feat], "value": v,
                             "roles": self._post_roles(base_roles, card),
                             "scalars": scalars, "card": self._card_info(card)})
                if v > best_v:
                    best_v, best_i = v, ci
            self._in_rollout = False
            stack[:] = saved_stack
            self.history = hist
            self._shop_removed_floor = srf
            self._hload(live)
            if os.environ.get("DRAFT_LOG", "0") == "1":
                self._draft_records.extend(recs)
            return best_i

    # ---------------- passive potion audit (POTION_AUDIT=1) ----------------
    # Observes LIVE steps only (never rollouts); decisions are unchanged.
    class Audited(Honest):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._pa_busy = 0
            self.potion_audit = []
            self._pa_fight = None
            env = self.env
            orig = env.step

            def step(idx):
                if self._pa_busy:
                    return orig(idx)
                pre_in = bool(env.in_combat)
                pre_pots = self._pa_pots()
                is_pot = False
                if pre_in:
                    try:
                        acts = env.bc.get_possible_actions()
                        a = acts[idx] if idx < len(acts) else acts[0]
                        is_pot = ((a.bits >> 29) & 0xF) == 1
                    except Exception:
                        pass
                out = orig(idx)
                post_in = bool(env.in_combat)
                if not pre_in and post_in:
                    room = str(env.gc.cur_room).split(".")[-1]
                    self._pa_fight = {"floor": int(env.gc.floor_num), "act": int(env.gc.act),
                                      "room": room, "enc": str(env.gc.encounter).split(".")[-1],
                                      "pots_start": self._pa_pots(), "used": 0,
                                      "cap": int(getattr(env.gc, "potion_capacity", 0) or 0)}
                elif pre_in and self._pa_fight is not None:
                    if is_pot:
                        self._pa_fight["used"] += 1
                    if not post_in or env.gc.outcome != _UND:
                        f = self._pa_fight
                        f["died"] = env.gc.outcome == slaythespire.GameOutcome.PLAYER_LOSS
                        f["pots_end"] = pre_pots - (1 if is_pot else 0) if f["died"] else self._pa_pots()
                        self.potion_audit.append(f)
                        self._pa_fight = None
                return out
            env.step = step

        def _pa_pots(self):
            try:
                o = self.env._get_observation()
                return int(sum(1 for v in o[417:422] if v >= 2))
            except Exception:
                return -1

        def _search_action(self):
            self._pa_busy += 1
            try:
                return super()._search_action()
            finally:
                self._pa_busy -= 1

        def _macro_action(self):
            self._pa_busy += 1
            try:
                return super()._macro_action()
            finally:
                self._pa_busy -= 1

    if os.environ.get("POTION_AUDIT", "0") == "1":
        Audited.__name__ = "HonestAudited" + base.__name__
        return Audited
    Honest.__name__ = "Honest" + base.__name__
    return Honest
