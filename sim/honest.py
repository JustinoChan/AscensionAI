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
# HARVEST=<dir>: passive Phase-3 data harvest (root combat states + K-sample Q per
# candidate + fight outcomes, and run-level states at every non-combat decision),
# written to <dir>/seed_<seed>.npz. Decisions are unchanged.
HARVEST = os.environ.get("HARVEST", "")
# Phase 3 step 1c (all default off):
#   HONEST_PRIOR=<npz>  candidate prior (NpNet layout) used to rank/prune root candidates
#   HONEST_TOPK=<m>     keep the top-m root candidates (default: TOPK of the base search)
#   HONEST_VNET=<npz>   value net; with HONEST_DEPTH=<d> >= 0 every rollout is cut after d
#                       default-policy steps and scored  G += gamma^steps * VSCALE * V(s_leaf)
#   HONEST_VSCALE       value scale (default 50, as trained by train_h1.py)
# Phase 3 step 2 (default off): honest macro search with short run-level rollouts.
#   MACRO_CARD=1  card reward: take_i vs skip
#   MACRO_REST=1  campfire: rest vs smith
#   MACRO_K=<n> sampled worlds per candidate (default 6), MACRO_H=<floors> horizon
#   (default 6, never past the end of the current act's boss), MACRO_COMBAT_K=<k>
#   combat inside rollouts: 0 = varN greedy (C++ obs+net, default), k>0 = honest
#   1-ply C++ search with k samples. Score = floor reached + MACRO_HPW * hp/max_hp
#   (default 2) + 100 * win; death ends the rollout at its floor.
MACRO_CARD = os.environ.get("MACRO_CARD", "0") == "1"
MACRO_REST = os.environ.get("MACRO_REST", "0") == "1"
MACRO_K = int(os.environ.get("MACRO_K", "6"))
MACRO_H = int(os.environ.get("MACRO_H", "6"))
MACRO_COMBAT_K = int(os.environ.get("MACRO_COMBAT_K", "0"))
MACRO_HPW = float(os.environ.get("MACRO_HPW", "2"))
MACRO_LOG = os.environ.get("MACRO_LOG", "0") == "1"
# per-decision overrides (default: the shared MACRO_* values above)
#   MACRO_CARD_ACTS=1[,2,3]  only search card rewards in these acts (default: all)
#   MACRO_K_CARD / MACRO_K_REST, MACRO_H_CARD / MACRO_H_REST, MACRO_COMBAT_K_CARD / MACRO_COMBAT_K_REST
MACRO_CARD_ACTS = [int(x) for x in os.environ.get("MACRO_CARD_ACTS", "").split(",") if x.strip()]
MACRO_KIND = {kind: {"K": int(os.environ.get("MACRO_K_" + kind.upper(), MACRO_K)),
                     "H": int(os.environ.get("MACRO_H_" + kind.upper(), MACRO_H)),
                     "CK": int(os.environ.get("MACRO_COMBAT_K_" + kind.upper(), MACRO_COMBAT_K))}
              for kind in ("card", "rest")}
# Phase 3 step 3.2 (default off): learned out-of-combat policy (rl_macro_train.py).
#   MACRO_POLICY=<npz>      candidate scorer replaces the heuristic macro at every live out-of-combat
#                           decision with >= 2 legal options. Each legal option i is described by
#                           [obs1200(now), obs1200(after i) - obs1200(now), screen one-hot(16)], where
#                           "after i" is computed on a DETERMINIZED copy (unknown nodes / random event
#                           outcomes are sampled, never read). MLP 2416-256-256-1, softmax over options.
#   MACRO_FEAT_LOG=<dir>    (no policy) log the same candidate features + the heuristic's choice (BC data)
#   MACRO_POLICY_SAMPLE=1   sample from softmax (training); default argmax (evaluation)
#   MACRO_POLICY_LOG=<dir>  write <dir>/seed_<s>.npz with (obs, mask, action, logp) + run outcome
MACRO_POLICY = os.environ.get("MACRO_POLICY", "")
MACRO_POLICY_SAMPLE = os.environ.get("MACRO_POLICY_SAMPLE", "0") == "1"
MACRO_POLICY_LOG = os.environ.get("MACRO_POLICY_LOG", "")
MACRO_FEAT_LOG = os.environ.get("MACRO_FEAT_LOG", "")
# MACRO_MAPFEAT=1: append 28 visible-map features (mapfeat.py: min/max elites, rests, shops, '?', treasure,
# monsters on the paths to the boss, floors left, path count; for the position now and after each option)
MACRO_MAPFEAT = os.environ.get("MACRO_MAPFEAT", "0") == "1"
MACRO_MAX_CANDS = 30
# LLM_MACRO=1: local-Ollama out-of-combat teacher (llm_macro.py; default off, never imported when off)
LLM_MACRO = os.environ.get("LLM_MACRO", "0") == "1"
# Phase 3 step 3.5 (default off): HONEST_ROLLOUT_NET=<npz> replaces varN as the C++ rollout (default) policy
# inside honest 1-ply search; same layout as policy_varN.npz (pbody.0/2/4 + pi). Root pruning still uses varN.
HONEST_ROLLOUT_NET = os.environ.get("HONEST_ROLLOUT_NET", "")
HONEST_PRIOR = os.environ.get("HONEST_PRIOR", "")
HONEST_TOPK = int(os.environ.get("HONEST_TOPK", "0"))
HONEST_VNET = os.environ.get("HONEST_VNET", "")
HONEST_DEPTH = int(os.environ.get("HONEST_DEPTH", "-1"))
HONEST_VSCALE = float(os.environ.get("HONEST_VSCALE", "50"))
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
            if MACRO_POLICY and not getattr(self, "_in_rollout", False) and not self.__dict__.get("_hstack"):
                a = self._policy_action()
                if a is not None:
                    self._hb()
                    return a
            self._hb()
            top = bool(HARVEST) and not getattr(self, "_in_rollout", False) and not self.__dict__.get("_hstack")
            if top:
                self._hv_close()
                try:
                    o = self._obs1200()
                    mk = self._mask()
                except Exception:
                    o, mk = None, None
            fl = None
            if MACRO_FEAT_LOG and not MACRO_POLICY and not getattr(self, "_in_rollout", False) and not self.__dict__.get("_hstack"):
                mk2 = self._mask()
                if int(mk2.sum()) >= 2:
                    fl = self._cand_feats(mk2)
            a = self._hwrapped("_macro_action", True)
            if fl is not None and a is not None and int(a) < len(fl[0]):
                gc = self.env.gc
                self.__dict__.setdefault("_pol_log", []).append((fl[0].astype("float16"), int(a), 0.0, fl[1], int(gc.floor_num), int(gc.act)))
            if top and a is not None and o is not None:
                gc = self.env.gc
                self._hv()["m"].append((o, mk, str(gc.screen_state).split(".")[-1], int(a),
                                        int(gc.floor_num), int(gc.act), int(gc.cur_hp), int(gc.max_hp),
                                        int(gc.gold), len(gc.deck)))
            if LLM_MACRO and a is not None and not getattr(self, "_in_rollout", False) and not self.__dict__.get("_hstack"):
                mk3 = self._mask()
                if int(mk3.sum()) >= 2:
                    T = self.__dict__.get("_teacher")
                    if T is None:
                        import llm_macro
                        T = self._teacher = llm_macro.Teacher(self)
                    c = T.choose(mk3, int(a))
                    if c is not None:
                        return int(c)
            return a

        # ---------------- learned out-of-combat policy (MACRO_POLICY) ----------------
        _SS = ["EVENT_SCREEN", "REWARDS", "BOSS_RELIC_REWAR", "CARD_SELECT", "MAP_SCREEN", "TREASURE_ROOM",
               "REST_ROOM", "SHOP_ROOM", "BATTLE", "GAME_OVER", "NEOW", "OTHER"]

        def _cand_feats(self, mask):
            """[n, 2416] float32 features for legal options 0..n-1 (honest afterstates on a determinized copy)."""
            import numpy as np
            n = min(int(mask.sum()), MACRO_MAX_CANDS)
            live = self._hsnap()
            o0 = self._obs1200().astype(np.float32)
            ss = str(self.env.gc.screen_state).split(".")[-1]
            oh = np.zeros(16, np.float32)
            oh[self._SS.index(ss) if ss in self._SS else 11] = 1
            det = self._hdet(0)
            gl = self.env.gc
            if MACRO_MAPFEAT:
                import mapfeat
                pos0 = (int(gl.cur_map_node_x), int(gl.cur_map_node_y), int(gl.act))
                m0 = mapfeat.summary(gl, pos0[0], pos0[1])
            F = np.zeros((n, 2416 + (2 * mapfeat.N if MACRO_MAPFEAT else 0)), np.float32)
            for k in range(n):
                self._hload(live, det)
                try:
                    self._ff_step(k)
                    o1 = self._obs1200().astype(np.float32)
                    if MACRO_MAPFEAT:
                        g = self.env.gc
                        pos1 = (int(g.cur_map_node_x), int(g.cur_map_node_y), int(g.act))
                        m1 = m0 if pos1 == pos0 or pos1[2] != pos0[2] else mapfeat.summary(gl, pos1[0], pos1[1])
                except Exception:
                    o1 = o0
                    m1 = m0 if MACRO_MAPFEAT else None
                F[k, :1200] = o0
                F[k, 1200:2400] = o1 - o0
                F[k, 2400:2416] = oh
                if MACRO_MAPFEAT:
                    F[k, 2416:2416 + mapfeat.N] = m0
                    F[k, 2416 + mapfeat.N:] = m1
            self._hload(live)
            return F, ss

        def _policy_action(self):
            import numpy as np
            mask = self._mask()
            if int(mask.sum()) < 2:
                return None
            P = self.__dict__.get("_pol")
            if P is None:
                z = np.load(MACRO_POLICY)
                P = self._pol = [(z["l%d.weight" % i], z["l%d.bias" % i]) for i in range(3)]
                self._pol_rng = np.random.default_rng((int(self.seed) * 2654435761 + 97) % (1 << 32))
                self._pol_log = []
            F, ss = self._cand_feats(mask)
            h = F
            for li, (W, b) in enumerate(P):
                x = h @ W.T + b
                h = x if li == len(P) - 1 else 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
            lg = h[:, 0].astype(np.float64)
            pr = np.exp(lg - lg.max()); pr /= pr.sum()
            a = int(self._pol_rng.choice(len(pr), p=pr)) if MACRO_POLICY_SAMPLE else int(np.argmax(lg))
            if MACRO_POLICY_LOG:
                gc = self.env.gc
                self._pol_log.append((F.astype(np.float16), a, float(np.log(pr[a] + 1e-12)), ss, int(gc.floor_num), int(gc.act)))
            return a

        def _pol_dump(self, seed, out, path):
            import numpy as np
            L = self.__dict__.get("_pol_log", [])
            os.makedirs(path, exist_ok=True)
            n = [len(x[0]) for x in L]
            p = os.path.join(path, "seed_%d" % int(seed))
            np.savez_compressed(p + ".tmp.npz",
                                feats=np.concatenate([x[0] for x in L]) if L else np.zeros((0, 2416), np.float16),
                                ncand=np.array(n, np.int16), action=np.array([x[1] for x in L], np.int16),
                                logp=np.array([x[2] for x in L], np.float32), ss=np.array([x[3] for x in L], "U16"),
                                floor=np.array([x[4] for x in L], np.int16), act=np.array([x[5] for x in L], np.int16),
                                run=np.array([int(seed), int(out["floor"]), int(out["win"]), int(out["act"])], np.int32))
            os.replace(p + ".tmp.npz", p + ".npz")

        # ---------------- Phase-3 harvest (HARVEST=<dir>) ----------------
        def _hv(self):
            return self.__dict__.setdefault("_hv_d", {"c": [], "f": [], "m": [], "open": None})

        def _hv_combat(self, legal, scores, best, K):
            h = self._hv()
            e = self.env
            gc, bc = e.gc, e.bc
            f = h["open"]
            if f is None:
                acts = bc.get_possible_actions()
                pots = len({a.bits & 0xFFFF for a in acts if ((a.bits >> 29) & 0xF) == 1})
                f = {"room": self._room(), "floor": int(gc.floor_num), "act": int(gc.act),
                     "hp_in": int(bc.player.cur_hp), "mhp": int(gc.max_hp), "pot_in": pots,
                     "pot_used": 0, "ndec": 0}
                h["open"] = f
            try:
                obs = self._obs1235()
            except Exception:
                obs = None
            f["pot_used"] += int(self._is_potion(best))
            f["ndec"] += 1
            h["c"].append((obs, list(legal), [list(r) for r in scores], int(best), len(h["f"]),
                           int(bc.player.cur_hp), int(bc.player.block), int(K)))

        def _hv_close(self):
            h = self._hv()
            f = h["open"]
            if f is None:
                return
            gc = self.env.gc
            f["hp_out"] = max(0, int(gc.cur_hp))
            f["mhp_out"] = int(gc.max_hp)
            f["won"] = int(gc.outcome != slaythespire.GameOutcome.PLAYER_LOSS)
            h["f"].append(f)
            h["open"] = None

        def _hv_dump(self, seed, out):
            import numpy as np
            h = self._hv()
            self._hv_close()
            c = [r for r in h["c"] if r[0] is not None]
            T = max([len(r[1]) for r in c] + [1])
            Kx = max([r[7] for r in c] + [1])
            N = len(c)
            q = np.full((N, T, Kx), np.nan, np.float32)
            leg = np.full((N, T), -1, np.int16)
            for i, r in enumerate(c):
                leg[i, :len(r[1])] = r[1]
                for j, row in enumerate(r[2]):
                    q[i, j, :len(row)] = row
            F = h["f"]
            M = [r for r in h["m"]]
            d = dict(
                c_obs=np.array([r[0] for r in c], np.float32).reshape(N, -1),
                c_legal=leg, c_q=q, c_chosen=np.array([r[3] for r in c], np.int16),
                c_fight=np.array([r[4] for r in c], np.int32), c_hp=np.array([r[5] for r in c], np.int16),
                c_block=np.array([r[6] for r in c], np.int16), c_k=np.array([r[7] for r in c], np.int16),
                f_room=np.array([x["room"] for x in F], "U8"),
                **{"f_" + k: np.array([x[k] for x in F], np.int16)
                   for k in ("floor", "act", "hp_in", "hp_out", "mhp", "mhp_out", "pot_in", "pot_used", "ndec", "won")},
                m_obs=np.array([r[0] for r in M], np.float32).reshape(len(M), -1),
                m_mask=np.array([r[1] for r in M], np.int8).reshape(len(M), -1),
                m_ss=np.array([r[2] for r in M], "U16"),
                **{"m_" + k: np.array([r[i] for r in M], np.int16)
                   for i, k in enumerate(("action", "floor", "act", "hp", "mhp", "gold", "deck"), start=3)},
                run=np.array([int(seed), int(out["floor"]), int(out["win"]), int(out["act"])], np.int32))
            os.makedirs(HARVEST, exist_ok=True)
            p = os.path.join(HARVEST, "seed_%d" % int(seed))
            np.savez_compressed(p + ".tmp.npz", **d)
            os.replace(p + ".tmp.npz", p + ".npz")
            out["harvest"] = {"decisions": N, "fights": len(F), "macro": len(M)}

        def play_v2(self, seed):
            out = super().play_v2(seed)
            if HARVEST:
                self._hv_dump(seed, out)
            if MACRO_POLICY and MACRO_POLICY_LOG:
                self._pol_dump(seed, out, MACRO_POLICY_LOG)
            if MACRO_FEAT_LOG and not MACRO_POLICY:
                self._pol_dump(seed, out, MACRO_FEAT_LOG)
            if LLM_MACRO and self.__dict__.get("_teacher") is not None:
                out["llm"] = self._teacher.summary()
            return out

        def _smith_best(self):
            return self._hwrapped("_smith_best", True)

        def _cnet(self):
            c = self.__dict__.get("_cnet_obj")
            if c is None:
                if HONEST_ROLLOUT_NET:
                    import numpy as np
                    z = np.load(HONEST_ROLLOUT_NET)
                    c = slaythespire.CombatNet([z["pbody.%d.weight" % i] for i in (0, 2, 4)] + [z["pi.weight"]],
                                               [z["pbody.%d.bias" % i] for i in (0, 2, 4)] + [z["pi.bias"]],
                                               self.maxima.astype("float32"))
                    self._cnet_obj = c
                    return c
                n = self.net
                c = slaythespire.CombatNet([w for w, _ in n.pbody] + [n.pi[0]],
                                           [b for _, b in n.pbody] + [n.pi[1]],
                                           self.maxima.astype("float32"))
                self._cnet_obj = c
            return c

        def _prior_net(self):
            n = self.__dict__.get("_prior_obj")
            if n is None:
                from np_policy import NpNet
                n = self._prior_obj = NpNet(HONEST_PRIOR)
            return n

        def _vnet(self):
            c = self.__dict__.get("_vnet_obj")
            if c is None:
                import numpy as np
                z = np.load(HONEST_VNET)
                c = slaythespire.CombatNet([z["pbody.%d.weight" % i] for i in (0, 2, 4)] + [z["pi.weight"]],
                                           [z["pbody.%d.bias" % i] for i in (0, 2, 4)] + [z["pi.bias"]],
                                           self.maxima.astype("float32"))
                self._vnet_obj = c
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
                ck = getattr(self, "_mck", MACRO_COMBAT_K)
                K = ck if (MACRO_CARD or MACRO_REST) and ck > 0 else HONEST_TAIL_K
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
            topk = TOPK if (getattr(self, "_in_rollout", False) or not HONEST_TOPK) else HONEST_TOPK
            if len(legal) > topk:
                pnet = self._prior_net() if HONEST_PRIOR else self.net
                prior = pnet.logits(self._obs1235(), mask)
                legal = sorted(legal, key=lambda a: prior[a], reverse=True)[:topk]
            clair = self.__dict__.get("_clair", False)     # headroom diagnostic only (headroom.py); never set in play
            dets = [0] if clair else [self._hdet(k) for k in range(K)]
            if HONEST_CPP:
                # native batched rollouts: same worlds, same policy, same score
                e = self.env
                last = getattr(e, "_last", None)
                if HONEST_VNET and HONEST_DEPTH >= 0:
                    scores, _ = slaythespire.rollout_qb(e.gc, e.bc, legal, dets, self._cnet(), 0,
                                                        GAMMA, ROLL_CAP, False,
                                                        [] if last is None else [int(v) for v in last],
                                                        self._vnet(), HONEST_DEPTH, HONEST_VSCALE)
                else:
                    scores, _ = slaythespire.rollout_qb(e.gc, e.bc, legal, dets, self._cnet(), 0,
                                                        GAMMA, ROLL_CAP, clair,
                                                        [] if last is None else [int(v) for v in last])
                best, bq = legal[0], -1e18
                for a, row in zip(legal, scores):
                    tot = 0.0
                    for v in row:
                        tot += v
                    qa = tot / len(dets)
                    if qa > bq:
                        bq, best = qa, a
                if HARVEST and not getattr(self, "_in_rollout", False):
                    self._hv_combat(legal, scores, best, K)
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

        # ---------------- honest macro search (MACRO_CARD / MACRO_REST) ----------------
        def _choose(self, ss, labels, deck, strike, defend, gold, act):
            if not getattr(self, "_in_rollout", False):
                if MACRO_CARD and ss in ("REWARDS", "CARD_SELECT") and any(L["dd"] >= 1 for L in labels)                         and not any(L.get("ud", 0) >= 1 for L in labels)                         and (not MACRO_CARD_ACTS or int(self.env.gc.act) in MACRO_CARD_ACTS):
                    takes = sorted([L for L in labels if L["dd"] >= 1], key=lambda x: x["i"])
                    noops = [L for L in labels if L["dd"] == 0 and L["rd"] == 0]
                    leave = [L for L in noops if L["to"] == "MAP_SCREEN"]
                    skip = leave or noops
                    cands = [L["i"] for L in takes] + ([skip[0]["i"]] if skip else [])
                    if len(cands) >= 2:
                        return self._macro_search("card", cands)
                if MACRO_REST and ss == "REST_ROOM":
                    smith = [L for L in labels if L["to"] == "CARD_SELECT" and L["dd"] == 0]
                    rest = [L for L in labels if L["hp_d"] > 0]
                    if smith and rest:
                        return self._macro_search("rest", [rest[0]["i"], smith[0]["i"]])
            return super()._choose(ss, labels, deck, strike, defend, gold, act)

        def _macro_search(self, kind, cands):
            gc = self.env.gc
            f0, a0 = int(gc.floor_num), int(gc.act)
            P = MACRO_KIND[kind]
            fmax = min(f0 + P["H"], 17 * a0)            # never past this act's boss floor
            live = self._hsnap()
            hist, srf = list(self.history), self._shop_removed_floor
            stack = self.__dict__.get("_hstack", [])
            saved = list(stack)
            dets = [self._hdet(k) for k in range(P["K"])]
            self._in_rollout = True
            self._mck = P["CK"]
            vals = []
            try:
                for ci in cands:
                    v = 0.0
                    for d in dets:                          # common random numbers across candidates
                        self._shop_removed_floor = srf
                        v += self._mrollout(live, ci, d, fmax)
                    vals.append(v / len(dets))
            finally:
                self._in_rollout = False
                stack[:] = saved
                self.history = hist
                self._shop_removed_floor = srf
                self._hload(live)
            best = cands[max(range(len(cands)), key=lambda j: vals[j])]
            if MACRO_LOG:
                self.__dict__.setdefault("macro_log", []).append(
                    {"kind": kind, "floor": f0, "cands": cands, "vals": [round(x, 3) for x in vals], "pick": best})
            return int(best)

        def _mrollout(self, live, first_a, det, fmax, cap=3000):
            self._hload(live, det)
            e = self.env
            e.step(int(first_a))
            steps = 0
            while steps < cap and e.gc.outcome == _UND:
                if not e.in_combat and int(e.gc.floor_num) >= fmax:
                    break
                if e.in_combat:
                    if getattr(self, "_mck", MACRO_COMBAT_K) > 0:
                        a = self._search_action()
                    else:
                        a = self._cnet().greedy_state(e.gc, e.bc)
                else:
                    a = self._macro_action()
                if a is None:
                    break
                e.step(int(a))
                steps += 1
            gc = e.gc
            if gc.outcome == _VIC:
                return 100.0 + int(gc.floor_num)
            if gc.outcome != _UND:
                return float(int(gc.floor_num))
            return int(gc.floor_num) + MACRO_HPW * gc.cur_hp / max(1, gc.max_hp)

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
