"""TRACE=1: passive, decision-level trace of one full run (for inspect_seed.py).

Installed on an evaluator instance (V2 or an honest subclass). Observes LIVE
env.step calls only -- steps made inside search/macro lookaheads are skipped
via a busy counter around _search_action/_macro_action -- and never changes
a decision. Adds out["trace"] to the run JSON:
  floors : per floor entered {floor, act, room, x, y, hp, max_hp, gold}
  acts   : per act entered {act, floor, deck[], relics[]}
  fights : {floor, act, room, enc, hp_start, max_hp, potions_start[], turns,
            potions_used[], hp_end, hp_lost, died}
  events : {floor, kind: card+/card-/relic+/potion+/upgrade, what}
  offers : card rewards seen {floor, act, offered[], picked (name or null = skipped)}
"""
import collections
import slaythespire as S

_UND = S.GameOutcome.UNDECIDED
_LOSS = S.GameOutcome.PLAYER_LOSS


def _name(x):
    return str(x).split(".")[-1]


def _potion_name(v):
    return int(v)          # Potion enum id; inspect_seed.py maps ids -> names from Potions.h


def install(ev):
    env = ev.env
    T = {"floors": [], "acts": [], "fights": [], "events": [], "offers": []}
    ev.trace = T
    st = {"busy": 0, "fight": None, "act": None, "floor": None}

    def deck():
        try:
            return [_name(c.id) + ("+" if c.upgraded else "") for c in env.gc.deck]
        except Exception:
            return []

    def relics():
        try:
            return [_name(r.id) for r in env.gc.relics]
        except Exception:
            return []

    def raw_slots():
        try:   # nn.getObservation directly: no FightEnv._last side effect
            return list(env.nn.getObservation(env.gc, env.bc if env.in_combat else None))[417:422]
        except Exception:
            return []

    def potions():
        return [_potion_name(v) for v in raw_slots() if v >= 2]

    def hp():
        return int(env.bc.player.cur_hp) if (env.in_combat and env.bc is not None) else int(env.gc.cur_hp)

    def snap_state():
        return collections.Counter(deck()), collections.Counter(relics()), collections.Counter(potions())

    def note_floor():
        gc = env.gc
        if st["act"] != int(gc.act):
            st["act"] = int(gc.act)
            T["acts"].append({"act": int(gc.act), "floor": int(gc.floor_num), "deck": deck(), "relics": relics(),
                              "hp": hp(), "max_hp": int(gc.max_hp)})
        if st["floor"] != int(gc.floor_num):
            st["floor"] = int(gc.floor_num)
            T["floors"].append({"floor": int(gc.floor_num), "act": int(gc.act), "room": _name(gc.cur_room),
                                "x": int(gc.cur_map_node_x), "y": int(gc.cur_map_node_y), "hp": hp(),
                                "max_hp": int(gc.max_hp), "gold": int(gc.gold)})

    for meth in ("_search_action", "_macro_action"):
        orig_m = getattr(ev, meth)

        def wrapped(*a, _o=orig_m, **kw):
            st["busy"] += 1
            try:
                return _o(*a, **kw)
            finally:
                st["busy"] -= 1
        setattr(ev, meth, wrapped)

    orig_step = env.step

    def step(idx):
        if st["busy"]:
            return orig_step(idx)
        pre_in = bool(env.in_combat)
        if st["act"] is None:
            note_floor()
        pre_slots = raw_slots()
        is_pot, pot_name = False, None
        if pre_in:
            try:
                acts = env.bc.get_possible_actions()
                a = acts[idx] if idx < len(acts) else acts[0]
                is_pot = ((a.bits >> 29) & 0xF) == 1
                if is_pot:
                    src = a.bits & 0xFFFF
                    pot_name = int(pre_slots[src]) if src < len(pre_slots) else -1
            except Exception:
                pass
            pre_turn = int(env.bc.turn)
        else:
            before = snap_state()
            offer = None
            if _name(env.gc.screen_state) in ("REWARDS", "CARD_SELECT"):
                try:
                    offered = sorted(_name(c.id) for c in env.gc.get_card_reward())
                except Exception:
                    offered = []
                if offered:
                    key = (int(env.gc.floor_num), tuple(offered))
                    if st.get("offer_key") != key:
                        st["offer_key"] = key
                        T["offers"].append({"floor": key[0], "act": int(env.gc.act), "offered": offered, "picked": None})
                    offer = T["offers"][-1]
        out = orig_step(idx)
        post_in = bool(env.in_combat)
        gc = env.gc
        if not pre_in and post_in:
            st["fight"] = {"floor": int(gc.floor_num), "act": int(gc.act), "room": _name(gc.cur_room),
                           "enc": _name(gc.encounter), "hp_start": int(env.bc.player.cur_hp),
                           "max_hp": int(gc.max_hp), "potions_start": potions(), "turns": 0,
                           "potions_used": [], "died": False}
        if pre_in and st["fight"] is not None:
            f = st["fight"]
            if is_pot:
                f["potions_used"].append(pot_name if pot_name is not None else -1)
            if post_in:
                f["turns"] = max(f["turns"], int(env.bc.turn))
            else:
                f["turns"] = max(f["turns"], pre_turn)
                f["died"] = gc.outcome == _LOSS
                f["hp_end"] = 0 if f["died"] else int(gc.cur_hp)
                f["hp_lost"] = f["hp_start"] - f["hp_end"]
                T["fights"].append(f)
                st["fight"] = None
        if not pre_in and not post_in:
            after = snap_state()
            fl = int(gc.floor_num)
            if offer is not None and offer["picked"] is None:
                for k in (after[0] - before[0]):
                    if k.rstrip("+") in offer["offered"]:
                        offer["picked"] = k.rstrip("+")
            for kind, i in (("card", 0), ("relic", 1), ("potion", 2)):
                for k, v in (after[i] - before[i]).items():
                    for _ in range(v):
                        T["events"].append({"floor": fl, "kind": kind + "+", "what": k})
                if kind != "potion":
                    for k, v in (before[i] - after[i]).items():
                        for _ in range(v):
                            T["events"].append({"floor": fl, "kind": kind + "-", "what": k})
        note_floor()
        return out

    env.step = step
    return T
