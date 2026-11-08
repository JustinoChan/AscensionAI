"""LLM out-of-combat teacher (local Ollama only; never a paid or remote API).

Toggles (all default off; LLM_MACRO unset => this module is never imported):
  LLM_MACRO=1           an LLM chooses at live out-of-combat decisions with >= 2 options
  LLM_MODEL=<name>      Ollama model (default qwen2.5-coder:7b)
  LLM_SCREENS=a,b       screens it decides (default all), e.g. REWARDS,CARD_SELECT,SHOP_ROOM,REST_ROOM,MAP_SCREEN
  LLM_HOST=<url>        default: http://<WSL default gateway>:11434 (Windows Ollama), else 127.0.0.1
  LLM_TIMEOUT=<s>       per request (default 60); on timeout / bad answer -> fall back to the
                        current policy and count the fallback
  LLM_DRYRUN=1          build prompts and log them, call nothing, always fall back
Honest: the prompt holds only what a player sees -- HP, gold, deck, relics, potions, floor/act/boss,
the visible map (map layout is regenerated from the run seed; it is fully visible in-game) and,
for each option, its visible effect computed on a DETERMINIZED copy (cards gained/removed/upgraded,
relics, gold/HP change, next screen). Unknown map nodes stay '?'.
Every decision is appended as one JSON line to <run dir>/llm_seed_<seed>.jsonl
(state text, options, choice, reason, latency, fallback flag) -- distillation data.
Temperature 0, fixed seed, JSON-schema output via Ollama's `format`.
"""
import collections, json, os, subprocess, time, urllib.request

import slaythespire as S

MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
SCREENS = [x.strip() for x in os.environ.get("LLM_SCREENS", "").split(",") if x.strip()]
# default: real choices only (map routing and free pickups stay with the current policy; LLM_SCREENS=ALL... to override)
DEFAULT_SCREENS = ["REWARDS", "CARD_SELECT", "REST_ROOM", "SHOP_ROOM", "BOSS_RELIC_REWAR", "EVENT_SCREEN"]
if SCREENS == ["ALL"]:
    SCREENS = DEFAULT_SCREENS + ["MAP_SCREEN", "TREASURE_ROOM"]
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
DRYRUN = os.environ.get("LLM_DRYRUN", "0") == "1"
THINK = os.environ.get("LLM_THINK", "")      # "1"/"0": send Ollama's think flag (thinking models, e.g. qwen3); unset: omit


def _host():
    h = os.environ.get("LLM_HOST")
    if h:
        return h.rstrip("/")
    try:
        gw = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=5).stdout
        ip = next(l.split()[2] for l in gw.splitlines() if l.startswith("default"))
        return "http://%s:11434" % ip
    except Exception:
        return "http://127.0.0.1:11434"


HOST = _host()
SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}, "option": {"type": "string"},
                                          "reason": {"type": "string"}},
          "required": ["choice", "option", "reason"]}
SYSTEM = ("You are an expert Slay the Spire player playing the Ironclad on Ascension 0. Your goal is to WIN the run "
          "(defeat the Act 3 boss), not just to survive the next fight: value scaling, card quality, deck size and "
          "HP management over the whole run. You see only what a human player sees. Pick exactly one option by its "
          "number. Answer as JSON: {\"choice\": <option number>, \"option\": \"<that option's text, copied exactly>\", "
          "\"reason\": \"<at most 20 words>\"}.")


def _n(x):
    return str(x).split(".")[-1]


def _deck(gc):
    return collections.Counter(_n(c.id) + ("+" if c.upgraded else "") for c in gc.deck)


def _relics(gc):
    return collections.Counter(_n(r.id) for r in gc.relics)


def _potions(env):
    try:
        raw = list(env.nn.getObservation(env.gc, None))[417:422]
        return [int(v) for v in raw if v >= 2]
    except Exception:
        return []


def _map_text(gc):
    try:
        m = S.SpireMap(int(gc.seed), 0, int(gc.act), True)
        return repr(m)
    except Exception:
        return ""


def _room_at(gc, x, y):
    try:
        m = S.SpireMap(int(gc.seed), 0, int(gc.act), True)     # LIVE gc: current act's map is visible
        r = _n(m.get_room_type(x, y))
        return {"EVENT": "? (unknown)", "MONSTER": "monster", "ELITE": "ELITE", "REST": "rest site", "SHOP": "shop",
                "TREASURE": "treasure", "BOSS": "BOSS"}.get(r, r)
    except Exception:
        return "?"


class Teacher:
    def __init__(self, agent):
        self.a = agent
        self.stats = {"decisions": 0, "llm": 0, "fallbacks": 0, "latency": 0.0, "dryrun": DRYRUN, "warmup_s": 0.0}
        self.log = None
        if not DRYRUN:                       # load the model once (not counted as a decision)
            t0 = time.time()
            try:
                body = json.dumps({"model": MODEL, "stream": False, "keep_alive": "30m", "options": {"num_predict": 1},
                                   "messages": [{"role": "user", "content": "ok"}]}).encode()
                urllib.request.urlopen(urllib.request.Request(HOST + "/api/chat", data=body,
                                       headers={"Content-Type": "application/json"}), timeout=600).read()
            except Exception:
                pass
            self.stats["warmup_s"] = round(time.time() - t0, 1)

    def _logf(self):
        if self.log is None:
            d = os.environ.get("HONEST_HB") or "sim_logs/llm"
            os.makedirs(d, exist_ok=True)
            self.log = open(os.path.join(d, "llm_seed_%d.jsonl" % int(self.a.seed)), "a")
        return self.log

    def describe(self, mask):
        """Visible state text + one line per legal option (effects on a determinized copy)."""
        A = self.a
        e = A.env
        gc = e.gc
        n = min(int(mask.sum()), 30)
        ss = _n(gc.screen_state)
        d0, r0, g0, h0, m0, p0 = _deck(gc), _relics(gc), int(gc.gold), int(gc.cur_hp), int(gc.max_hp), _potions(e)
        live = A._hsnap()
        ctr = A.__dict__.get("_hctr", 0)
        det = A._hdet(0)
        A._hctr = ctr          # describing options must not shift the combat-sample stream (fallback == baseline)
        opts = []
        for k in range(n):
            A._hload(live, det)
            try:
                A._ff_step(k)
                g = A.env.gc
                parts = []
                dd = _deck(g)
                add, rem = dd - d0, d0 - dd
                for c, v in add.items():
                    parts.append("add %s%s" % (c, " x%d" % v if v > 1 else ""))
                for c, v in rem.items():
                    parts.append("remove %s%s" % (c, " x%d" % v if v > 1 else ""))
                for rr in (_relics(g) - r0):
                    parts.append("gain relic %s" % rr)
                if len(_potions(A.env)) > len(p0):
                    parts.append("gain a potion")
                if int(g.gold) != g0:
                    parts.append("gold %+d" % (int(g.gold) - g0))
                if int(g.max_hp) != m0:
                    parts.append("max HP %+d" % (int(g.max_hp) - m0))
                if int(g.cur_hp) != h0:
                    parts.append("HP %+d" % (int(g.cur_hp) - h0))
                if ss == "MAP_SCREEN":
                    parts.insert(0, "go to %s node at (x=%d, y=%d)" % (_room_at(gc, int(g.cur_map_node_x), int(g.cur_map_node_y)),
                                                                   int(g.cur_map_node_x), int(g.cur_map_node_y)))
                nxt = _n(g.screen_state)
                if nxt != ss and ss != "MAP_SCREEN":
                    parts.append("-> %s" % nxt)
                opts.append("; ".join(parts) or "no visible change (skip / leave / proceed)")
            except Exception:
                opts.append("(effect unknown)")
        A._hload(live)
        deck = ", ".join("%s x%d" % (c, v) if v > 1 else c for c, v in sorted(d0.items()))
        st = ["Screen: %s. Act %d, floor %d. HP %d/%d, gold %d." % (ss, int(gc.act), int(gc.floor_num), h0, m0, g0),
              "Act boss: %s." % _n(gc.boss),
              "Deck (%d cards): %s." % (sum(d0.values()), deck),
              "Relics: %s." % (", ".join(sorted(r0)) or "none"),
              "Potions held: %d of 3 slots." % len(p0)]
        if ss == "MAP_SCREEN":
            st.append("Map (current act; M monster, E elite, R rest, $ shop, ? unknown, T treasure, B boss); you are at (x=%d, y=%d):\n%s"
                      % (int(gc.cur_map_node_x), int(gc.cur_map_node_y), _map_text(gc)))
        return "\n".join(st), opts, ss

    def choose(self, mask, policy_choice=None):
        """Returns an option index, or None to fall back to the current policy."""
        gc0 = self.a.env.gc
        ss0 = _n(gc0.screen_state)
        if ss0 not in (SCREENS or DEFAULT_SCREENS):
            return None
        state, opts, ss = self.describe(mask)
        # REWARDS: free pickups (gold / potion / relic) stay with the policy; the LLM only picks among cards (+ skip)
        if ss == "REWARDS" and not any(o.startswith("add ") for o in opts):
            return None
        self.stats["decisions"] += 1
        user = state + "\n\nOptions:\n" + "\n".join("%d. %s" % (i + 1, o) for i, o in enumerate(opts)) + "\n\nWhich option?"
        rec = {"seed": int(self.a.seed), "floor": int(self.a.env.gc.floor_num), "screen": ss, "state": state, "options": opts,
               "model": MODEL, "think": THINK, "policy_choice": policy_choice, "t": time.time()}
        choice, reason, err, lat = None, "", None, 0.0
        if DRYRUN:
            err = "dryrun"
        else:
            req_d = {"model": MODEL, "stream": False, "format": SCHEMA, "keep_alive": "30m",
                     "options": {"temperature": 0, "seed": 42, "num_predict": 2048 if THINK == "1" else 96},
                     "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
            if THINK in ("0", "1"):
                req_d["think"] = THINK == "1"
            body = json.dumps(req_d).encode()
            t0 = time.time()
            try:
                req = urllib.request.Request(HOST + "/api/chat", data=body, headers={"Content-Type": "application/json"})
                resp = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
                ans = json.loads(resp["message"]["content"])
                c = int(ans["choice"]) - 1                       # the prompt numbers options from 1
                txt = str(ans.get("option", "")).strip().lower()
                reason = str(ans.get("reason", ""))[:200]
                hits = [i for i, o in enumerate(opts) if txt and txt == o.lower()]
                if hits and c not in hits and len(set(opts[i] for i in hits)) == 1:
                    rec["index_fixed"] = [c, hits[0]]             # number and echoed text disagree: trust the text
                    c = hits[0]
                if 0 <= c < len(opts):
                    choice = c
                else:
                    err = "out of range: %s" % (c + 1)
            except Exception as ex:
                err = "%s: %s" % (type(ex).__name__, str(ex)[:120])
            lat = time.time() - t0
            self.stats["latency"] += lat
        if choice is None:
            self.stats["fallbacks"] += 1
        else:
            self.stats["llm"] += 1
        rec.update({"choice": choice, "reason": reason, "error": err, "latency": round(lat, 3), "fallback": choice is None})
        self._logf().write(json.dumps(rec) + "\n"); self.log.flush()
        return choice

    def summary(self):
        s = dict(self.stats)
        s["mean_latency"] = round(s["latency"] / max(1, s["decisions"]), 3) if not DRYRUN else 0.0
        s["fallback_rate"] = round(s["fallbacks"] / max(1, s["decisions"]), 3)
        return s
