"""Data layer for the control-panel analytics (JSON on stdout; cheap, cached).

  dash_data.py progress                 every ledger run: seed set, n, wins + Wilson CI, floor + CI, honest/clairvoyant
  dash_data.py deaths RUN [RUN_B]       deaths by floor x room, killer encounters (needs traces), diff vs RUN_B
  dash_data.py fights RUN               per-encounter win rate / HP lost / potions (traces); act-entry HP winners vs losers;
                                        harvest per-(act, room) fight stats (mainline_h1, cached)
  dash_data.py cards RUN                pick rate when offered, mean floor / win rate when taken; deck size + upgrades
                                        at act entry, winners vs losers (traces)
  dash_data.py seeddiff SEED RUN_A RUN_B  first divergence + floor/HP curves (replays a missing trace, TRACE=1, cached)
  dash_data.py health                   CPU, RAM, GPU, disk, throughput, watchdog heartbeat, queue summary, alerts
  dash_data.py training                 per-epoch curves of the GPU nets (nets/*/train_history.json or parsed logs)
  dash_data.py digest [--write]         daily digest (markdown) -> sim_logs/DIGEST_<date>.md
  dash_data.py traced                   runs that have trace data
Traces come from sim_logs/traces/<run>/seed_*.json (trace_backfill.sh) or a run's own TRACE=1 seed JSON.
"""
import collections, glob, json, math, os, re, sqlite3, subprocess, sys, time

os.chdir(os.path.expanduser("~/sts"))
DB = "sim_logs/results.db"
EVAL0, HOLD0 = 10007, 20011
REF = {"eval": "cpp_1ply_K64", "holdout": "holdout_K64"}


def con():
    return sqlite3.connect(DB)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def seed_set(seeds):
    s = set(seeds)
    if EVAL0 in s:
        return "eval"
    if HOLD0 in s:
        return "holdout"
    return "training" if s and min(s) >= 1197857 else "other"


# ------------------------------------------------------------------ progress
def progress():
    c = con()
    out = []
    for name, desc, grp, env, started, ended, cv, total in c.execute(
            "select name, description, grp, env, started_at, ended_at, clairvoyant, total from runs"):
        rows = c.execute("select seed, win, floor from seeds where run=? and coalesce(timeout,0)=0", (name,)).fetchall()
        if not rows:
            continue
        n = len(rows); w = sum(r[1] for r in rows); fl = [r[2] for r in rows]
        m = sum(fl) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in fl) / max(1, n - 1))
        lo, hi = wilson(w, n)
        out.append({"name": name, "desc": desc, "grp": grp, "env": env, "t": started or 0, "t_end": ended,
                    "clairvoyant": bool(cv), "set": seed_set([r[0] for r in rows]), "n": n, "total": total,
                    "wins": w, "win_pct": 100.0 * w / n, "win_lo": 100 * lo, "win_hi": 100 * hi,
                    "floor": m, "floor_ci": 1.96 * sd / math.sqrt(n)})
    out.sort(key=lambda r: r["t"])
    best = {}
    for r in out:
        if r["clairvoyant"] or r["n"] < 150 or r["set"] not in ("eval", "holdout"):
            continue
        b = best.get(r["set"])
        if b is None or (r["floor"], r["wins"]) > (b["floor"], b["wins"]):
            best[r["set"]] = r
    return {"runs": out, "best_honest": {k: {x: v[x] for x in ("name", "wins", "n", "floor", "floor_ci", "win_pct")}
                                         for k, v in best.items()}}


# ------------------------------------------------------------------ traces
def run_env(run):
    r = con().execute("select env from runs where name=?", (run,)).fetchone()
    if r and r[0]:
        return r[0]
    p = "sim_logs/honest/%s/.env" % run
    return open(p).read().strip() if os.path.exists(p) else ""


def traces(run):
    """{seed: (run_json_without_trace, trace)} from backfill dir or the run's own TRACE=1 output."""
    out = {}
    for p in glob.glob("sim_logs/traces/%s/seed_*.json" % run):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "trace" in d:
            out[int(d["seed"])] = (d, d.pop("trace"))
    if not out:
        for (raw,) in con().execute("select raw from seeds where run=?", (run,)):
            try:
                d = json.loads(raw)
            except Exception:
                continue
            if "trace" in d:
                out[int(d["seed"])] = (d, d.pop("trace"))
    return out


def traced():
    names = set(os.path.basename(os.path.dirname(p)) for p in glob.glob("sim_logs/traces/*/seed_*.json"))
    for (name,) in con().execute("select distinct run from seeds where raw like '%\"trace\"%'"):
        names.add(name)
    return {"runs": sorted(names),
            "counts": {n: len(glob.glob("sim_logs/traces/%s/seed_*.json" % n)) for n in names}}


# ------------------------------------------------------------------ deaths
def _deaths_one(run):
    c = con()
    rows = c.execute("select seed, win, floor, death_room, death_act, bucket from seeds where run=?", (run,)).fetchall()
    hist = collections.Counter()
    buckets = collections.Counter()
    for seed, win, floor, room, act, b in rows:
        buckets[b or ("WIN" if win else "?")] += 1
        if not win:
            hist[(int(floor), room or "?")] += 1
    enc = {}
    T = traces(run)
    for seed, (d, t) in T.items():
        for f in t.get("fights", []):
            k = "A%d %s" % (f["act"], f["enc"])
            e = enc.setdefault(k, {"enc": f["enc"], "act": f["act"], "room": f["room"], "met": 0, "deaths": 0, "hp_lost": 0})
            e["met"] += 1; e["deaths"] += int(f["died"]); e["hp_lost"] += f.get("hp_lost", 0)
    for e in enc.values():
        e["win_rate"] = 1 - e["deaths"] / e["met"]
        e["mean_hp_lost"] = e["hp_lost"] / e["met"]
    return {"run": run, "n": len(rows), "wins": sum(r[1] for r in rows),
            "hist": [{"floor": f, "room": r, "n": n} for (f, r), n in sorted(hist.items())],
            "buckets": dict(buckets), "traced_seeds": len(T),
            "encounters": sorted(enc.values(), key=lambda e: -e["deaths"])}


def deaths(run, run_b=None):
    a = _deaths_one(run)
    out = {"a": a}
    if run_b:
        b = _deaths_one(run_b)
        out["b"] = b
        ka = {(e["act"], e["enc"]): e for e in a["encounters"]}
        kb = {(e["act"], e["enc"]): e for e in b["encounters"]}
        diff = []
        for k in set(ka) | set(kb):
            ea, eb = ka.get(k, {}), kb.get(k, {})
            diff.append({"act": k[0], "enc": k[1], "deaths_a": ea.get("deaths", 0), "deaths_b": eb.get("deaths", 0),
                         "met_a": ea.get("met", 0), "met_b": eb.get("met", 0),
                         "win_a": ea.get("win_rate"), "win_b": eb.get("win_rate")})
        out["enc_diff"] = sorted(diff, key=lambda x: -abs(x["deaths_b"] - x["deaths_a"]))
        out["bucket_diff"] = {k: [a["buckets"].get(k, 0), b["buckets"].get(k, 0)] for k in set(a["buckets"]) | set(b["buckets"])}
    return out


# ------------------------------------------------------------------ fights
def harvest_fights(hv="sim_logs/harvest/h1"):
    cache = "sim_logs/dash/harvest_fights.json"
    files = glob.glob(hv + "/seed_*.npz")
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("files") == len(files):
            return c
    import numpy as np
    agg = {}
    for f in files:
        z = np.load(f)
        for room, act, hi, ho, mhp, pu, won in zip(z["f_room"], z["f_act"], z["f_hp_in"], z["f_hp_out"], z["f_mhp"],
                                                  z["f_pot_used"], z["f_won"]):
            k = "%d_%s" % (act, room)
            a = agg.setdefault(k, [0, 0, 0.0, 0])
            a[0] += 1; a[1] += int(won); a[2] += max(0, int(hi) - int(ho)) / max(1, int(mhp)); a[3] += int(pu)
    rows = [{"act": int(k.split("_")[0]), "room": k.split("_", 1)[1], "n": v[0], "win_rate": v[1] / v[0],
             "hp_lost_frac": v[2] / v[0], "potions_per_fight": v[3] / v[0]} for k, v in agg.items()]
    out = {"files": len(files), "rows": sorted(rows, key=lambda r: (r["act"], r["room"]))}
    os.makedirs("sim_logs/dash", exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def fights(run):
    T = traces(run)
    enc = {}
    entry = {"win": collections.defaultdict(list), "loss": collections.defaultdict(list)}
    for seed, (d, t) in T.items():
        for f in t.get("fights", []):
            k = (f["act"], f["enc"])
            e = enc.setdefault(k, {"act": f["act"], "enc": f["enc"], "room": f["room"], "n": 0, "won": 0, "hp": 0, "pots": 0, "turns": 0})
            e["n"] += 1; e["won"] += int(not f["died"]); e["hp"] += f.get("hp_lost", 0)
            e["pots"] += len(f.get("potions_used", [])); e["turns"] += f.get("turns", 0)
        for fl in t.get("floors", []):         # HP walking into each act boss (act entry is always full HP)
            if fl["room"] == "BOSS":
                entry["win" if d.get("win") else "loss"][fl["act"]].append(fl["hp"] / max(1, fl["max_hp"]))
    rows = [{**e, "win_rate": e["won"] / e["n"], "mean_hp_lost": e["hp"] / e["n"], "potions": e["pots"] / e["n"],
             "turns": e["turns"] / e["n"]} for e in enc.values()]
    return {"run": run, "traced_seeds": len(T), "encounters": sorted(rows, key=lambda r: (r["act"], -r["n"])),
            "boss_entry_hp": {k: {str(a): v for a, v in vv.items()} for k, vv in entry.items()},
            "harvest": harvest_fights()}


# ------------------------------------------------------------------ cards
def cards(run):
    T = traces(run)
    card = {}
    acts = {"win": collections.defaultdict(list), "loss": collections.defaultdict(list)}
    for seed, (d, t) in T.items():
        won, floor = int(d.get("win", 0)), int(d.get("floor", 0))
        taken = set()
        for o in t.get("offers", []):
            for nm in o["offered"]:
                c = card.setdefault(nm, {"card": nm, "offered": 0, "picked": 0, "runs": 0, "wins": 0, "floor_sum": 0})
                c["offered"] += 1
            if o.get("picked"):
                card[o["picked"]]["picked"] += 1
        for ev in t.get("events", []):
            if ev["kind"] == "card+":
                taken.add(ev["what"].rstrip("+"))
        for nm in taken:
            c = card.setdefault(nm, {"card": nm, "offered": 0, "picked": 0, "runs": 0, "wins": 0, "floor_sum": 0})
            c["runs"] += 1; c["wins"] += won; c["floor_sum"] += floor
        for a in t.get("acts", []):
            dk = a.get("deck", [])
            acts["win" if won else "loss"][a["act"]].append({"size": len(dk), "upgraded": sum(1 for x in dk if x.endswith("+")),
                                                             "starters": sum(1 for x in dk if x.rstrip("+") in ("STRIKE_RED", "DEFEND_RED"))})
    n = max(1, len(T))
    base_floor = sum(d.get("floor", 0) for d, _ in T.values()) / n
    rows = []
    for c in card.values():
        rows.append({**c, "pick_rate": c["picked"] / c["offered"] if c["offered"] else None,
                     "mean_floor": c["floor_sum"] / c["runs"] if c["runs"] else None,
                     "win_rate": c["wins"] / c["runs"] if c["runs"] else None})
    summ = {}
    for k, vv in acts.items():
        summ[k] = {str(a): {m: sum(x[m] for x in v) / len(v) for m in ("size", "upgraded", "starters")} | {"n": len(v)}
                   for a, v in sorted(vv.items())}
    return {"run": run, "traced_seeds": len(T), "base_floor": base_floor,
            "cards": sorted(rows, key=lambda r: -(r["offered"] + r["runs"])), "act_entry_deck": summ}


# ------------------------------------------------------------------ seed diff
def trace_for(run, seed, allow_replay=True):
    p = "sim_logs/traces/%s/seed_%d.json" % (run, seed)
    if os.path.exists(p):
        d = json.load(open(p)); return d, d.pop("trace")
    r = con().execute("select raw from seeds where run=? and seed=?", (run, seed)).fetchone()
    if r:
        d = json.loads(r[0])
        if "trace" in d:
            return d, d.pop("trace")
    if not allow_replay:
        return None, None
    env = run_env(run)
    if not env:
        return None, None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    e = dict(os.environ)
    for kv in env.split():
        k, _, v = kv.partition("=")
        e[k] = v
    e["TRACE"] = "1"
    for k in ("HARVEST", "HONEST_HB"):
        e.pop(k, None)
    out = subprocess.run(["nice", "-n", "19", ".venv/bin/python", "eval_fullrun_analytics_v2.py", str(seed), "macro"],
                         capture_output=True, text=True, env=e).stdout
    d = json.loads(out.strip().splitlines()[-1])
    json.dump(d, open(p, "w"))
    return d, d.pop("trace")


def seeddiff(seed, a, b):
    da, ta = trace_for(a, seed)
    db, tb = trace_for(b, seed)
    if ta is None or tb is None:
        return {"error": "no trace and no env to replay for %s" % (a if ta is None else b)}

    def seq(t):
        s = [("floor", f["floor"], "%s@(%d,%d)" % (f["room"], f["x"], f["y"])) for f in t["floors"]]
        s += [("event", e["floor"], "%s %s" % (e["kind"], e["what"])) for e in t["events"]]
        s += [("fight", f["floor"], "%s hp %d->%d" % (f["enc"], f["hp_start"], f.get("hp_end", 0))) for f in t["fights"]]
        return sorted(s, key=lambda x: (x[1], ["floor", "fight", "event"].index(x[0]), x[2]))
    sa, sb = seq(ta), seq(tb)
    first = None
    for i in range(min(len(sa), len(sb))):
        if sa[i] != sb[i]:
            first = {"index": i, "floor": min(sa[i][1], sb[i][1]), "a": " ".join(map(str, sa[i])), "b": " ".join(map(str, sb[i])),
                     "context": [" ".join(map(str, x)) for x in sa[max(0, i - 4):i]]}
            break
    if first is None and len(sa) != len(sb):
        i = min(len(sa), len(sb))
        first = {"index": i, "floor": (sa + sb)[i][1] if i < len(sa + sb) else None, "a": "end" if i >= len(sa) else " ".join(map(str, sa[i])),
                 "b": "end" if i >= len(sb) else " ".join(map(str, sb[i])), "context": []}
    curve = lambda t: [[f["floor"], f["hp"], f["max_hp"]] for f in t["floors"]]
    return {"seed": seed, "a": {"run": a, "floor": da["floor"], "win": da["win"], "curve": curve(ta)},
            "b": {"run": b, "floor": db["floor"], "win": db["win"], "curve": curve(tb)}, "first_divergence": first}


# ------------------------------------------------------------------ health
def _cpu():
    def snap():
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        return sum(v), v[3] + v[4]
    t0, i0 = snap(); time.sleep(0.5); t1, i1 = snap()
    return 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))


def health():
    h = {"t": time.time(), "cpu_pct": round(_cpu(), 1)}
    mi = dict((l.split(":")[0], int(l.split()[1])) for l in open("/proc/meminfo"))
    h["ram_used_gb"] = round((mi["MemTotal"] - mi["MemAvailable"]) / 2 ** 20, 1)
    h["ram_total_gb"] = round(mi["MemTotal"] / 2 ** 20, 1)
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split(",")
        h["gpu_util"], h["gpu_temp"], h["gpu_mem"], h["gpu_mem_total"] = [int(x) for x in g]
    except Exception:
        h["gpu_util"] = None
    st = os.statvfs(os.path.expanduser("~"))
    h["disk_free_gb"] = round(st.f_bavail * st.f_frsize / 2 ** 30, 1)
    now = time.time()
    recent = [p for p in glob.glob("sim_logs/honest/*/seed_*.json") if now - os.path.getmtime(p) < 1800]
    h["seeds_per_hour"] = len(recent) * 2
    hb = None
    try:
        last = open("sim_logs/overnight/watchdog.log").read().strip().splitlines()[-1]
        hb = now - time.mktime(time.strptime(last[1:20], "%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass
    h["watchdog_age_s"] = hb
    h["keepalive"] = subprocess.run(["pgrep", "-f", "keepalive.sh"], capture_output=True).returncode == 0
    q = {"overnight": {}, "jobq": {}}
    try:
        for j in json.load(open("sim_logs/overnight/state.json"))["jobs"]:
            q["overnight"][j["state"]] = q["overnight"].get(j["state"], 0) + 1
            if j["state"] == "running":
                q["running"] = j["name"]
    except Exception:
        pass
    try:
        for j in json.load(open("sim_logs/jobs/queue.json"))["jobs"]:
            q["jobq"][j["state"]] = q["jobq"].get(j["state"], 0) + 1
            if j["state"] == "running":
                q["running"] = j["name"]
    except Exception:
        pass
    h["queue"] = q
    alerts = []
    if hb is None or hb > 1200:
        alerts.append("watchdog heartbeat %s old (keep-alive down?)" % ("never" if hb is None else "%d min" % (hb // 60)))
    if not h["keepalive"]:
        alerts.append("keepalive.sh is not running")
    if h["disk_free_gb"] < 20:
        alerts.append("WSL disk low: %.1f GB free" % h["disk_free_gb"])
    if q.get("running") and h["seeds_per_hour"] == 0:
        alerts.append("a job is running but no seed finished in 30 min")
    h["alerts"] = alerts
    return h


# ------------------------------------------------------------------ training
def training():
    out = {}
    for d in sorted(glob.glob("nets/*/")):
        name = os.path.basename(d.rstrip("/"))
        hist = os.path.join(d, "train_history.json")
        if os.path.exists(hist):
            out[name] = json.load(open(hist)); continue
        log = "nets_%s_train.log" % name
        if not os.path.exists(log):
            continue
        h = {"prior": [], "value": []}
        for line in open(log):
            m = re.match(r"prior ep (\d+) loss ([\d.]+)\s+val (\{.*\})", line)
            if m:
                v = eval(m.group(3), {"__builtins__": {}})
                h["prior"].append({"epoch": int(m.group(1)), "loss": float(m.group(2)), **v})
            m = re.match(r"value ep (\d+) loss ([\d.]+)\s+val V_r2 ([-\d.]+) V_mae ([\d.]+) \(const ([\d.]+)\) hpl_mae ([\d.]+) brier ([\d.]+) \(const ([\d.]+)\) ece ([\d.]+)", line)
            if m:
                g = list(map(float, m.groups()[1:]))
                h["value"].append({"epoch": int(m.group(1)), "loss": g[0], "V_r2": g[1], "V_mae": g[2], "V_mae_const": g[3],
                                   "hpl_mae": g[4], "win_brier": g[5], "win_brier_const": g[6], "win_ece": g[7]})
        gate = os.path.join(d, "GATE.json")
        h["gate"] = json.load(open(gate)) if os.path.exists(gate) else None
        out[name] = h
    return out


# ------------------------------------------------------------------ digest
def digest(write=False, hours=24):
    now = time.time()
    c = con()
    P = {r["name"]: r for r in progress()["runs"]}
    recent = [r for r in P.values() if (r.get("t_end") or r["t"] or 0) > now - hours * 3600]
    L = ["# Daily digest %s" % time.strftime("%Y-%m-%d"), "", "Generated %s (last %d h)." % (time.strftime("%F %T %Z"), hours), ""]
    L += ["## Runs completed", "", "| run | set | n | wins | avg floor | env |", "|---|---|---|---|---|---|"]
    for r in sorted(recent, key=lambda r: r["t"]):
        L.append("| %s | %s | %d | %d | %.2f ± %.2f | `%s` |" % (r["name"], r["set"], r["n"], r["wins"], r["floor"], r["floor_ci"], (r["env"] or "")[:80]))
    pb = progress()["best_honest"]
    L += ["", "## Best honest so far", ""] + ["- %s: **%s** — %d/%d wins, floor %.2f" % (k, v["name"], v["wins"], v["n"], v["floor"]) for k, v in pb.items()]
    L += ["", "## Compares vs reference (same seeds)", ""]
    for r in recent:
        ref = REF.get(r["set"])
        if ref and r["name"] != ref and r["n"] >= 100:
            out = subprocess.run(["python3", "ledger.py", "compare", ref, r["name"]], capture_output=True, text=True).stdout
            L += ["```", "\n".join(out.strip().splitlines()[:6]), "```"]
    fails = []
    try:
        fails += ["overnight: %s (%s)" % (j["name"], j.get("note", "")) for j in json.load(open("sim_logs/overnight/state.json"))["jobs"] if j["state"] == "failed"]
    except Exception:
        pass
    try:
        fails += ["jobq #%d %s rc=%s" % (j["id"], j["name"], j.get("rc")) for j in json.load(open("sim_logs/jobs/queue.json"))["jobs"] if j["state"] == "failed"]
    except Exception:
        pass
    L += ["", "## Failures", ""] + (["- " + f for f in fails] or ["- none"])
    to = [r for r in recent if c.execute("select count(*) from seeds where run=? and timeout=1", (r["name"],)).fetchone()[0]]
    if to:
        L += ["", "Timeouts: " + ", ".join(r["name"] for r in to)]
    h = health()
    L += ["", "## Machine", "", "CPU %.0f%%, RAM %.1f/%.1f GB, GPU %s, disk free %.0f GB, %d seeds/h; alerts: %s" % (
        h["cpu_pct"], h["ram_used_gb"], h["ram_total_gb"], h.get("gpu_util"), h["disk_free_gb"], h["seeds_per_hour"],
        "; ".join(h["alerts"]) or "none")]
    md = "\n".join(L) + "\n"
    path = "sim_logs/DIGEST_%s.md" % time.strftime("%Y-%m-%d")
    if write:
        open(path, "w").write(md)
    return {"path": path, "markdown": md}


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "progress"
    if cmd == "progress":
        r = progress()
    elif cmd == "deaths":
        r = deaths(a[1], a[2] if len(a) > 2 else None)
    elif cmd == "fights":
        r = fights(a[1])
    elif cmd == "cards":
        r = cards(a[1])
    elif cmd == "seeddiff":
        r = seeddiff(int(a[1]), a[2], a[3])
    elif cmd == "health":
        r = health()
    elif cmd == "training":
        r = training()
    elif cmd == "digest":
        r = digest(write="--write" in a)
        if "--md" in a:
            print(r["markdown"]); sys.exit(0)
    elif cmd == "traced":
        r = traced()
    else:
        sys.exit(__doc__)
    print(json.dumps(r, default=float))
