#!/usr/bin/env python3
"""Status of sim experiment runs (feeds the AscensionAI control panel).

Convention: every runner writes  sim_logs/<run>/RUN.json  at launch
  {name, description, total, parallel, env, started_at, pgid, log, out_dir}
per-seed results land in  <out_dir>/seed_<s>.json  (a fixed-size 'timeout'
stub on kill), optional live heartbeats in  <out_dir>/seed_<s>.prog
("floor epoch"), and SUMMARY.txt when the run finishes.
Queues: sim_logs/queues/<q>.json  {name, runs:[...run names...], est_secs:{run: s}}
Phases: sim_logs/PHASES.json  [{phase, steps:[{name, state}]}]

usage: run_status.py [--json]
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.expanduser("~/sts")
LOGS = os.path.join(ROOT, "sim_logs")
WINDOW = 20            # ETA from throughput over the last N completed seeds
EXPECT_FLOOR = 30.0    # heartbeat partial credit: floor / EXPECT_FLOOR (capped)


def _alive_pgid(pgid):
    if not pgid:
        return False
    try:
        os.killpg(int(pgid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        return False


def _load(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def run_status(meta, now):
    out_dir = os.path.join(ROOT, meta.get("out_dir", ""))
    total = int(meta.get("total", 0))
    rows, mtimes = [], []
    for p in glob.glob(os.path.join(out_dir, "seed_*.json")):
        try:
            if os.path.getsize(p) == 0:
                continue
            r = _load(p)
            if r is None:
                continue
            rows.append(r)
            mtimes.append(os.path.getmtime(p))
        except OSError:
            continue
    done = len(rows)
    wins = sum(int(r.get("win", 0)) for r in rows)
    timeouts = sum(int(r.get("timeout", 0)) for r in rows)
    real = [r for r in rows if not r.get("timeout")]
    avg_floor = (sum(r.get("floor", 0) for r in rows) / done) if done else None
    secs = [r["secs"] for r in real if "secs" in r]

    # partial credit from live heartbeats (seeds in flight)
    partial = 0.0
    inflight = []
    for p in glob.glob(os.path.join(out_dir, "seed_*.prog")):
        if os.path.exists(p[:-5] + ".json") and os.path.getsize(p[:-5] + ".json") > 0:
            continue
        try:
            fl = float(open(p).read().split()[0])
            inflight.append(int(fl))
            partial += min(0.95, fl / EXPECT_FLOOR)
        except Exception:
            pass

    started = float(meta.get("started_at", 0) or 0)
    ended = None
    endf = os.path.join(out_dir, ".end")
    if os.path.exists(endf):
        try:
            ended = float(open(endf).read().strip())
        except Exception:
            ended = os.path.getmtime(endf)
    summary = os.path.exists(os.path.join(out_dir, "SUMMARY.txt"))
    alive = _alive_pgid(meta.get("pgid"))
    if meta.get("state") == "queued" and not started:
        state = "queued"
    elif done >= total and total > 0 and (summary or ended):
        state = "done"
    elif alive:
        state = "running"
    elif done >= total and total > 0:
        state = "done"
    elif meta.get("stopped"):
        state = "stopped"
    else:
        state = "failed" if done == 0 else "stopped"

    elapsed = ((ended or now) - started) if started else 0.0
    eta = None
    rate = None
    if state == "running":
        remaining = max(0.0, total - done - partial)
        mt = sorted(mtimes)
        win = mt[-WINDOW:]
        if len(win) >= 3 and win[-1] > win[0]:
            rate = (len(win) - 1) / (win[-1] - win[0])          # seeds/s, recent
            # if nothing finished for a while, don't over-promise
            since = now - win[-1]
            if since > 3.0 / rate:
                rate = min(rate, len(win) / (now - win[0]))
        elif done + partial > 0 and elapsed > 0:
            rate = (done + partial) / elapsed
        # bursty producers (RL: games come in batches with update / validation pauses): never let a
        # momentary lull push the estimate far below the whole-run average
        if done + partial > 50 and elapsed > 0:
            overall = (done + partial) / elapsed
            if rate is None or rate < overall / 3:
                rate = overall
        if rate:
            eta = remaining / rate
    pct = 100.0 * (done + (partial if state == "running" else 0)) / total if total else 0.0
    return {
        "name": meta.get("name"), "description": meta.get("description", ""),
        "group": meta.get("group", ""), "env": meta.get("env", ""),
        "total": total, "done": done, "pct": round(min(100.0, pct), 1),
        "wins": wins, "avg_floor": round(avg_floor, 2) if avg_floor is not None else None,
        "timeouts": timeouts, "state": state, "started_at": started or None,
        "elapsed": round(elapsed, 1), "eta": round(eta, 1) if eta is not None else None,
        "rate_per_min": round(rate * 60, 2) if rate else None,
        "secs_per_seed_median": round(sorted(secs)[len(secs) // 2], 1) if secs else None,
        "inflight_floors": sorted(inflight), "parallel": meta.get("parallel"),
        "log": meta.get("log", ""), "out_dir": meta.get("out_dir", ""), "pgid": meta.get("pgid"),
    }


def main():
    now = time.time()
    runs = []
    for p in sorted(glob.glob(os.path.join(LOGS, "**", "RUN.json"), recursive=True)):
        meta = _load(p)
        if meta:
            meta.setdefault("out_dir", os.path.relpath(os.path.dirname(p), ROOT))
            runs.append(run_status(meta, now))
    runs.sort(key=lambda r: (r["started_at"] or 9e18))
    by = {r["name"]: r for r in runs}

    queues = []
    for p in sorted(glob.glob(os.path.join(LOGS, "queues", "*.json"))):
        q = _load(p)
        if not q:
            continue
        tot = dn = 0
        eta = 0.0
        unknown = False
        states = []
        for name in q.get("runs", []):
            r = by.get(name)
            est = (q.get("est_secs") or {}).get(name)
            if r is None:
                states.append("queued")
                tot += int((q.get("totals") or {}).get(name, 0))
                if est is not None:
                    eta += est
                else:
                    unknown = True
                continue
            states.append(r["state"])
            tot += r["total"]
            dn += r["done"]
            if r["state"] == "running":
                if r["eta"] is not None:
                    eta += r["eta"]
                elif est is not None:
                    eta += max(0.0, est - r["elapsed"])
                else:
                    unknown = True
            elif r["state"] == "queued":
                if est is not None:
                    eta += est
                else:
                    unknown = True
        if all(s == "done" for s in states):
            st = "done"
        elif any(s == "running" for s in states):
            st = "running"
        elif any(s in ("failed", "stopped") for s in states):
            st = "stopped"
        else:
            st = "queued"
        queues.append({"name": q.get("name", os.path.basename(p)[:-5]), "runs": q.get("runs", []),
                       "total": tot, "done": dn, "pct": round(100.0 * dn / tot, 1) if tot else 0.0,
                       "state": st, "eta": None if (unknown or st == "done") else round(eta, 1),
                       "eta_partial": round(eta, 1) if unknown else None})

    phases = _load(os.path.join(LOGS, "PHASES.json"), [])
    res = {"now": now, "runs": runs, "queues": queues, "phases": phases}
    if "--json" in sys.argv:
        print(json.dumps(res))
        return

    def fmt(s):
        if s is None:
            return "--"
        s = int(s)
        return "%dh%02dm" % (s // 3600, (s % 3600) // 60) if s >= 3600 else "%dm%02ds" % (s // 60, s % 60)
    for r in runs:
        print("%-22s %-8s %4d/%-4d %5.1f%%  wins %-3d floor %-6s  elapsed %-8s ETA %s" % (
            r["name"], r["state"], r["done"], r["total"], r["pct"], r["wins"],
            r["avg_floor"] if r["avg_floor"] is not None else "--", fmt(r["elapsed"]), fmt(r["eta"])))
    for q in queues:
        print("QUEUE %-16s %-8s %4d/%-4d %5.1f%%  ETA %s" % (q["name"], q["state"], q["done"], q["total"], q["pct"], fmt(q["eta"])))


if __name__ == "__main__":
    main()
