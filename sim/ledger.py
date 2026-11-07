#!/usr/bin/env python3
"""Results ledger: every sim run + every seed in one SQLite file (sim_logs/results.db).

  ledger.py ingest                 (re)load all sim_logs/**/RUN.json runs; idempotent
  ledger.py list                   one line per run
  ledger.py compare A B [--json]   paired comparison on shared seeds:
                                   wins + Wilson 95% CI, exact McNemar on win flips,
                                   mean paired floor diff + bootstrap 95% CI,
                                   death-bucket diff table
  ledger.py taxonomy RUN           build_loss_taxonomy.py on that run
  ledger.py export DIR             public JSON summary (DIR/sim_runs.json) for the website

Reference (clairvoyant, seed-peeking) runs carry clairvoyant=1. Pure stdlib.
"""
import glob
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = os.path.expanduser("~/sts")
LOGS = os.path.join(ROOT, "sim_logs")
DB = os.path.join(LOGS, "results.db")
BUCKETS = ["A1_boss", "A1_elite", "A1_other", "A2_monster", "A2_elite", "A2_boss", "A2_other", "A3", "timeout", "other"]
# runs whose search replayed the TRUE seed (pre-014)
CLAIRVOYANT = {"repro_v3", "vdraft3_A", "ds_ceiling"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  name TEXT PRIMARY KEY, description TEXT, grp TEXT, env TEXT, out_dir TEXT,
  total INTEGER, seeds_done INTEGER, started_at REAL, ended_at REAL, wall_secs REAL,
  git_commit TEXT, clairvoyant INTEGER, ingested_at REAL);
CREATE TABLE IF NOT EXISTS seeds(
  run TEXT, seed INTEGER, win INTEGER, floor INTEGER, act INTEGER,
  death_act INTEGER, death_room TEXT, bucket TEXT, enemy_hp_frac REAL,
  max_hp INTEGER, gold INTEGER, relics INTEGER, deck_size INTEGER,
  act2_deck_size INTEGER, act2_upgraded INTEGER,
  potions_used INTEGER, potions_used_boss INTEGER, potions_used_elite INTEGER,
  death_enemy TEXT, secs REAL, timeout INTEGER, raw TEXT,
  PRIMARY KEY(run, seed));
"""


def bucket(r):
    a, rm = r.get("death_act"), (r.get("death_room") or "").upper()
    if r.get("win"):
        return "WIN"
    if r.get("timeout"):
        return "timeout"
    if a == 1:
        return {"BOSS": "A1_boss", "ELITE": "A1_elite"}.get(rm, "A1_other")
    if a == 2:
        return {"MONSTER": "A2_monster", "BOSS": "A2_boss", "ELITE": "A2_elite"}.get(rm, "A2_other")
    if a == 3:
        return "A3"
    return "other"


def _git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def ingest(quiet=False):
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    head = _git_head()
    n_runs = n_seeds = 0
    for p in sorted(glob.glob(os.path.join(LOGS, "**", "RUN.json"), recursive=True)):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        d = os.path.dirname(p)
        name = m.get("name")
        rows = []
        for sp in glob.glob(os.path.join(d, "seed_*.json")):
            try:
                if os.path.getsize(sp):
                    rows.append(json.load(open(sp)))
            except Exception:
                pass
        ended = None
        if os.path.exists(os.path.join(d, ".end")):
            try:
                ended = float(open(os.path.join(d, ".end")).read())
            except Exception:
                pass
        started = m.get("started_at")
        prev = con.execute("SELECT git_commit FROM runs WHERE name=?", (name,)).fetchone()
        commit = m.get("git_commit") or (prev[0] if prev and prev[0] else head)
        con.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            name, m.get("description", ""), m.get("group", ""), m.get("env", ""), os.path.relpath(d, ROOT),
            m.get("total"), len(rows), started, ended, (ended - started) if (ended and started) else None,
            commit, int(bool(m.get("clairvoyant")) or name in CLAIRVOYANT), time.time()))
        con.execute("DELETE FROM seeds WHERE run=?", (name,))
        for r in rows:
            pa = r.get("potion_audit") or []
            a2 = r.get("act2_deck") or {}
            fd = r.get("final_deck") or {}
            con.execute("INSERT OR REPLACE INTO seeds VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                name, r.get("seed"), int(r.get("win", 0)), r.get("floor"), r.get("act"),
                r.get("death_act"), r.get("death_room"), bucket(r), r.get("enemy_hp_frac"),
                r.get("max_hp"), r.get("gold"), r.get("relics"),
                fd.get("size") if isinstance(fd, dict) else None,
                a2.get("size") if isinstance(a2, dict) else None,
                a2.get("upgraded") if isinstance(a2, dict) else None,
                sum(f.get("used", 0) for f in pa) if pa else None,
                sum(f.get("used", 0) for f in pa if f.get("room") == "BOSS") if pa else None,
                sum(f.get("used", 0) for f in pa if f.get("room") == "ELITE") if pa else None,
                (pa[-1].get("enc") if pa and pa[-1].get("died") else r.get("death_enemy")),
                r.get("secs"), int(r.get("timeout", 0)), json.dumps(r)))
        n_runs += 1
        n_seeds += len(rows)
    con.commit()
    if not quiet:
        print("ingested %d runs, %d seeds -> %s" % (n_runs, n_seeds, DB))
    return con


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """two-sided exact binomial test on discordant pairs (b: A wins & B loses, c: reverse)"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_mean(xs, iters=5000, seed=12345):
    rnd = random.Random(seed)
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    ms = sorted(sum(xs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1])


def compare(con, a, b):
    q = ("SELECT s1.seed, s1.win, s2.win, s1.floor, s2.floor, s1.bucket, s2.bucket FROM seeds s1 "
         "JOIN seeds s2 ON s1.seed=s2.seed WHERE s1.run=? AND s2.run=?")
    rows = con.execute(q, (a, b)).fetchall()
    n = len(rows)
    wa = sum(r[1] for r in rows)
    wb = sum(r[2] for r in rows)
    only_a = sum(1 for r in rows if r[1] and not r[2])
    only_b = sum(1 for r in rows if r[2] and not r[1])
    diffs = [r[4] - r[3] for r in rows]
    md = sum(diffs) / n if n else 0.0
    lo, hi = bootstrap_mean(diffs)
    ba = {k: 0 for k in BUCKETS + ["WIN"]}
    bb = dict(ba)
    for r in rows:
        ba[r[5]] = ba.get(r[5], 0) + 1
        bb[r[6]] = bb.get(r[6], 0) + 1
    meta = {r[0]: r for r in con.execute("SELECT name, clairvoyant, env, description FROM runs WHERE name IN (?,?)", (a, b))}
    return {
        "a": a, "b": b, "shared_seeds": n,
        "a_clairvoyant": bool(meta.get(a, (0, 0))[1]), "b_clairvoyant": bool(meta.get(b, (0, 0))[1]),
        "wins_a": wa, "wins_b": wb, "wilson_a": wilson(wa, n), "wilson_b": wilson(wb, n),
        "win_flips_a_only": only_a, "win_flips_b_only": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
        "floor_a": sum(r[3] for r in rows) / n if n else 0, "floor_b": sum(r[4] for r in rows) / n if n else 0,
        "floor_diff_b_minus_a": md, "floor_diff_ci95": (lo, hi),
        "buckets": [(k, ba.get(k, 0), bb.get(k, 0)) for k in ["WIN"] + BUCKETS if ba.get(k, 0) or bb.get(k, 0)],
    }


def fmt_compare(c):
    L = []
    tag = lambda x: " (clairvoyant)" if x else ""
    L.append("A = %s%s\nB = %s%s\nshared seeds: %d" % (c["a"], tag(c["a_clairvoyant"]), c["b"], tag(c["b_clairvoyant"]),
                                                     c["shared_seeds"]))
    n = max(1, c["shared_seeds"])
    L.append("wins   A %3d (%.1f%%, 95%% CI %.1f-%.1f%%)   B %3d (%.1f%%, 95%% CI %.1f-%.1f%%)" % (
        c["wins_a"], 100 * c["wins_a"] / n, 100 * c["wilson_a"][0], 100 * c["wilson_a"][1],
        c["wins_b"], 100 * c["wins_b"] / n, 100 * c["wilson_b"][0], 100 * c["wilson_b"][1]))
    L.append("win flips: A-only %d, B-only %d  -> exact McNemar p = %.3f" % (
        c["win_flips_a_only"], c["win_flips_b_only"], c["mcnemar_p"]))
    L.append("avg floor A %.2f  B %.2f   paired diff B-A = %+.2f  (bootstrap 95%% CI %+.2f .. %+.2f)%s" % (
        c["floor_a"], c["floor_b"], c["floor_diff_b_minus_a"], c["floor_diff_ci95"][0], c["floor_diff_ci95"][1],
        "  *" if (c["floor_diff_ci95"][0] > 0 or c["floor_diff_ci95"][1] < 0) else ""))
    L.append("\n%-12s %5s %5s %6s" % ("outcome", "A", "B", "B-A"))
    for k, x, y in c["buckets"]:
        L.append("%-12s %5d %5d %+6d" % (k, x, y, y - x))
    return "\n".join(L)


def taxonomy(con, run):
    rows = con.execute("SELECT raw FROM seeds WHERE run=?", (run,)).fetchall()
    if not rows:
        print("no such run / no seeds:", run)
        return
    t = tempfile.mkdtemp()
    os.makedirs(os.path.join(t, "sim_logs"))
    jl = os.path.join(t, "run.jsonl")
    with open(jl, "w") as f:
        for (raw,) in rows:
            f.write(raw + "\n")
    subprocess.run([os.path.join(ROOT, ".venv/bin/python"), os.path.join(ROOT, "build_loss_taxonomy.py"), jl],
                   cwd=t, capture_output=True)
    print(open(os.path.join(t, "sim_logs", "loss_taxonomy.txt")).read().replace("(mainline_v3, 150 fixed seeds)",
                                                                                "(%s)" % run))


def export(con, dst):
    out = []
    for r in con.execute("SELECT name, description, grp, env, seeds_done, total, wall_secs, clairvoyant FROM runs "
                         "ORDER BY COALESCE(started_at, 0)"):
        name = r[0]
        s = con.execute("SELECT COUNT(*), SUM(win), AVG(floor), SUM(act>=2), SUM(act>=3), AVG(secs) FROM seeds WHERE run=?",
                        (name,)).fetchone()
        if not s[0] or (r[5] and s[0] < r[5]):
            continue
        dc = dict(con.execute("SELECT bucket, COUNT(*) FROM seeds WHERE run=? GROUP BY bucket", (name,)).fetchall())
        out.append({"name": name, "description": r[1], "group": r[2], "env": r[3], "seeds": s[0],
                    "wins": s[1], "win_rate_pct": round(100.0 * s[1] / s[0], 2), "avg_floor": round(s[2], 2),
                    "reached_act2": s[3], "reached_act3": s[4],
                    "secs_per_seed_mean": round(s[5], 1) if s[5] else None,
                    "wall_secs": int(r[6]) if r[6] else None, "clairvoyant": bool(r[7]),
                    "deaths": {k: dc[k] for k in BUCKETS if dc.get(k)}})
    os.makedirs(dst, exist_ok=True)
    json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M"), "eval_seeds": "10007 + i*7919",
               "note": "Headless sts_lightspeed sim, Ironclad A0. clairvoyant = search replayed the true seed "
                       "(saw future draws/rolls); honest = search samples hidden information.",
               "bucket_order": BUCKETS, "runs": out}, open(os.path.join(dst, "sim_runs.json"), "w"), indent=1)
    print("exported %d runs -> %s" % (len(out), os.path.join(dst, "sim_runs.json")))


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return
    if a[0] == "ingest":
        ingest()
        return
    con = sqlite3.connect(DB) if os.path.exists(DB) else ingest(quiet=True)
    if a[0] == "list":
        for r in con.execute("SELECT r.name, r.clairvoyant, COUNT(s.seed), SUM(s.win), AVG(s.floor), r.env FROM runs r "
                             "LEFT JOIN seeds s ON s.run=r.name GROUP BY r.name ORDER BY COALESCE(r.started_at,0)"):
            print("%-26s %-3s n=%-4d wins=%-3s floor=%-6s %s" % (r[0], "CV" if r[1] else "", r[2], r[3],
                                                                 "%.2f" % r[4] if r[4] else "--", r[5] or ""))
    elif a[0] == "compare" and len(a) >= 3:
        c = compare(con, a[1], a[2])
        print(json.dumps(c) if "--json" in a else fmt_compare(c))
    elif a[0] == "taxonomy" and len(a) >= 2:
        taxonomy(con, a[1])
    elif a[0] == "export" and len(a) >= 2:
        export(con, a[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
