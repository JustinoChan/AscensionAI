#!/usr/bin/env python3
"""Seed inspector: replay one seed of a run with TRACE=1 and write a
self-contained HTML report (sim_logs/inspect/<run>_<seed>.html).

  inspect_seed.py RUN SEED [--open-path]

The seed is re-run with the run's own toggles (RUN.json env) plus TRACE=1
(passive), then checked against the stored seed JSON: the report says whether
the replayed trajectory is identical to the recorded one.
Shows: HP/floor timeline, path, deck + relics at each act, potions gained/used,
every fight (enemy, turns, HP lost, potions) and the death fight.
"""
import glob
import html
import json
import os
import re
import subprocess
import sys

ROOT = os.path.expanduser("~/sts")
LOGS = os.path.join(ROOT, "sim_logs")


def potion_names():
    try:
        src = open(os.path.join(ROOT, "include/constants/Potions.h")).read()
        body = src[src.index("enum class Potion"):]
        body = body[body.index("{") + 1: body.index("}")]
        names = [re.sub(r"=.*", "", x).strip() for x in body.split(",")]
        return [n for n in names if n]
    except Exception:
        return []


PN = potion_names()


def pname(v):
    v = int(v)
    return PN[v].replace("_", " ").title() if 0 <= v < len(PN) else "potion#%d" % v


def find_run(run):
    for p in glob.glob(os.path.join(LOGS, "**", "RUN.json"), recursive=True):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        if m.get("name") == run:
            return m, os.path.dirname(p)
    raise SystemExit("no run named %r (see ledger.py list)" % run)


def replay(meta, seed):
    env = dict(os.environ)
    for kv in (meta.get("env") or "").split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    env["TRACE"] = "1"
    p = subprocess.run(["nice", "-n", "10", os.path.join(ROOT, ".venv/bin/python"), "eval_fullrun_analytics_v2.py",
                        str(seed), "macro"], cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise SystemExit("replay failed:\n" + p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def svg_hp(floors, fights, death_floor):
    if not floors:
        return ""
    W, H, P = 900, 260, 36
    maxf = max(f["floor"] for f in floors) or 1
    maxhp = max(max(f["max_hp"] for f in floors), 1)
    X = lambda fl: P + (W - 2 * P) * fl / max(maxf, 1)
    Y = lambda hp: H - P - (H - 2 * P) * hp / maxhp
    pts = " ".join("%.1f,%.1f" % (X(f["floor"]), Y(f["hp"])) for f in floors)
    mpts = " ".join("%.1f,%.1f" % (X(f["floor"]), Y(f["max_hp"])) for f in floors)
    col = {"ELITE": "#e0a100", "BOSS": "#d33", "MONSTER": "#4a7", "EVENT": "#58c", "REST": "#a6a", "SHOP": "#999",
           "TREASURE": "#c90"}
    dots = "".join('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"><title>floor %d %s hp %d/%d</title></circle>' % (
        X(f["floor"]), Y(f["hp"]), col.get(f["room"], "#777"), f["floor"], f["room"], f["hp"], f["max_hp"]) for f in floors)
    acts = "".join('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#bbb" stroke-dasharray="4 3"/>'
                   '<text x="%.1f" y="%d" font-size="11" fill="#888">Act %d</text>' % (
                       X(fl), P - 8, X(fl), H - P, X(fl) + 3, P - 12, a) for a, fl in ((2, 17), (3, 34), (4, 51)) if fl <= maxf)
    death = ""
    if death_floor is not None:
        death = '<text x="%.1f" y="%.1f" font-size="16" fill="#d33">&#10006;</text>' % (X(death_floor) - 5, Y(0) - 2)
    axis = ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#444"/><line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#444"/>'
            % (P, H - P, W - P, H - P, P, P, P, H - P))
    ticks = "".join('<text x="%.1f" y="%d" font-size="10" text-anchor="middle" fill="#666">%d</text>' % (X(t), H - P + 14, t)
                    for t in range(0, maxf + 1, 5))
    yt = "".join('<text x="%d" y="%.1f" font-size="10" text-anchor="end" fill="#666">%d</text>' % (P - 4, Y(t) + 3, t)
                 for t in range(0, maxhp + 1, 20))
    return ('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="HP by floor">%s%s%s%s'
            '<polyline points="%s" fill="none" stroke="#bbb" stroke-dasharray="3 3"/>'
            '<polyline points="%s" fill="none" stroke="#2b6cb0" stroke-width="2"/>%s%s'
            '<text x="%d" y="%d" font-size="11" fill="#666">floor</text></svg>' % (
                W, H, axis, ticks, yt, acts, mpts, pts, dots, death, W - P - 20, H - 6))


def render(run, meta, seed, out, stored, same):
    T = out.get("trace", {})
    floors, fights, acts, events = T.get("floors", []), T.get("fights", []), T.get("acts", []), T.get("events", [])
    death = next((f for f in fights if f.get("died")), None)
    e = html.escape
    tag = '<span class="cv">clairvoyant (seed-peeking)</span>' if meta.get("clairvoyant") or run in (
        "repro_v3", "vdraft3_A", "ds_ceiling") else '<span class="hon">honest</span>'
    res = "WIN" if out.get("win") else "died on floor %d (act %d, %s)" % (out.get("floor"), out.get("act"),
                                                                          out.get("death_room", "?"))
    rows = []
    for f in fights:
        rows.append("<tr%s><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%d/%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            ' class="death"' if f.get("died") else (' class="boss"' if f["room"] == "BOSS" else ""),
            f["floor"], f["act"], e(f["room"]), e(f["enc"].replace("_", " ").title()), f["hp_start"], f["max_hp"],
            f.get("turns", ""), "died" if f.get("died") else "-%d" % f.get("hp_lost", 0),
            e(", ".join(pname(p) for p in f.get("potions_start", []))) or "&ndash;",
            e(", ".join(pname(p) for p in f.get("potions_used", []))) or "&ndash;"))
    path = " &rarr; ".join('<span class="r %s" title="floor %d hp %d/%d">%s</span>' % (
        f["room"].lower(), f["floor"], f["hp"], f["max_hp"], {"MONSTER": "M", "ELITE": "E", "BOSS": "B", "EVENT": "?",
                                                             "REST": "R", "SHOP": "$", "TREASURE": "T"}.get(f["room"], f["room"][:1]))
        for f in floors if f["floor"] > 0)
    actblocks = []
    for a in acts:
        deck = sorted(a["deck"])
        actblocks.append("<div class='act'><h3>Act %d (floor %d, hp %d/%d)</h3><p><b>Deck (%d):</b> %s</p><p><b>Relics:</b> %s</p></div>" % (
            a["act"], a["floor"], a["hp"], a["max_hp"], len(deck), e(", ".join(c.replace("_", " ").title() for c in deck)),
            e(", ".join(r.replace("_", " ").title() for r in a["relics"]))))
    ev = "".join("<li>floor %d: %s %s</li>" % (x["floor"], {"card+": "took card", "card-": "removed/transformed",
                                                         "relic+": "relic", "relic-": "lost relic", "potion+": "potion"}.get(x["kind"], x["kind"]),
                                                e(pname(x["what"]) if x["kind"].startswith("potion") else str(x["what"]).replace("_", " ").title()))
                 for x in events)
    dsec = ""
    if death:
        dsec = "<div class='deathbox'><h2>Death fight</h2><p>Floor %d, act %d, %s vs <b>%s</b>: entered at %d/%d HP, lasted %s turns; potions held %s, used %s. Enemy HP left: %s.</p></div>" % (
            death["floor"], death["act"], e(death["room"]), e(death["enc"].replace("_", " ").title()), death["hp_start"],
            death["max_hp"], death.get("turns"), e(", ".join(pname(p) for p in death.get("potions_start", [])) or "none"),
            e(", ".join(pname(p) for p in death.get("potions_used", [])) or "none"),
            "%.0f%%" % (100 * out.get("enemy_hp_frac", 0)) if out.get("enemy_hp_frac") is not None else "?")
    check = ("<span class='ok'>replay identical to the recorded run</span>" if same else
             ("<span class='bad'>replay DIFFERS from the recorded run</span>" if stored is not None else "no recorded seed to check"))
    return """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seed {seed} - {run}</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--card:#f6f7f9;--bd:#e3e5e8}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#a0a0a0;--card:#1f2228;--bd:#30343b}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0 auto;max-width:1000px;padding:16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:22px 0 8px}} h3{{font-size:14px;margin:8px 0 4px}}
.mut{{color:var(--mut)}} .card{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;margin:10px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--bd);padding:4px 6px;text-align:left}}
tr.boss td{{background:rgba(220,60,60,.08)}} tr.death td{{background:rgba(220,60,60,.22);font-weight:600}}
.r{{display:inline-block;min-width:16px;text-align:center;border-radius:4px;padding:0 3px;font-weight:600}}
.monster{{background:#cfe9d6;color:#174}} .elite{{background:#ffe7a6;color:#7a5800}} .boss{{background:#f7c2c2;color:#8a1010}}
.event{{background:#d4e3f7;color:#1c4a85}} .rest{{background:#ead6f2;color:#5b2b72}} .shop{{background:#e2e2e2;color:#333}} .treasure{{background:#f5e0b0;color:#6a4b00}}
.cv{{background:#ffe7a6;color:#7a5800;border-radius:4px;padding:1px 6px}} .hon{{background:#cfe9d6;color:#174;border-radius:4px;padding:1px 6px}}
.ok{{color:#2e7d32}} .bad{{color:#c62828;font-weight:600}} .deathbox{{border-left:4px solid #d33;padding:4px 12px;background:var(--card)}}
.wrap{{overflow-x:auto}} ul{{columns:2;font-size:13px}} @media(max-width:640px){{ul{{columns:1}}}}
</style></head><body>
<h1>Seed {seed} &mdash; {run} {tag}</h1>
<p class="mut">{desc}<br>toggles: <code>{env}</code><br>result: <b>{res}</b> &middot; max HP {mhp}, gold {gold}, relics {nrel} &middot; {check}</p>
<div class="card">{svg}</div>
<h2>Path</h2><div class="card">{path}</div>
{dsec}
<h2>Fights ({nf})</h2><div class="wrap"><table><tr><th>Floor</th><th>Act</th><th>Room</th><th>Enemy</th><th>HP in</th><th>Turns</th><th>HP</th><th>Potions held</th><th>Potions used</th></tr>{rows}</table></div>
<h2>Deck and relics at each act</h2>{acts}
<h2>Pickups</h2><ul>{ev}</ul>
<p class="mut">Generated by inspect_seed.py from a TRACE=1 replay. Eval seed set: 10007 + i&middot;7919.</p>
</body></html>""".format(seed=seed, run=e(run), tag=tag, desc=e(meta.get("description", "")), env=e(meta.get("env") or "(defaults)"),
                         res=e(res), mhp=out.get("max_hp"), gold=out.get("gold"), nrel=out.get("relics"), check=check,
                         svg=svg_hp(floors, fights, death["floor"] if death else None), path=path, dsec=dsec,
                         nf=len(fights), rows="".join(rows), acts="".join(actblocks), ev=ev)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    run, seed = sys.argv[1], int(sys.argv[2])
    meta, d = find_run(run)
    out = replay(meta, seed)
    sp = os.path.join(d, "seed_%d.json" % seed)
    stored = json.load(open(sp)) if os.path.exists(sp) and os.path.getsize(sp) else None
    same = False
    if stored is not None:
        strip = lambda x: {k: v for k, v in x.items() if k not in ("secs", "trace", "potion_audit")}
        same = strip(out) == strip(stored)
    os.makedirs(os.path.join(LOGS, "inspect"), exist_ok=True)
    dst = os.path.join(LOGS, "inspect", "%s_%d.html" % (run, seed))
    open(dst, "w").write(render(run, meta, seed, out, stored, same))
    print(json.dumps({"html": dst, "identical_to_recorded": same, "floor": out.get("floor"), "win": out.get("win")}))


if __name__ == "__main__":
    main()
