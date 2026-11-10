"""Analytics tab + system-health strip for the AscensionAI control panel.

All numbers come from ~/sts/dash_data.py in WSL (JSON over `wsl.exe`), fetched on a
background thread; the UI thread only draws. Charts use matplotlib (TkAgg).
Panels: Progress, Where we die, Fights, Cards, Seed diff, Training, Digest.
"""
import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import matplotlib
except ImportError:          # launched without the per-user site dir on sys.path (e.g. from a service/agent env)
    import os
    import sys
    # project-local install first (pip install --target .pylibs matplotlib), then the per-user site dir
    for _p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pylibs"),
               os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python",
                            "Python%d%d" % sys.version_info[:2], "site-packages")):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.append(_p)
    import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

WSL = ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc"]
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ROOM_COL = {"MONSTER": "#90a4ae", "ELITE": "#ef6c00", "BOSS": "#c62828", "EVENT": "#7e57c2", "?": "#bdbdbd"}
SET_MARK = {"eval": "o", "holdout": "s", "training": "^", "other": "x"}


def dash(cmd, timeout=900):
    p = subprocess.run(WSL + ["cd ~/sts && nice -n 10 .venv/bin/python dash_data.py " + cmd], capture_output=True,
                       text=True, timeout=timeout, creationflags=_NOWIN, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError((p.stderr or "dash_data failed").strip()[-600:])
    out = p.stdout.strip()
    return out if cmd.startswith("digest") and "--md" in cmd else json.loads(out)


def _q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


class _Panel:
    """Base: a frame with a toolbar, a figure and/or a table, and async loading."""

    def __init__(self, parent, app, title):
        self.app = app
        self.frame = ttk.Frame(parent, padding=6)
        parent.add(self.frame, text="  %s  " % title)
        self.bar = ttk.Frame(self.frame)
        self.bar.pack(fill="x")
        self.status = tk.StringVar(value="")
        ttk.Label(self.bar, textvariable=self.status, foreground="gray").pack(side="right")

    def fig(self, h=4.2):
        f = Figure(figsize=(9, h), dpi=96, tight_layout=True)
        c = FigureCanvasTkAgg(f, master=self.frame)
        c.get_tk_widget().pack(fill="both", expand=True)
        return f, c

    def table(self, cols, widths, height=10):
        fr = ttk.Frame(self.frame)
        fr.pack(fill="both", expand=True, pady=(4, 0))
        t = ttk.Treeview(fr, columns=cols, show="headings", height=height)
        for c, w in zip(cols, widths):
            t.heading(c, text=c, command=lambda c=c, t=t: _sort(t, c))
            t.column(c, width=w, anchor="w" if w > 120 else "e")
        sb = ttk.Scrollbar(fr, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        t.pack(side="left", fill="both", expand=True)
        return t

    def load(self, cmd, cb, timeout=900):
        self.status.set("loading %s ..." % cmd.split()[0])
        t0 = time.time()

        def go():
            try:
                r = dash(cmd, timeout)
                self.app.post(lambda: (self.status.set("updated %s (%.1fs)" % (time.strftime("%H:%M:%S"), time.time() - t0)), cb(r)))
            except Exception as e:  # noqa: BLE001
                self.app.post(lambda: self.status.set("error: %s" % str(e)[-160:]))
        threading.Thread(target=go, daemon=True).start()


def _sort(t, col):
    rows = [(t.set(k, col), k) for k in t.get_children("")]
    def key(v):
        try:
            return (0, float(v[0]))
        except ValueError:
            return (1, v[0])
    rev = getattr(t, "_rev", {}).get(col, False)
    rows.sort(key=key, reverse=not rev)
    for i, (_, k) in enumerate(rows):
        t.move(k, "", i)
    t._rev = {col: not rev}


def _fill(t, rows):
    t.delete(*t.get_children())
    for r in rows:
        t.insert("", "end", values=r)


def _f(x, d=2):
    return "" if x is None else ("%.*f" % (d, x))


# ------------------------------------------------------------------ panels
class ProgressPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Progress")
        ttk.Button(self.bar, text="Refresh", command=self.refresh).pack(side="left")
        self.show_cv = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.bar, text="show clairvoyant", variable=self.show_cv, command=self._draw).pack(side="left", padx=6)
        self.head = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.head, font=("Segoe UI", 11, "bold"), foreground="#2e7d32").pack(anchor="w", pady=(4, 0))
        self.f, self.c = self.fig(4.6)
        self.t = self.table(("run", "set", "n", "wins", "win %", "win CI", "avg floor", "± CI", "kind"),
                            (220, 70, 50, 50, 60, 100, 75, 60, 90), 8)
        self.data = None

    def refresh(self):
        self.load("progress", self._got)

    def _got(self, d):
        self.data = d
        b = d.get("best_honest", {})
        self.head.set("Best honest so far —  " + "   |   ".join(
            "%s: %s  %d/%d wins (%.1f%%), floor %.2f" % (k, v["name"], v["wins"], v["n"], v["win_pct"], v["floor"]) for k, v in sorted(b.items())))
        _fill(self.t, [(r["name"], r["set"], r["n"], r["wins"], _f(r["win_pct"], 1), "%.1f-%.1f" % (r["win_lo"], r["win_hi"]),
                        _f(r["floor"]), _f(r["floor_ci"]), "clairvoyant" if r["clairvoyant"] else "honest") for r in d["runs"]])
        self._draw()

    def _draw(self):
        if not self.data:
            return
        self.f.clear()
        a1, a2 = self.f.add_subplot(121), self.f.add_subplot(122)
        runs = [r for r in self.data["runs"] if r["n"] >= 50 and (self.show_cv.get() or not r["clairvoyant"])]
        for i, r in enumerate(runs):
            col = "#c62828" if r["clairvoyant"] else "#1565c0"
            m = SET_MARK.get(r["set"], "x")
            a1.errorbar(i, r["floor"], yerr=r["floor_ci"], fmt=m, color=col, ms=5, capsize=2)
            a2.errorbar(i, r["win_pct"], yerr=[[r["win_pct"] - r["win_lo"]], [r["win_hi"] - r["win_pct"]]], fmt=m, color=col, ms=5, capsize=2)
        for a, yl in ((a1, "avg floor (95% CI)"), (a2, "win % (Wilson 95% CI)")):
            a.set_xticks(range(len(runs)))
            a.set_xticklabels([r["name"] for r in runs], rotation=75, fontsize=6)
            a.set_ylabel(yl); a.grid(alpha=.3)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color="#1565c0", marker="o", ls="", label="honest eval-150"),
             Line2D([], [], color="#1565c0", marker="s", ls="", label="honest holdout-150"),
             Line2D([], [], color="#c62828", marker="o", ls="", label="clairvoyant"),
             Line2D([], [], color="gray", marker="^", ls="", label="training seeds")]
        a1.legend(handles=h, fontsize=7, loc="lower right")
        a1.set_title("Every run in run order", fontsize=9)
        self.c.draw_idle()


class RunPicker:
    def __init__(self, panel, two=False, label_b="vs"):
        self.a, self.b = tk.StringVar(), tk.StringVar()
        ttk.Label(panel.bar, text="run:").pack(side="left")
        self.box_a = ttk.Combobox(panel.bar, textvariable=self.a, width=26, state="readonly")
        self.box_a.pack(side="left", padx=(2, 6))
        self.box_b = None
        if two:
            ttk.Label(panel.bar, text=label_b).pack(side="left")
            self.box_b = ttk.Combobox(panel.bar, textvariable=self.b, width=26, state="readonly")
            self.box_b.pack(side="left", padx=(2, 6))
        panel.app.pickers.append(self)

    def set_runs(self, names, traced):
        vals = [n + ("  [traced]" if n in traced else "") for n in names]
        self.box_a.configure(values=vals)
        if self.box_b is not None:
            self.box_b.configure(values=[""] + vals)
        if not self.a.get() and vals:
            pref = [v for v in vals if "[traced]" in v]
            self.a.set(pref[0] if pref else vals[-1])

    @staticmethod
    def clean(v):
        return v.split()[0] if v else ""


class DeathsPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Where we die")
        self.p = RunPicker(self, two=True, label_b="diff vs:")
        ttk.Button(self.bar, text="Show", command=self.refresh).pack(side="left")
        self.f, self.c = self.fig(3.4)
        self.note = tk.StringVar(value="Encounter tables need traced runs ([traced]); floor histogram works for every run.")
        ttk.Label(self.frame, textvariable=self.note, foreground="gray").pack(anchor="w")
        self.t = self.table(("act", "encounter", "room", "met", "deaths", "win rate", "mean HP lost", "deaths B", "met B", "win B"),
                            (40, 200, 80, 50, 55, 70, 90, 65, 55, 60), 9)

    def refresh(self):
        a, b = RunPicker.clean(self.p.a.get()), RunPicker.clean(self.p.b.get())
        if a:
            self.load("deaths %s %s" % (_q(a), _q(b) if b else ""), self._got)

    def _got(self, d):
        self.f.clear()
        ax = self.f.add_subplot(111)
        A = d["a"]
        rooms = sorted({h["room"] for h in A["hist"]})
        floors = list(range(1, 56))
        bottom = [0] * len(floors)
        for rm in rooms:
            v = [sum(h["n"] for h in A["hist"] if h["floor"] == f and h["room"] == rm) for f in floors]
            ax.bar(floors, v, bottom=bottom, color=ROOM_COL.get(rm, "#bdbdbd"), label=rm, width=0.85)
            bottom = [x + y for x, y in zip(bottom, v)]
        if "b" in d:
            B = d["b"]
            vb = [sum(h["n"] for h in B["hist"] if h["floor"] == f) for f in floors]
            ax.step(floors, vb, where="mid", color="black", lw=1.2, label="%s (total)" % B["run"])
        for x in (16.5, 33.5):
            ax.axvline(x, color="gray", ls=":", lw=1)
        ax.set_xlabel("floor of death"); ax.set_ylabel("runs")
        ax.set_title("%s: %d runs, %d wins — deaths by floor and room" % (A["run"], A["n"], A["wins"]), fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
        self.c.draw_idle()
        rows = []
        if "enc_diff" in d:
            ka = {(e["act"], e["enc"]): e for e in A["encounters"]}
            for x in d["enc_diff"]:
                e = ka.get((x["act"], x["enc"]), {})
                rows.append((x["act"], x["enc"], e.get("room", ""), x["met_a"], x["deaths_a"], _f(x["win_a"]), _f(e.get("mean_hp_lost"), 1),
                             x["deaths_b"], x["met_b"], _f(x["win_b"])))
        else:
            rows = [(e["act"], e["enc"], e["room"], e["met"], e["deaths"], _f(e["win_rate"]), _f(e["mean_hp_lost"], 1), "", "", "")
                    for e in A["encounters"]]
        _fill(self.t, rows)
        self.note.set("%s: %d traced seeds%s. Buckets: %s" % (A["run"], A["traced_seeds"],
                      (" | %s: %d traced" % (d["b"]["run"], d["b"]["traced_seeds"])) if "b" in d else "",
                      ", ".join("%s %d" % kv for kv in sorted(A["buckets"].items(), key=lambda kv: -kv[1]))))


class FightsPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Fights")
        self.p = RunPicker(self)
        ttk.Button(self.bar, text="Show", command=self.refresh).pack(side="left")
        self.f, self.c = self.fig(3.2)
        self.t = self.table(("act", "encounter", "room", "fights", "win rate", "net HP lost", "potions/fight", "turns"),
                            (40, 210, 80, 55, 70, 90, 90, 55), 10)

    def refresh(self):
        a = RunPicker.clean(self.p.a.get())
        if a:
            self.load("fights %s" % _q(a), self._got)

    def _got(self, d):
        self.f.clear()
        a1, a2 = self.f.add_subplot(121), self.f.add_subplot(122)
        hp = d["boss_entry_hp"]
        for k, col in (("win", "#2e7d32"), ("loss", "#c62828")):
            for act, vals in sorted(hp.get(k, {}).items()):
                if vals:
                    a1.boxplot([vals], positions=[int(act) + (0.18 if k == "win" else -0.18)], widths=0.3, manage_ticks=False,
                               patch_artist=True, boxprops={"facecolor": col, "alpha": .5}, medianprops={"color": "black"})
        a1.set_xticks([1, 2, 3], labels=["Act 1 boss", "Act 2 boss", "Act 3 boss"]); a1.set_xlim(0.5, 3.5)
        a1.set_ylabel("HP / max HP entering the boss"); a1.set_ylim(0, 1.05)
        a1.set_title("HP entering each act boss: runs that won (green) vs lost (red)", fontsize=9)
        H = d["harvest"]["rows"]
        labels = ["A%d %s" % (r["act"], r["room"][:4]) for r in H]
        a2.bar(range(len(H)), [r["hp_lost_frac"] * 100 for r in H], color=[ROOM_COL.get(r["room"], "#bdbdbd") for r in H])
        a2b = a2.twinx()
        a2b.plot(range(len(H)), [100 * (1 - r["win_rate"]) for r in H], "k.-", lw=1)
        a2.set_xticks(range(len(H))); a2.set_xticklabels(labels, rotation=60, fontsize=7)
        a2.set_ylabel("HP lost, % of max (bars)"); a2b.set_ylabel("death rate % (line)")
        a2.set_title("mainline_h1 harvest: %d seeds, all fights" % d["harvest"]["files"], fontsize=9)
        self.c.draw_idle()
        _fill(self.t, [(e["act"], e["enc"], e["room"], e["n"], _f(e["win_rate"]), _f(e["mean_hp_lost"], 1), _f(e["potions"]), _f(e["turns"], 1))
                       for e in d["encounters"]])
        self.status.set(self.status.get() + "   (%d traced seeds)" % d["traced_seeds"])


class CardsPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Cards")
        self.p = RunPicker(self)
        ttk.Button(self.bar, text="Show", command=self.refresh).pack(side="left")
        self.f, self.c = self.fig(3.0)
        self.t = self.table(("card", "offered", "picked", "pick rate", "runs with it", "mean floor", "Δ floor vs all", "win rate"),
                            (190, 60, 60, 70, 85, 80, 95, 65), 12)

    def refresh(self):
        a = RunPicker.clean(self.p.a.get())
        if a:
            self.load("cards %s" % _q(a), self._got)

    def _got(self, d):
        base = d["base_floor"]
        _fill(self.t, [(c["card"], c["offered"], c["picked"], _f(c["pick_rate"]), c["runs"], _f(c["mean_floor"], 1),
                        _f(c["mean_floor"] - base if c["mean_floor"] is not None else None, 1), _f(c["win_rate"]))
                       for c in d["cards"]])
        self.f.clear()
        axs = [self.f.add_subplot(1, 3, i + 1) for i in range(3)]
        for ax, m, yl in zip(axs, ("size", "upgraded", "starters"), ("deck size", "upgraded cards", "Strikes+Defends")):
            for k, col, off in (("win", "#2e7d32", 0.18), ("loss", "#c62828", -0.18)):
                s = d["act_entry_deck"].get(k, {})
                xs = sorted(int(a) for a in s)
                ax.bar([x + off for x in xs], [s[str(x)][m] for x in xs], width=0.34, color=col, alpha=.7, label=k)
            ax.set_xticks([1, 2, 3]); ax.set_title(yl + " at act entry", fontsize=9)
        axs[0].legend(fontsize=7)
        self.c.draw_idle()
        self.status.set(self.status.get() + "   (%d traced seeds, mean floor %.2f)" % (d["traced_seeds"], base))


class SeedDiffPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Seed diff")
        self.p = RunPicker(self, two=True, label_b="B:")
        ttk.Label(self.bar, text="seed:").pack(side="left")
        self.seed = tk.StringVar(value="10007")
        ttk.Entry(self.bar, textvariable=self.seed, width=9).pack(side="left", padx=(2, 6))
        ttk.Button(self.bar, text="Diff", command=self.refresh).pack(side="left")
        self.info = tk.Text(self.frame, height=7, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.info.pack(fill="x", pady=(4, 0))
        self.f, self.c = self.fig(3.6)

    def refresh(self):
        a, b, s = RunPicker.clean(self.p.a.get()), RunPicker.clean(self.p.b.get()), self.seed.get().strip()
        if not (a and b and s.isdigit()):
            messagebox.showinfo("Seed diff", "Pick run A, run B and a numeric seed.")
            return
        self.info.delete("1.0", "end")
        self.info.insert("end", "Diffing seed %s: %s vs %s (replays any run without a trace, TRACE=1, nice 19; can take minutes)..." % (s, a, b))
        self.load("seeddiff %s %s %s" % (s, _q(a), _q(b)), self._got, timeout=3600)

    def _got(self, d):
        self.info.delete("1.0", "end")
        if "error" in d:
            self.info.insert("end", d["error"]); return
        A, B, fd = d["a"], d["b"], d["first_divergence"]
        L = ["seed %d   A %s: floor %d win %d   |   B %s: floor %d win %d" % (d["seed"], A["run"], A["floor"], A["win"], B["run"], B["floor"], B["win"])]
        if fd:
            L += ["first divergence at floor %s:" % fd["floor"], "   A: %s" % fd["a"], "   B: %s" % fd["b"]]
            L += ["   before:  " + " | ".join(fd["context"])]
        else:
            L.append("trajectories identical")
        self.info.insert("end", "\n".join(L))
        self.f.clear()
        ax = self.f.add_subplot(111)
        for X, col in ((A, "#1565c0"), (B, "#ef6c00")):
            ax.plot([c[0] for c in X["curve"]], [c[1] for c in X["curve"]], "-", color=col, label="%s HP" % X["run"])
            ax.plot([c[0] for c in X["curve"]], [c[2] for c in X["curve"]], ":", color=col, lw=1)
        if fd and fd["floor"] is not None:
            ax.axvline(fd["floor"], color="red", ls="--", lw=1, label="first divergence")
        ax.set_xlabel("floor"); ax.set_ylabel("HP (dotted: max HP)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        self.c.draw_idle()


class TrainingPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Training")
        ttk.Button(self.bar, text="Refresh", command=self.refresh).pack(side="left")
        self.gate = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.gate, wraplength=1000, foreground="#2e7d32").pack(anchor="w")
        self.f, self.c = self.fig(5)

    def refresh(self):
        self.load("training", self._got)

    def _got(self, d):
        self.f.clear()
        if not d:
            return
        if "rl" in d and d["rl"].get("rl"):
            h = d["rl"]
            R, Vv = h["rl"], h.get("rl_val", [])
            self.gate.set("nets/rl: out-of-combat policy PPO, %d iterations, %d games; latest val: %s" % (
                len(R), R[-1]["games"], ("%s floor %.2f, %d wins / %d" % (Vv[-1]["tag"], Vv[-1]["floor"], Vv[-1]["wins"], Vv[-1]["n"])) if Vv else "--"))
            a = self.f.add_subplot(221); a.plot([x["iter"] for x in R], [x["floor"] for x in R]); a.set_title("RL: mean floor per batch (sampled policy)", fontsize=9)
            a = self.f.add_subplot(222); a.plot([x["iter"] for x in R], [100 * x["win_rate"] for x in R]); a.set_title("RL: win % per batch", fontsize=9)
            a = self.f.add_subplot(223); a.plot([x["iter"] for x in R], [x["entropy"] for x in R]); a.set_title("RL: policy entropy", fontsize=9); a.set_xlabel("iteration")
            a = self.f.add_subplot(224)
            a.plot(range(len(Vv)), [x["floor"] for x in Vv], "o-"); a.set_xticks(range(len(Vv)), [x["tag"] for x in Vv], rotation=45, fontsize=7)
            a.set_title("RL: greedy policy on 100 validation seeds (floor)", fontsize=9)
            self.c.draw_idle()
            return
        name, h = sorted(d.items())[-1]
        g = h.get("gate")
        self.gate.set("nets/%s gate: %s" % (name, ("prior %s, value %s — %s" % ("PASS" if g["prior"] else "FAIL", "PASS" if g["value"] else "FAIL", g["why"])) if g else "n/a"))
        P, V = h.get("prior", []), h.get("value", [])
        a = self.f.add_subplot(221); a.plot([x["epoch"] for x in P], [x["loss"] for x in P]); a.set_title("prior: train loss", fontsize=9)
        a = self.f.add_subplot(222)
        for k in ("regret@1", "regret@2", "regret@3"):
            a.plot([x["epoch"] for x in P], [x[k] for x in P], label=k)
        a.set_title("prior: val regret vs K=64 Q", fontsize=9); a.legend(fontsize=7)
        a = self.f.add_subplot(223); a.plot([x["epoch"] for x in V], [x["loss"] for x in V]); a.set_title("value: train loss", fontsize=9)
        a.set_xlabel("epoch")
        a = self.f.add_subplot(224)
        a.plot([x["epoch"] for x in V], [x["V_mae"] for x in V], label="V MAE")
        if V:
            a.axhline(V[0]["V_mae_const"], color="gray", ls=":", label="constant MAE")
        a2 = a.twinx(); a2.plot([x["epoch"] for x in V], [x["V_r2"] for x in V], color="#2e7d32", label="R²"); a2.set_ylabel("R²")
        a.set_title("value: val MAE (blue) / R² (green)", fontsize=9); a.legend(fontsize=7, loc="upper left"); a.set_xlabel("epoch")
        self.c.draw_idle()


class DigestPanel(_Panel):
    def __init__(self, nb, app):
        super().__init__(nb, app, "Digest")
        ttk.Button(self.bar, text="Build digest (last 24 h)", command=self.refresh).pack(side="left")
        ttk.Button(self.bar, text="Save to sim_logs/DIGEST_<date>.md", command=self.save).pack(side="left", padx=6)
        self.text = tk.Text(self.frame, font=("Consolas", 9), wrap="none")
        self.text.pack(fill="both", expand=True, pady=(4, 0))

    def refresh(self, write=False):
        self.load("digest --md" + (" --write" if write else ""), self._got)

    def save(self):
        self.refresh(write=True)

    def _got(self, md):
        self.text.delete("1.0", "end")
        self.text.insert("end", md)


class AnalyticsTab:
    def __init__(self, notebook, logger=None):
        self.frame = ttk.Frame(notebook, padding=4)
        notebook.add(self.frame, text="  Analytics  ")
        self._ui: queue.Queue = queue.Queue()
        self.pickers = []
        top = ttk.Frame(self.frame)
        top.pack(fill="x")
        ttk.Button(top, text="Reload run lists", command=self.load_runs).pack(side="left")
        ttk.Label(top, text="  Data: ~/sts/dash_data.py (results ledger, traces, harvest). Traced runs show [traced].",
                  foreground="gray").pack(side="left")
        nb = ttk.Notebook(self.frame)
        nb.pack(fill="both", expand=True, pady=(4, 0))
        self.panels = [ProgressPanel(nb, self), DeathsPanel(nb, self), FightsPanel(nb, self), CardsPanel(nb, self),
                       SeedDiffPanel(nb, self), TrainingPanel(nb, self), DigestPanel(nb, self)]
        self.frame.after(300, self._drain)
        self.frame.after(500, self.load_runs)
        self.frame.after(800, self.panels[0].refresh)

    def post(self, fn):
        self._ui.put(fn)

    def _drain(self):
        try:
            while True:
                self._ui.get_nowait()()
        except queue.Empty:
            pass
        self.frame.after(200, self._drain)

    def load_runs(self):
        def go():
            try:
                pr, tr = dash("progress"), dash("traced")
                names = [r["name"] for r in pr["runs"]]
                self.post(lambda: [p.set_runs(names, set(tr["runs"])) for p in self.pickers])
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=go, daemon=True).start()


class HealthStrip:
    """One-line machine/queue health with a red banner on alerts (polls every 20 s)."""

    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg="#eceff1")
        self.var = tk.StringVar(value="health: connecting...")
        self.lbl = tk.Label(self.frame, textvariable=self.var, bg="#eceff1", anchor="w", font=("Segoe UI", 9))
        self.lbl.pack(fill="x", padx=6, pady=2)
        self.alert = tk.Label(self.frame, text="", bg="#c62828", fg="white", anchor="w", font=("Segoe UI", 9, "bold"))
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self.frame.after(500, self._drain)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._q.put(dash("health", 60))
            except Exception as e:  # noqa: BLE001
                self._q.put({"error": str(e)})
            self._stop.wait(20)

    def _drain(self):
        try:
            while True:
                self._apply(self._q.get_nowait())
        except queue.Empty:
            pass
        if not self._stop.is_set():
            self.frame.after(1000, self._drain)

    def _apply(self, h):
        if "error" in h:
            self.var.set("health: WSL unreachable (%s)" % h["error"][-80:]); return
        wd = h.get("watchdog_age_s")
        q = h.get("queue", {})
        self.var.set(("CPU %.0f%%   RAM %.1f/%.0f GB   GPU %s%% %s°C %s/%s MB   disk %.0f GB free   %d seeds/h (30 min)   "
                      "watchdog %s ago, keep-alive %s   |   running: %s   overnight %s   jobq %s") % (
            h["cpu_pct"], h["ram_used_gb"], h["ram_total_gb"], h.get("gpu_util"), h.get("gpu_temp"), h.get("gpu_mem"), h.get("gpu_mem_total"),
            h["disk_free_gb"], h["seeds_per_hour"], "--" if wd is None else "%dm" % (wd // 60), "on" if h.get("keepalive") else "OFF",
            q.get("running", "-"), q.get("overnight", {}), q.get("jobq", {})))
        if h.get("alerts"):
            self.alert.configure(text="  ⚠  " + "   |   ".join(h["alerts"]))
            self.alert.pack(fill="x")
        else:
            self.alert.pack_forget()

    def close(self):
        self._stop.set()
