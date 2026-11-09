"""Sim Experiments tab for the AscensionAI control panel.

Tracks the headless-sim experiment runs that live in WSL (~/sts/sim_logs),
via `run_status.py --json` (polled in a background thread every few seconds),
and can launch preset runs / stop a run (kills its process group) through
`launch_run.sh`. Everything here is read-only except the Launch/Stop buttons.
"""
import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

WSL = ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc"]
POLL_SECS = 5.0
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PRESETS = [
    ("mainline_v3 reproduction (150 seeds)", "repro_v3"),
    ("Honest combat K=1 (150 seeds)", "honest_K1"),
    ("Honest combat K=4 (150 seeds)", "honest_K4"),
    ("Honest combat K=8 (150 seeds)", "honest_K8"),
    ("Honest draft teacher K=4 (50 seeds)", "honest_draft50"),
]

STATE_ICON = {"done": "✔", "running": "▶", "queued": "⏸",
              "stopped": "■", "failed": "✖", "pending": "○"}
STATE_COLOR = {"done": "#2e7d32", "running": "#1565c0", "queued": "#757575",
               "stopped": "#ef6c00", "failed": "#c62828", "pending": "#9e9e9e"}


def _wsl(cmd, timeout=30):
    return subprocess.run(WSL + [cmd], capture_output=True, text=True, timeout=timeout,
                          creationflags=_NOWIN, encoding="utf-8", errors="replace")


def fmt_dur(s):
    if s is None:
        return "--"
    s = int(max(0, s))
    if s >= 3600:
        return "%dh %02dm" % (s // 3600, (s % 3600) // 60)
    if s >= 60:
        return "%dm %02ds" % (s // 60, s % 60)
    return "%ds" % s


class SimExperimentsTab:
    def __init__(self, notebook: ttk.Notebook, logger=None):
        self.log = logger
        self.frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.frame, text="  Sim Experiments  ")
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._selected: str | None = None
        self._rows: dict[str, dict] = {}
        self._run_order: list[str] = []
        self._last_status = None
        self._last_ok = 0.0
        st = ttk.Style()
        for nm, col in (("Run", "#1e88e5"), ("Done", "#43a047"), ("Stop", "#fb8c00"),
                        ("Fail", "#e53935"), ("Queue", "#9e9e9e")):
            st.configure("%s.Horizontal.TProgressbar" % nm, background=col, troughcolor="#e6e6e6",
                         bordercolor="#bdbdbd", lightcolor=col, darkcolor=col)
        self._build()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.frame.after(200, self._drain)

    # ------------------------------------------------------------------ UI
    def _build(self):
        f = self.frame
        top = ttk.Frame(f)
        top.pack(fill="x")
        self.phase_box = ttk.LabelFrame(top, text="  Plan checklist  ", padding=6)
        self.phase_box.pack(side="left", fill="both", expand=True)
        self.phase_inner = ttk.Frame(self.phase_box)
        self.phase_inner.pack(fill="both", expand=True)

        ctl = ttk.LabelFrame(top, text="  Launch / control  ", padding=6)
        ctl.pack(side="left", fill="y", padx=(8, 0))
        self.preset_var = tk.StringVar(value=PRESETS[1][0])
        ttk.Combobox(ctl, textvariable=self.preset_var, values=[p[0] for p in PRESETS],
                     state="readonly", width=36).pack(anchor="w")
        b = ttk.Frame(ctl)
        b.pack(fill="x", pady=(6, 0))
        ttk.Button(b, text="Launch preset", command=self._launch).pack(side="left")
        self.stop_btn = ttk.Button(b, text="Stop selected run", command=self._stop_selected)
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.conn_var = tk.StringVar(value="connecting to WSL...")
        ttk.Label(ctl, textvariable=self.conn_var, foreground="gray").pack(anchor="w", pady=(6, 0))

        # queue (combined) rows
        self.queue_box = ttk.LabelFrame(f, text="  Queues (combined)  ", padding=6)
        self.queue_box.pack(fill="x", pady=(8, 0))
        self.queue_inner = ttk.Frame(self.queue_box)
        self.queue_inner.pack(fill="x")
        self._queue_rows: dict[str, dict] = {}

        # job queue (overnight chain + jobq)
        jq = ttk.LabelFrame(f, text="  Job queue  (overnight chain + jobq; runs one job at a time in WSL)  ", padding=6)
        jq.pack(fill="x", pady=(8, 0))
        cols = ("src", "name", "state", "note")
        self.jq_tree = ttk.Treeview(jq, columns=cols, show="headings", height=5)
        for c, w in zip(cols, (90, 430, 80, 360)):
            self.jq_tree.heading(c, text=c)
            self.jq_tree.column(c, width=w, stretch=(c in ("name", "note")))
        for stt, col in STATE_COLOR.items():
            self.jq_tree.tag_configure(stt, foreground=col)
        self.jq_tree.tag_configure("cancelled", foreground="#9e9e9e")
        self.jq_tree.tag_configure("skipped", foreground="#9e9e9e")
        self.jq_tree.tag_configure("todo", foreground="#9e9e9e")
        self.jq_tree.pack(fill="x")
        jb = ttk.Frame(jq)
        jb.pack(fill="x", pady=(4, 0))
        self.jq_preset = tk.StringVar(value="eval150_vs_K64")
        self.jq_preset_box = ttk.Combobox(jb, textvariable=self.jq_preset, width=16, state="readonly",
                                          values=["eval150", "eval150_vs_K64", "holdout150"])
        self.jq_preset_box.pack(side="left")
        ttk.Label(jb, text="arm:").pack(side="left", padx=(6, 0))
        self.jq_arm = tk.StringVar(value="my_arm")
        ttk.Entry(jb, textvariable=self.jq_arm, width=18).pack(side="left")
        ttk.Label(jb, text="env:").pack(side="left", padx=(6, 0))
        self.jq_env = tk.StringVar(value="HONEST=1 HONEST_CPP=1 HONEST_K=64")
        ttk.Entry(jb, textvariable=self.jq_env, width=46).pack(side="left")
        ttk.Button(jb, text="Append", command=self._jq_add).pack(side="left", padx=(6, 0))
        for txt, act in (("Cancel", "cancel"), ("▲", "up"), ("▼", "down"), ("Retry", "retry")):
            ttk.Button(jb, text=txt, width=7 if len(txt) > 2 else 3,
                       command=lambda a=act: self._jq_act(a)).pack(side="left", padx=(4, 0))
        self._jq_events_n = None
        self._jq_ids = {}

        # LLM teacher (local Ollama)
        lt = ttk.LabelFrame(f, text="  LLM teacher  (local Ollama; out-of-combat choices, honest state only)  ", padding=6)
        lt.pack(fill="x", pady=(8, 0))
        self.llm_var = tk.StringVar(value="Ollama: checking...")
        ttk.Label(lt, textvariable=self.llm_var, font=("Consolas", 9)).pack(anchor="w")
        lb = ttk.Frame(lt)
        lb.pack(fill="x", pady=(4, 0))
        ttk.Label(lb, text="model:").pack(side="left")
        self.llm_model = tk.StringVar(value="qwen2.5-coder:7b")
        self.llm_model_box = ttk.Combobox(lb, textvariable=self.llm_model, width=22, state="readonly", values=["qwen2.5-coder:7b"])
        self.llm_model_box.pack(side="left", padx=(2, 8))
        self.llm_think = tk.StringVar(value="")
        ttk.Label(lb, text="thinking:").pack(side="left")
        ttk.Combobox(lb, textvariable=self.llm_think, width=6, state="readonly", values=["", "1", "0"]).pack(side="left", padx=(2, 8))
        ttk.Label(lb, text="sim workers:").pack(side="left")
        self.llm_par = tk.StringVar(value="4")
        ttk.Entry(lb, textvariable=self.llm_par, width=4).pack(side="left", padx=(2, 8))
        ttk.Button(lb, text="Queue pilot (3-seed smoke + 50 eval500 seeds)", command=self._llm_launch).pack(side="left")

        # per-run table
        runs = ttk.LabelFrame(f, text="  Runs  (click a row to tail its log; newest at the bottom)  ", padding=6)
        runs.pack(fill="x", pady=(8, 0))
        self._runs_canvas = tk.Canvas(runs, height=250, highlightthickness=0)
        vs = ttk.Scrollbar(runs, orient="vertical", command=self._runs_canvas.yview)
        self._runs_canvas.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        self._runs_canvas.pack(side="left", fill="both", expand=True)
        self.grid = ttk.Frame(self._runs_canvas)
        self._grid_win = self._runs_canvas.create_window((0, 0), window=self.grid, anchor="nw")
        self.grid.bind("<Configure>", lambda _e: self._runs_canvas.configure(scrollregion=self._runs_canvas.bbox("all")))
        self._runs_canvas.bind("<Configure>", lambda e: self._runs_canvas.itemconfigure(self._grid_win, width=e.width))
        self._runs_canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")
        hdr = ["", "Run", "Progress", "%", "Done", "Wins", "Avg floor", "Elapsed", "ETA", "State"]
        for c, h in enumerate(hdr):
            ttk.Label(self.grid, text=h, font=("Segoe UI", 9, "bold")).grid(row=0, column=c, sticky="w", padx=4)
        self.grid.columnconfigure(2, weight=1)

        # compare / inspect (results ledger)
        cf = ttk.LabelFrame(f, text="  Compare & inspect (results ledger)  ", padding=6)
        cf.pack(fill="x", pady=(8, 0))
        self.cmp_a = tk.StringVar()
        self.cmp_b = tk.StringVar()
        ttk.Label(cf, text="A:").pack(side="left")
        self.cmp_a_box = ttk.Combobox(cf, textvariable=self.cmp_a, width=24, state="readonly")
        self.cmp_a_box.pack(side="left", padx=(2, 6))
        ttk.Label(cf, text="B:").pack(side="left")
        self.cmp_b_box = ttk.Combobox(cf, textvariable=self.cmp_b, width=24, state="readonly")
        self.cmp_b_box.pack(side="left", padx=(2, 6))
        ttk.Button(cf, text="Compare A vs B", command=self._compare).pack(side="left")
        ttk.Separator(cf, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(cf, text="Seed:").pack(side="left")
        self.seed_var = tk.StringVar(value="10007")
        ttk.Entry(cf, textvariable=self.seed_var, width=9).pack(side="left", padx=(2, 4))
        ttk.Button(cf, text="Inspect seed (run A)", command=self._inspect).pack(side="left")
        ttk.Separator(cf, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(cf, text="Leaderboard", command=self._leaderboard).pack(side="left")
        ttk.Button(cf, text="Export to website", command=self._export).pack(side="left", padx=(4, 0))

        # log pane
        lf = ttk.LabelFrame(f, text="  Log  ", padding=4)
        lf.pack(fill="both", expand=True, pady=(8, 0))
        self.log_title = tk.StringVar(value="(select a run)")
        ttk.Label(lf, textvariable=self.log_title, font=("Consolas", 9)).pack(anchor="w")
        tf = ttk.Frame(lf)
        tf.pack(fill="both", expand=True)
        self.text = tk.Text(tf, wrap="none", font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                            height=12, state="disabled")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

    def _on_wheel(self, e):
        try:
            w = self.frame.winfo_containing(e.x_root, e.y_root)
            while w is not None and w is not self._runs_canvas:
                w = w.master
            if w is self._runs_canvas:
                self._runs_canvas.yview_scroll(int(-e.delta / 120), "units")
        except Exception:
            pass

    def _set_phases(self, phases):
        for w in self.phase_inner.winfo_children():
            w.destroy()
        col = 0
        for ph in phases or []:
            box = ttk.Frame(self.phase_inner)
            box.grid(row=0, column=col, sticky="nw", padx=(0, 18))
            col += 1
            steps = ph.get("steps", [])
            nd = sum(1 for s in steps if s.get("state") == "done")
            ttk.Label(box, text="%s  (%d/%d)" % (ph.get("phase", ""), nd, len(steps)),
                      font=("Segoe UI", 9, "bold")).pack(anchor="w")
            for s in steps:
                st = s.get("state", "pending")
                tk.Label(box, text="%s %s" % (STATE_ICON.get(st, "○"), s.get("name", "")),
                         fg=STATE_COLOR.get(st, "#000"), font=("Segoe UI", 9), anchor="w").pack(anchor="w")

    def _row_widgets(self, r_idx, name):
        g = self.grid
        sel = tk.StringVar(value="")
        w = {
            "sel": ttk.Label(g, textvariable=sel, width=2),
            "name": ttk.Label(g, text=name, font=("Consolas", 9), cursor="hand2"),
            "bar": ttk.Progressbar(g, orient="horizontal", mode="determinate", maximum=100, length=220),
            "pct": ttk.Label(g, width=7, font=("Consolas", 9)),
            "done": ttk.Label(g, width=13, font=("Consolas", 9)),
            "wins": ttk.Label(g, width=5, font=("Consolas", 9)),
            "floor": ttk.Label(g, width=8, font=("Consolas", 9)),
            "elapsed": ttk.Label(g, width=9, font=("Consolas", 9)),
            "eta": ttk.Label(g, width=10, font=("Consolas", 9)),
            "state": tk.Label(g, width=10, font=("Segoe UI", 9, "bold"), anchor="w"),
        }
        for c, k in enumerate(["sel", "name", "bar", "pct", "done", "wins", "floor", "elapsed", "eta", "state"]):
            w[k].grid(row=r_idx, column=c, sticky="we" if k == "bar" else "w", padx=4, pady=1)
            w[k].bind("<Button-1>", lambda _e, n=name: self._select(n))
        w["selvar"] = sel
        return w

    def _queue_row(self, name):
        if name in self._queue_rows:
            return self._queue_rows[name]
        r = len(self._queue_rows)
        lab = ttk.Label(self.queue_inner, text=name, font=("Segoe UI", 9, "bold"), width=18)
        bar = ttk.Progressbar(self.queue_inner, orient="horizontal", mode="determinate", maximum=100, length=360)
        txt = ttk.Label(self.queue_inner, font=("Consolas", 9))
        lab.grid(row=r, column=0, sticky="w")
        bar.grid(row=r, column=1, sticky="we", padx=6)
        txt.grid(row=r, column=2, sticky="w")
        self.queue_inner.columnconfigure(1, weight=1)
        self._queue_rows[name] = {"bar": bar, "txt": txt}
        return self._queue_rows[name]

    # ------------------------------------------------------------- updates
    def _apply(self, st):
        self._last_status = st
        self._set_phases(st.get("phases"))
        names = [r["name"] for r in st.get("runs", [])]
        if list(self.cmp_a_box["values"] or ()) != names:
            self.cmp_a_box["values"] = names
            self.cmp_b_box["values"] = names
            if not self.cmp_a.get() and names:
                self.cmp_a.set("combat_K8" if "combat_K8" in names else names[0])
            if not self.cmp_b.get() and names:
                self.cmp_b.set(names[-1])
        if names != self._run_order:
            for row in self._rows.values():
                for k, v in row.items():
                    if k != "selvar":
                        v.destroy()
            self._rows = {n: self._row_widgets(i + 1, n) for i, n in enumerate(names)}
            self._run_order = names
            self.frame.after(100, lambda: self._runs_canvas.yview_moveto(1.0))   # newest runs in view
        for r in st.get("runs", []):
            w = self._rows[r["name"]]
            w["bar"]["value"] = r["pct"]
            w["bar"].configure(style=_bar_style(r["state"]))
            w["pct"]["text"] = "%5.1f%%" % r["pct"]
            w["done"]["text"] = "%d/%d" % (r["done"], r["total"])
            w["wins"]["text"] = str(r["wins"])
            w["floor"]["text"] = "%.2f" % r["avg_floor"] if r["avg_floor"] is not None else "--"
            w["elapsed"]["text"] = "ref" if r.get("group") == "clairvoyant_ref" else fmt_dur(r["elapsed"])
            if r["state"] == "running":
                w["eta"]["text"] = fmt_dur(r["eta"]) if r["eta"] is not None else "estimating"
            else:
                w["eta"]["text"] = "--"
            w["state"].configure(text="%s %s" % (STATE_ICON.get(r["state"], ""), r["state"]),
                                 fg=STATE_COLOR.get(r["state"], "#000"))
            w["selvar"].set("▸" if r["name"] == self._selected else "")
        for q in st.get("queues", []):
            w = self._queue_row(q["name"])
            w["bar"]["value"] = q["pct"]
            w["bar"].configure(style=_bar_style(q["state"]))
            if q["state"] == "done":
                eta = "done"
            elif q["eta"] is not None:
                eta = "ETA " + fmt_dur(q["eta"])
            elif q.get("eta_partial"):
                eta = "ETA >= " + fmt_dur(q["eta_partial"]) + " (estimating)"
            else:
                eta = "ETA estimating"
            w["txt"]["text"] = "%5.1f%%  %d/%d seeds  %s  [%s]" % (q["pct"], q["done"], q["total"], eta, q["state"])
        if self._selected is None and not getattr(self, "_pinned", False):
            running = [r["name"] for r in st.get("runs", []) if r["state"] == "running"]
            if running:
                self._select(running[0])

    def _select(self, name):
        self._selected = name
        self._pinned = False
        for n, w in self._rows.items():
            w["selvar"].set("▸" if n == name else "")
        r = self._run(name)
        if r:
            extra = ""
            if r.get("inflight_floors"):
                extra = "   in-flight floors: " + ",".join(map(str, r["inflight_floors"]))
            self.log_title.set("%s  |  %s  |  env: %s%s" % (name, r.get("description", ""), r.get("env") or "(defaults)", extra))
        self._q.put(("tail_now", None))

    def _run(self, name):
        for r in (self._last_status or {}).get("runs", []):
            if r["name"] == name:
                return r
        return None

    def _set_log(self, text):
        y = self.text.yview()
        at_end = y[1] > 0.98
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", text)
        self.text.configure(state="disabled")
        if at_end:
            self.text.see("end")

    # ------------------------------------------------------- background IO
    def _poll_loop(self):
        tail_now = threading.Event()
        self._tail_now = tail_now
        last_tail = 0.0
        while not self._stop.is_set():
            try:
                p = _wsl("cd ~/sts && .venv/bin/python run_status.py --json", timeout=40)
                if p.returncode == 0 and p.stdout.strip():
                    self._q.put(("status", json.loads(p.stdout)))
                else:
                    self._q.put(("err", (p.stderr or "run_status failed").strip()[-200:]))
            except Exception as e:  # noqa: BLE001
                self._q.put(("err", str(e)[-200:]))
            try:
                since = self._jq_events_n if self._jq_events_n is not None else -1
                p = _wsl("cd ~/sts && python3 jobq.py dash %d" % max(0, since), timeout=30)
                if p.returncode == 0 and p.stdout.strip():
                    d = json.loads(p.stdout)
                    if since < 0:            # first poll: don't replay old events as notifications
                        d["events"] = []
                    self._q.put(("jobq", d))
            except Exception:  # noqa: BLE001
                pass
            self._llm_tick = getattr(self, "_llm_tick", 0) + 1
            if self._llm_tick % 3 == 1:
                try:
                    p = _wsl("cd ~/sts && .venv/bin/python dash_data.py llm", timeout=30)
                    if p.returncode == 0 and p.stdout.strip():
                        self._q.put(("llm", json.loads(p.stdout)))
                except Exception:  # noqa: BLE001
                    pass
            name = self._selected
            r = self._run(name) if name else None
            if r and r.get("log"):
                try:
                    p = _wsl("cd ~/sts && tail -n 300 %s 2>/dev/null" % _shq(r["log"]), timeout=20)
                    self._q.put(("log", p.stdout or "(log is empty)"))
                except Exception as e:  # noqa: BLE001
                    self._q.put(("log", "tail failed: %s" % e))
            self._stop.wait(POLL_SECS)

    def _drain(self):
        try:
            while True:
                kind, val = self._q.get_nowait()
                if kind == "status":
                    self._last_ok = time.time()
                    self.conn_var.set("WSL ok  |  updated %s" % time.strftime("%H:%M:%S"))
                    self._apply(val)
                elif kind == "err":
                    self.conn_var.set("WSL error: %s" % val)
                elif kind == "jobq":
                    self._apply_jobq(val)
                elif kind == "llm":
                    self._apply_llm(val)
                elif kind == "log":
                    if self._selected is not None:
                        self._set_log(val)
                elif kind == "pinned":
                    title, txt, after = val
                    if after is not None:
                        txt = after(txt)
                    self._show_text(title, txt)
        except queue.Empty:
            pass
        if not self._stop.is_set():
            self.frame.after(300, self._drain)

    # ------------------------------------------------------------ job queue
    def _apply_jobq(self, d):
        sel = self.jq_tree.selection()
        sel_key = self.jq_tree.item(sel[0], "values")[1] if sel else None
        self.jq_tree.delete(*self.jq_tree.get_children())
        self._jq_ids = {}
        for j in d.get("overnight", []):
            iid = self.jq_tree.insert("", "end", values=("overnight", j["name"], j["state"], j.get("note", "")),
                                      tags=(j["state"],))
            self._jq_ids[iid] = None
        for j in d.get("jobs", []):
            iid = self.jq_tree.insert("", "end", values=("jobq #%d" % j["id"], j["name"], j["state"],
                                                         ("rc %s" % j["rc"]) if "rc" in j else j["cmd"][:80]),
                                      tags=(j["state"],))
            self._jq_ids[iid] = j["id"]
            if sel_key == j["name"]:
                self.jq_tree.selection_set(iid)
        if d.get("presets"):
            self.jq_preset_box.configure(values=d["presets"])
        for ev in d.get("events", []):
            if ev["state"] in ("done", "failed"):
                _toast("AscensionAI job %s" % ev["state"], ev["name"])
        self._jq_events_n = d.get("events_n", self._jq_events_n)

    def _apply_llm(self, d):
        if d.get("installed"):
            self.llm_model_box.configure(values=d["installed"])
        loaded = ", ".join("%s (%.1f GB VRAM)" % (m["name"], m["vram_gb"]) for m in d.get("loaded", [])) or "none"
        line1 = "Ollama %s at %s   loaded: %s" % ("UP" if d.get("up") else "DOWN", d.get("host"), loaded)
        if d.get("run"):
            line2 = ("run %s  model %s  thinking %s   seeds %s/%s   decisions %d   %.1f/min   latency mean %s s, median %s s"
                     "   fallback %.1f%%   card picks differing from policy %s") % (
                d["run"], d.get("model"), {"1": "on", "0": "off"}.get(d.get("think") or "", "default"), d.get("done"), d.get("total"),
                d.get("decisions", 0), d.get("per_min", 0), d.get("mean_latency"), d.get("median_latency"),
                100 * (d.get("fallback_rate") or 0),
                "--" if d.get("card_disagree") is None else "%.0f%%" % (100 * d["card_disagree"]))
        else:
            line2 = "no LLM runs yet"
        self.llm_var.set(line1 + "\n" + line2)

    def _llm_launch(self):
        m, t, par = self.llm_model.get().strip(), self.llm_think.get().strip(), self.llm_par.get().strip() or "4"
        if not m or not par.isdigit():
            return
        if not messagebox.askyesno("LLM pilot", "Queue the LLM teacher pilot?\n\nmodel %s, thinking %s, %s sim workers\n"
                                   "3-seed smoke test, then 50 eval500 seeds (K=64 combat), compared with eval500_K64."
                                   % (m, t or "default", par)):
            return
        self._bg("queue LLM pilot", "python3 jobq.py add 'LLM pilot %s think=%s' -- ./llm_pilot.sh %s %s %s && ./overnight_watchdog.sh && python3 jobq.py list"
                 % (m, t or "-", _shq(m), _shq(t) if t else "''", par))

    def _jq_selected_id(self):
        sel = self.jq_tree.selection()
        return self._jq_ids.get(sel[0]) if sel else None

    def _jq_add(self):
        arm, env, pre = self.jq_arm.get().strip(), self.jq_env.get().strip(), self.jq_preset.get()
        if not arm or not env or " " in arm:
            messagebox.showinfo("Append job", "Give an arm name (no spaces) and an env string.")
            return
        if not messagebox.askyesno("Append job", "Append to the job queue?\n\n%s  %s\n%s" % (pre, arm, env)):
            return
        self._bg("append job", "python3 jobq.py add-preset %s %s %s && ./overnight_watchdog.sh && python3 jobq.py list"
                 % (_shq(pre), _shq(arm), _shq(env)))

    def _jq_act(self, act):
        jid = self._jq_selected_id()
        if jid is None:
            messagebox.showinfo("Job queue", "Select a jobq job first (overnight-chain jobs are fixed).")
            return
        cmd = "move %d %s" % (jid, act) if act in ("up", "down") else "%s %d" % (act, jid)
        self._bg("job queue: %s" % cmd, "python3 jobq.py %s" % cmd)

    # ----------------------------------------------------------- actions
    def _launch(self):
        label = self.preset_var.get()
        key = dict(PRESETS).get(label)
        if not key:
            return
        if not messagebox.askyesno("Launch run", "Launch '%s' in WSL?\nIt runs detached; this window can be closed." % label):
            return
        def go():
            try:
                p = _wsl("cd ~/sts && ./launch_run.sh %s" % key, timeout=60)
                msg = (p.stdout + p.stderr).strip() or "launched"
            except Exception as e:  # noqa: BLE001
                msg = "launch failed: %s" % e
            self._q.put(("pinned", ("launch %s" % key, msg, None)))
        threading.Thread(target=go, daemon=True).start()

    def _stop_selected(self):
        name = self._selected
        r = self._run(name) if name else None
        if not r or r["state"] != "running":
            messagebox.showinfo("Stop run", "Select a running run first.")
            return
        others = [x["name"] for x in self._last_status["runs"]
                  if x is not r and x.get("pgid") == r.get("pgid") and x["state"] in ("running", "queued")]
        warn = ""
        if r.get("log") == "phase1.nohup" or others:
            warn = "\n\nThis run belongs to a queue; stopping it stops the whole queue."
        if not messagebox.askyesno("Stop run", "Kill run '%s' (process group %s)?%s" % (name, r.get("pgid"), warn)):
            return
        def go():
            try:
                p = _wsl("cd ~/sts && ./launch_run.sh stop %s" % _shq(name), timeout=60)
                self._q.put(("err", (p.stdout + p.stderr).strip()))
            except Exception as e:  # noqa: BLE001
                self._q.put(("err", "stop failed: %s" % e))
        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------- ledger / inspector
    def _show_text(self, title, text):
        """Pin a result in the log pane (pauses log tailing until another run row is clicked)."""
        self._selected = None
        self._pinned = True
        for w in self._rows.values():
            w["selvar"].set("")
        self.log_title.set(title)
        self._set_log(text)
        self.text.see("1.0")

    def _bg(self, title, cmd, timeout=600, after=None):
        self._show_text(title, "running...\n$ " + cmd)
        def go():
            try:
                p = _wsl("cd ~/sts && " + cmd, timeout=timeout)
                txt = (p.stdout or "") + (("\n" + p.stderr[-1500:]) if p.returncode else "")
            except Exception as e:  # noqa: BLE001
                txt = "failed: %s" % e
            if after is not None:          # post-processing stays off the UI thread
                txt = after(txt)
            self._q.put(("pinned", (title, txt, None)))
        threading.Thread(target=go, daemon=True).start()

    def _compare(self):
        a, b = self.cmp_a.get(), self.cmp_b.get()
        if a and b:
            self._bg("compare %s vs %s" % (a, b),
                     "nice -n 10 python3 ledger.py ingest >/dev/null && python3 ledger.py compare %s %s" % (_shq(a), _shq(b)))

    def _leaderboard(self):
        self._bg("results ledger", "nice -n 10 python3 ledger.py ingest >/dev/null && python3 ledger.py list")

    def _export(self):
        if messagebox.askyesno("Export", "Write docs/experiments/sim_runs.json in C:\\Code\\AscensionAI from the ledger?"):
            self._bg("export to website",
                     "nice -n 10 python3 ledger.py ingest >/dev/null && python3 ledger.py export /mnt/c/Code/AscensionAI/docs/experiments")

    def _inspect(self):
        run, seed = self.cmp_a.get(), self.seed_var.get().strip()
        if not run or not seed.isdigit():
            messagebox.showinfo("Inspect seed", "Pick run A and enter a numeric seed (eval seeds: 10007 + i*7919).")
            return
        def opened(txt):
            try:
                info = json.loads(txt.strip().splitlines()[-1])
                src = info["html"]
                import os
                import tempfile
                dst = os.path.join(tempfile.gettempdir(), os.path.basename(src))
                p = _wsl("cat %s" % _shq(src), timeout=60)
                with open(dst, "w", encoding="utf-8") as fh:
                    fh.write(p.stdout)
                os.startfile(dst)
                return "opened %s\nreplay identical to recorded run: %s  (floor %s, win %s)" % (
                    dst, info.get("identical_to_recorded"), info.get("floor"), info.get("win"))
            except Exception as e:  # noqa: BLE001
                return "inspect failed: %s\n%s" % (e, txt[-1500:])
        self._bg("inspect %s seed %s (replaying with TRACE=1...)" % (run, seed),
                 "nice -n 10 .venv/bin/python inspect_seed.py %s %s" % (_shq(run), seed), timeout=1800, after=opened)

    def close(self):
        self._stop.set()


def _bar_style(state):
    return {"running": "Run", "done": "Done", "stopped": "Stop", "failed": "Fail"}.get(state, "Queue") \
        + ".Horizontal.TProgressbar"


def _shq(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


_TOAST_PS = r"""
$ErrorActionPreference='SilentlyContinue'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$x = New-Object Windows.Data.Xml.Dom.XmlDocument
$x.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$env:T_TITLE</text><text>$env:T_MSG</text></binding></visual></toast>")
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($x))
"""


def _toast(title, msg):
    """Windows toast via the built-in WinRT API (nothing to install); silent no-op on failure."""
    import os
    clean = lambda s: str(s).replace("<", "(").replace(">", ")").replace("&", "+").replace('"', "'")[:200]
    env = dict(os.environ, T_TITLE=clean(title), T_MSG=clean(msg))
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", _TOAST_PS],
                         env=env, creationflags=_NOWIN)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":      # standalone: python sim_tab.py
    root = tk.Tk()
    root.title("AscensionAI - Sim Experiments")
    root.geometry("1100x760")
    ttk.Style().theme_use("clam")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
    tab = SimExperimentsTab(nb)
    root.protocol("WM_DELETE_WINDOW", lambda: (tab.close(), root.destroy()))
    root.mainloop()
