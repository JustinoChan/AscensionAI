"""
AscensionAI Control Panel — desktop GUI for managing RL training.

Double-click to launch. Detects hardware, recommends worker count, starts/stops
STS instances + the offline trainer, and shows live log output.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure we're running inside the project venv so dependencies are available.
# When double-clicked, Windows runs .pyw with the system pythonw.exe which
# won't have psutil/torch. Re-launch ourselves with the venv interpreter.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_VENV_PYTHONW = _ROOT / ".venv" / "Scripts" / "pythonw.exe"
_VENV_PYTHON = _ROOT / ".venv" / "Scripts" / "python.exe"

_current_exe = Path(sys.executable).resolve()
_in_venv = _current_exe == _VENV_PYTHONW.resolve() or _current_exe == _VENV_PYTHON.resolve()

if not _in_venv and _VENV_PYTHONW.exists():
    os.execv(str(_VENV_PYTHONW), [str(_VENV_PYTHONW), str(Path(__file__).resolve())] + sys.argv[1:])

import csv as _csv
import logging
import time
import traceback
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = _ROOT
SCRIPTS = ROOT / "scripts"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STS_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire")
MTS_LAUNCHER = STS_DIR / "mts-launcher.jar"
JAVA_EXE = STS_DIR / "jre" / "bin" / "java.exe"
CONFIG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "ModTheSpire" / "CommunicationMod"
CONFIG_FILE = CONFIG_DIR / "config.properties"

# ---------------------------------------------------------------------------
# File-based debug logger — always writes, survives GUI crashes
# ---------------------------------------------------------------------------
(ROOT / "logs").mkdir(exist_ok=True)
_log_file = ROOT / "logs" / "control_panel_debug.log"
_logger = logging.getLogger("AscensionAI")
_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(str(_log_file), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger.addHandler(_fh)
_logger.info("=" * 60)
_logger.info("AscensionAI Control Panel starting")
_logger.info(f"Python: {sys.executable}")
_logger.info(f"Platform: {platform.platform()}")
_logger.info(f"ROOT: {ROOT}")
_logger.info(f"VENV_PYTHON: {VENV_PYTHON}  exists={VENV_PYTHON.exists()}")
_logger.info(f"JAVA_EXE: {JAVA_EXE}  exists={JAVA_EXE.exists()}")
_logger.info(f"MTS_LAUNCHER: {MTS_LAUNCHER}  exists={MTS_LAUNCHER.exists()}")
_logger.info(f"CONFIG_FILE: {CONFIG_FILE}")
_logger.info(f"STS_DIR: {STS_DIR}  exists={STS_DIR.exists()}")

MODES = {
    "Parallel Workers": "worker",
    "Collect Rollouts (No Training)": "collect",
    "Single-Instance Training": "train",
    "BC \u2192 PPO (End-to-End)": "bc_ppo",
    "Behavior Cloning": "bc",
    "Evaluation (Greedy)": "eval",
    "Game Logger (Passive)": "logger",
}

# (label_text, min, max, default) — None label = spinner hidden for that mode
SPINNER_CONFIG = {
    "worker":  ("Workers:", 1, 8, None),
    "collect": ("Workers:", 1, 8, None),
    "train":   (None, 0, 0, 0),
    "bc_ppo":  ("BC Games:", 10, 200, 50),
    "bc":      (None, 0, 0, 0),
    "eval":    ("Games:", 1, 100, 20),
    "logger":  (None, 0, 0, 0),
}


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def detect_hardware() -> dict:
    _logger.info("Detecting hardware...")
    info = {}

    info["cpu_name"] = _detect_wmi("(Get-CimInstance Win32_Processor).Name") or platform.processor() or "Unknown CPU"
    info["cpu_cores"] = os.cpu_count() or 4
    _logger.info(f"CPU: {info['cpu_name']} ({info['cpu_cores']} cores)")

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["ram_avail_gb"] = round(mem.available / (1024 ** 3), 1)
        _logger.info(f"RAM: {info['ram_total_gb']}GB total, {info['ram_avail_gb']}GB available (psutil)")
    except ImportError:
        info["ram_total_gb"] = 0
        info["ram_avail_gb"] = 0
        _logger.warning("psutil not available — RAM detection disabled")

    info["gpu_name"] = _detect_wmi("(Get-CimInstance Win32_VideoController).Name", skip_virtual=True) or "Unknown GPU"
    _logger.info(f"GPU: {info['gpu_name']}")

    per_instance_mb = 500
    reserve_mb = 4096
    avail_mb = info["ram_avail_gb"] * 1024
    n_ram = max(1, int((avail_mb - reserve_mb) / per_instance_mb))
    n_cpu = max(1, info["cpu_cores"] - 1)
    info["recommended_workers"] = max(1, min(n_ram, n_cpu, 6))
    _logger.info(f"Worker calc: n_ram={n_ram}, n_cpu={n_cpu}, recommended={info['recommended_workers']}")

    return info


def _detect_wmi(ps_expression: str, skip_virtual: bool = False) -> str:
    """Run a PowerShell one-liner to query WMI. Returns first non-empty line or ''."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_expression],
            text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        _logger.debug(f"WMI query '{ps_expression}' returned {len(lines)} line(s): {lines}")
        if skip_virtual:
            skip_keywords = ("virtual", "microsoft basic", "remote", "parsec")
            real = [l for l in lines if not any(k in l.lower() for k in skip_keywords)]
            if real:
                return real[0]
        return lines[0] if lines else ""
    except Exception as e:
        _logger.warning(f"WMI query failed: {ps_expression!r} — {e}")
        return ""


# ---------------------------------------------------------------------------
# Config writing + process management
# ---------------------------------------------------------------------------
def escape_properties_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace(":", "\\:")


def write_config(command: str):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%a %b %d %H:%M:%S %z %Y")
    content = f"#{ts}\nverbose=true\ncommand={command}\nrunAtGameStart=true\n"
    CONFIG_FILE.write_text(content, encoding="ascii")
    _logger.info(f"Config written to {CONFIG_FILE}")
    _logger.debug(f"Config content:\n{content.rstrip()}")


def build_command(mode: str, worker_id: int = 1, games: int = 20, bc_games: int = 50) -> str:
    py = escape_properties_path(VENV_PYTHON)
    root = escape_properties_path(ROOT)
    if mode == "worker":
        cmd = f"{py} {root}/scripts/rollout_worker.py --model models/ppo_sts.pt --out rollouts_shared --id {worker_id}"
    elif mode == "train":
        cmd = f"{py} {root}/scripts/train_ppo.py --save models/ppo_sts.pt --resume models/ppo_sts.pt --save-every 5"
    elif mode == "bc_ppo":
        cmd = f"{py} {root}/scripts/train_bc_ppo.py --bc-games {bc_games} --ppo-games 200 --save models/ppo_sts.pt"
    elif mode == "bc":
        cmd = f"{py} {root}/scripts/behavior_clone.py --games 50 --save models/ppo_sts.pt"
    elif mode == "eval":
        cmd = f"{py} {root}/scripts/eval_model.py --model models/ppo_sts.pt --games {games}"
    elif mode == "logger":
        cmd = f"{py} {root}/scripts/game_logger.py"
    else:
        _logger.error(f"build_command: unknown mode '{mode}'")
        cmd = ""
    _logger.info(f"build_command(mode={mode}, worker_id={worker_id}, games={games}, bc_games={bc_games}) -> {cmd}")
    return cmd


def launch_sts() -> subprocess.Popen:
    args = [str(JAVA_EXE), "-jar", str(MTS_LAUNCHER)]
    _logger.info(f"Launching STS: {args}  cwd={STS_DIR}")
    proc = subprocess.Popen(
        args,
        cwd=str(STS_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    _logger.info(f"STS process spawned: PID {proc.pid}")
    return proc


# ---------------------------------------------------------------------------
# Log tailer thread
# ---------------------------------------------------------------------------
class LogTailer:
    """Tails a log file and pushes new lines to a callback."""

    def __init__(self, path: Path, callback):
        self.path = path
        self.callback = callback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        _logger.debug(f"LogTailer waiting for {self.path} to exist...")
        while not self._stop.is_set():
            if not self.path.exists():
                self._stop.wait(1.0)
                continue
            break

        if self._stop.is_set():
            _logger.debug(f"LogTailer for {self.path} stopped before file appeared")
            return

        _logger.debug(f"LogTailer opening {self.path} (size={self.path.stat().st_size})")
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                while not self._stop.is_set():
                    line = f.readline()
                    if line:
                        self.callback(line.rstrip("\n"))
                    else:
                        self._stop.wait(0.3)
        except Exception as e:
            _logger.error(f"LogTailer error on {self.path}: {e}")
        _logger.debug(f"LogTailer for {self.path} finished")


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class AscensionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AscensionAI Control Panel")
        self.root.geometry("820x740")
        self.root.minsize(700, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.processes: list[subprocess.Popen] = []
        self.trainer_proc: subprocess.Popen | None = None
        self.tailers: list[LogTailer] = []
        self.running = False
        self.worker_launcher_pids: dict[int, int] = {}
        self.worker_sts_pids: dict[int, set[int]] = {}

        self.hw = detect_hardware()

        self._build_ui()
        self._refresh_stats()
        _logger.info(f"GUI initialized — hw={self.hw}")

    def _dbg(self, tab: str, msg: str):
        """Log to file always; show in GUI when verbose is checked."""
        _logger.debug(f"[{tab}] {msg}")
        if getattr(self, "verbose_var", None) and self.verbose_var.get():
            self._append_log(tab, f"[DBG] {msg}")

    # ----- UI construction -----

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        pad = {"padx": 10, "pady": 5}

        # Hardware panel
        hw_frame = ttk.LabelFrame(self.root, text="  Hardware  ", padding=10)
        hw_frame.pack(fill="x", **pad)

        hw_info = (
            f"CPU:  {self.hw['cpu_name']}  ({self.hw['cpu_cores']} logical cores)\n"
            f"RAM:  {self.hw['ram_total_gb']} GB total,  {self.hw['ram_avail_gb']} GB available\n"
            f"GPU:  {self.hw['gpu_name']}\n"
            f"Recommended workers:  {self.hw['recommended_workers']}"
        )
        ttk.Label(hw_frame, text=hw_info, font=("Consolas", 10)).pack(anchor="w")

        # Controls panel
        ctrl_frame = ttk.LabelFrame(self.root, text="  Controls  ", padding=10)
        ctrl_frame.pack(fill="x", **pad)

        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill="x", pady=(0, 8))

        ttk.Label(row1, text="Mode:").pack(side="left", padx=(0, 5))
        self.mode_var = tk.StringVar(value="Parallel Workers")
        mode_combo = ttk.Combobox(
            row1, textvariable=self.mode_var, values=list(MODES.keys()),
            state="readonly", width=24,
        )
        mode_combo.pack(side="left", padx=(0, 20))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.spinner_label_var = tk.StringVar(value="Workers:")
        self.spinner_label_widget = ttk.Label(row1, textvariable=self.spinner_label_var)
        self.spinner_label_widget.pack(side="left", padx=(0, 5))
        self.workers_var = tk.IntVar(value=self.hw["recommended_workers"])
        self.spin_minus = ttk.Button(row1, text="-", width=3, command=self._dec_workers)
        self.spin_minus.pack(side="left")
        self.workers_label = ttk.Label(row1, textvariable=self.workers_var, width=3,
                                       anchor="center", font=("Consolas", 11, "bold"))
        self.workers_label.pack(side="left", padx=2)
        self.spin_plus = ttk.Button(row1, text="+", width=3, command=self._inc_workers)
        self.spin_plus.pack(side="left")

        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill="x")

        self.start_btn = ttk.Button(row2, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=(0, 10))
        self.graceful_btn = ttk.Button(row2, text="Finish && Stop", command=self._graceful_stop, state="disabled")
        self.graceful_btn.pack(side="left", padx=(0, 5))
        self.stop_btn = ttk.Button(row2, text="Stop Now", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))

        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Verbose Logs", variable=self.verbose_var).pack(side="left", padx=(0, 10))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(row2, textvariable=self.status_var, font=("Segoe UI", 10, "italic")).pack(side="left")

        # Progress panel
        stats_frame = ttk.LabelFrame(self.root, text="  Progress  ", padding=10)
        stats_frame.pack(fill="x", **pad)

        self.stats_vars = {
            "train_line": tk.StringVar(value="Training:  no data yet"),
            "eval_line": tk.StringVar(value="Eval:  no data yet"),
        }
        ttk.Label(stats_frame, textvariable=self.stats_vars["train_line"],
                  font=("Consolas", 10)).pack(anchor="w")
        ttk.Label(stats_frame, textvariable=self.stats_vars["eval_line"],
                  font=("Consolas", 10)).pack(anchor="w", pady=(2, 0))

        btn_row = ttk.Frame(stats_frame)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="View Training Plot", command=self._show_plot).pack(side="left")

        # Log panel
        log_frame = ttk.LabelFrame(self.root, text="  Logs  ", padding=5)
        log_frame.pack(fill="both", expand=True, **pad)

        top_row = ttk.Frame(log_frame)
        top_row.pack(fill="x", pady=(0, 3))
        ttk.Button(top_row, text="Clear", command=self._clear_logs).pack(side="right")

        self.log_notebook = ttk.Notebook(log_frame)
        self.log_notebook.pack(fill="both", expand=True)

        self.log_tabs: dict[str, tk.Text] = {}
        self._add_log_tab("All")

    def _add_log_tab(self, name: str) -> tk.Text:
        frame = ttk.Frame(self.log_notebook)
        self.log_notebook.add(frame, text=f"  {name}  ")

        text = tk.Text(frame, wrap="word", font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                       insertbackground="#d4d4d4", selectbackground="#264f78",
                       state="disabled", height=12)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        self.log_tabs[name] = text
        return text

    def _remove_extra_tabs(self):
        keep = {"All"}
        to_remove = [name for name in self.log_tabs if name not in keep]
        for name in to_remove:
            for i in range(self.log_notebook.index("end")):
                if self.log_notebook.tab(i, "text").strip() == name:
                    self.log_notebook.forget(i)
                    break
            del self.log_tabs[name]

    # ----- Spinner controls -----

    def _inc_workers(self):
        mode = MODES.get(self.mode_var.get(), "worker")
        _, _, hi, _ = SPINNER_CONFIG.get(mode, (None, 1, 8, None))
        v = self.workers_var.get()
        if v < hi:
            self.workers_var.set(v + 1)

    def _dec_workers(self):
        mode = MODES.get(self.mode_var.get(), "worker")
        _, lo, _, _ = SPINNER_CONFIG.get(mode, (None, 1, 8, None))
        v = self.workers_var.get()
        if v > max(1, lo):
            self.workers_var.set(v - 1)

    def _on_mode_change(self, event=None):
        mode = MODES.get(self.mode_var.get(), "worker")
        label, lo, hi, default = SPINNER_CONFIG.get(mode, (None, 0, 0, 0))
        if label:
            self.spinner_label_var.set(label)
            self.spin_minus.configure(state="normal")
            self.spin_plus.configure(state="normal")
            if default is not None:
                self.workers_var.set(default)
            elif mode in ("worker", "collect"):
                self.workers_var.set(self.hw["recommended_workers"])
        else:
            self.spinner_label_var.set("")
            self.spin_minus.configure(state="disabled")
            self.spin_plus.configure(state="disabled")

    # ----- Logging -----

    def _append_log(self, tab_name: str, text: str):
        def _do():
            for name in (tab_name, "All"):
                widget = self.log_tabs.get(name)
                if widget is None:
                    continue
                widget.configure(state="normal")
                prefix = f"[{tab_name}] " if name == "All" else ""
                widget.insert("end", prefix + text + "\n")
                widget.see("end")
                widget.configure(state="disabled")
        self.root.after(0, _do)

    def _clear_logs(self):
        for widget in self.log_tabs.values():
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")

    # ----- Progress stats -----

    def _load_training_stats(self) -> dict | None:
        csv_path = ROOT / "logs" / "training_stats.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(_csv.DictReader(f))
        except Exception as e:
            _logger.warning(f"Failed to parse training_stats.csv: {e}")
            return None
        if not rows:
            return None

        floors, rewards, wins, best, updates = [], [], 0, 0, 0
        parse_errors = 0
        for r in rows:
            try:
                fl = float(r.get("final_floor") or 0)
                floors.append(fl)
                if fl > best:
                    best = fl
            except Exception:
                parse_errors += 1
            try:
                rewards.append(float(r.get("total_reward") or 0))
            except Exception:
                parse_errors += 1
            try:
                if int(float(r.get("victory") or 0)):
                    wins += 1
            except Exception:
                parse_errors += 1
            try:
                updates = int(float(r.get("total_updates") or 0))
            except Exception:
                parse_errors += 1
        if parse_errors:
            _logger.debug(f"training_stats.csv: {parse_errors} field parse error(s) "
                          f"across {len(rows)} rows")

        recent_floors = floors[-25:]
        recent_rewards = rewards[-25:]
        return {
            "total": len(rows),
            "wins": wins,
            "win_rate": wins / len(rows) if rows else 0.0,
            "best_floor": int(best),
            "avg_floor": sum(recent_floors) / len(recent_floors) if recent_floors else 0.0,
            "avg_reward": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0,
            "updates": updates,
        }

    def _load_eval_stats(self) -> dict | None:
        csv_path = ROOT / "logs" / "eval_stats.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(_csv.DictReader(f))
        except Exception as e:
            _logger.warning(f"Failed to parse eval_stats.csv: {e}")
            return None
        if not rows:
            return None
        last_tag = rows[-1].get("run", "")
        run_rows = [r for r in rows if r.get("run") == last_tag]
        try:
            wins = sum(1 for r in run_rows if int(float(r.get("victory") or 0)))
            floors = [float(r.get("final_floor") or 0) for r in run_rows]
        except Exception as e:
            _logger.warning(f"Error parsing eval_stats.csv run '{last_tag}': {e}")
            return None
        return {
            "games": len(run_rows),
            "win_rate": wins / len(run_rows) if run_rows else 0.0,
            "avg_floor": sum(floors) / len(floors) if floors else 0.0,
            "run": last_tag,
        }

    def _refresh_stats(self):
        try:
            ts = self._load_training_stats()
            if ts:
                self.stats_vars["train_line"].set(
                    f"Training:  {ts['total']} games  |  "
                    f"Wins: {ts['wins']} ({ts['win_rate']:.0%})  |  "
                    f"Best Floor: {ts['best_floor']}  |  "
                    f"Avg Floor: {ts['avg_floor']:.1f}  |  "
                    f"Avg Reward: {ts['avg_reward']:.1f}  |  "
                    f"Updates: {ts['updates']}"
                )

            es = self._load_eval_stats()
            if es:
                self.stats_vars["eval_line"].set(
                    f"Eval ({es['run']}):  {es['games']} games  |  "
                    f"Win Rate: {es['win_rate']:.0%}  |  "
                    f"Avg Floor: {es['avg_floor']:.1f}"
                )
        except Exception as e:
            _logger.error(f"_refresh_stats error: {e}")
        self.root.after(5000, self._refresh_stats)

    def _show_plot(self):
        csv_path = ROOT / "logs" / "training_stats.csv"
        if not csv_path.exists():
            _logger.debug("_show_plot: no training_stats.csv")
            messagebox.showinfo("No Data", "No training stats yet.\nRun some training games first.")
            return
        out_path = ROOT / "logs" / "training_plot.png"
        plot_cmd = [str(VENV_PYTHON), str(SCRIPTS / "plot_training.py"),
                    "--save", str(out_path)]
        _logger.info(f"Generating plot: {plot_cmd}")
        try:
            result = subprocess.run(
                plot_cmd,
                cwd=str(ROOT), check=True, timeout=30,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            _logger.info(f"Plot generated: {out_path} "
                         f"(size={out_path.stat().st_size} bytes)")
            if result.stderr:
                _logger.debug(f"Plot stderr: {result.stderr.decode(errors='replace')[:500]}")
            os.startfile(str(out_path))
        except Exception as e:
            _logger.error(f"Plot generation failed: {e}")
            messagebox.showerror("Plot Error", f"Failed to generate plot:\n{e}")

    # ----- Start / Stop -----

    def _start(self):
        _logger.info("=" * 40)
        _logger.info("START pressed")
        errors = self._validate()
        if errors:
            _logger.error(f"Validation failed:\n{errors}")
            messagebox.showerror("Cannot Start", errors)
            return

        self.running = True
        self._graceful_stopping = False
        self._current_mode = MODES.get(self.mode_var.get(), "worker")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.graceful_btn.configure(state="normal")
        self._remove_extra_tabs()
        self._clear_logs()

        verbose = self.verbose_var.get()
        os.environ["ASCENSION_VERBOSE"] = "1" if verbose else "0"

        mode = MODES.get(self.mode_var.get(), "worker")
        n = self.workers_var.get()

        model_path = ROOT / "models" / "ppo_sts.pt"
        _logger.info(f"Mode: {mode} | Spinner: {n} | Verbose: {verbose}")
        _logger.info(f"Model checkpoint: {model_path}  exists={model_path.exists()}")
        if model_path.exists():
            _logger.info(f"Model size: {model_path.stat().st_size} bytes, "
                         f"modified: {datetime.fromtimestamp(model_path.stat().st_mtime)}")

        (ROOT / "models").mkdir(exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)

        if mode == "worker":
            (ROOT / "rollouts_shared").mkdir(exist_ok=True)
            rollout_count = len(list((ROOT / "rollouts_shared").glob("*.npz")))
            _logger.info(f"rollouts_shared/ has {rollout_count} existing .npz files")
            self.status_var.set(f"Launching {n} workers + trainer...")
            threading.Thread(target=self._launch_parallel, args=(n,), daemon=True).start()
        elif mode == "collect":
            (ROOT / "rollouts_shared").mkdir(exist_ok=True)
            rollout_count = len(list((ROOT / "rollouts_shared").glob("*.npz")))
            _logger.info(f"rollouts_shared/ has {rollout_count} existing .npz files")
            self.status_var.set(f"Launching {n} collectors (no trainer)...")
            threading.Thread(target=self._launch_parallel,
                             args=(n,), kwargs={"with_trainer": False},
                             daemon=True).start()
        elif mode == "train":
            self.status_var.set("Launching single-instance training...")
            threading.Thread(target=self._launch_single, args=("train",), daemon=True).start()
        elif mode == "bc_ppo":
            self.status_var.set(f"Launching BC\u2192PPO ({n} BC games)...")
            threading.Thread(target=self._launch_single, args=("bc_ppo", n), daemon=True).start()
        elif mode == "bc":
            self.status_var.set("Launching behavior cloning...")
            threading.Thread(target=self._launch_single, args=("bc",), daemon=True).start()
        elif mode == "eval":
            self.status_var.set(f"Launching evaluation ({n} games)...")
            threading.Thread(target=self._launch_single, args=("eval", n), daemon=True).start()
        elif mode == "logger":
            self.status_var.set("Launching passive game logger...")
            threading.Thread(target=self._launch_single, args=("logger",), daemon=True).start()
        _logger.info(f"Launch thread started for mode={mode}")

    def _launch_single(self, mode: str, games: int = 20):
        _logger.info(f"_launch_single: mode={mode}, games={games}")
        try:
            cmd = build_command(mode, games=games, bc_games=games)
            write_config(cmd)

            _mode_info = {
                "train":   ("Training",   "logs/train_debug.log"),
                "bc_ppo":  ("BC\u2192PPO",     "logs/train_bc_ppo_debug.log"),
                "bc":      ("BC",          "logs/bc_debug.log"),
                "eval":    ("Evaluation",  "logs/eval_debug.log"),
                "logger":  ("Logger",      "logs/game_logger_debug.log"),
            }
            log_name, log_file = _mode_info.get(mode, ("Training", "logs/train_debug.log"))
            _logger.info(f"Log tab: {log_name}, log file: {log_file}")

            self.root.after(0, lambda: self._add_log_tab(log_name))
            time.sleep(0.1)

            self._append_log(log_name, f"Config written, launching STS ({mode})...")
            proc = launch_sts()
            self.processes.append(proc)
            self._append_log(log_name, f"STS launched (PID {proc.pid})")
            _logger.info(f"STS launched for {mode}: PID {proc.pid}, total processes tracked: {len(self.processes)}")

            log_path = ROOT / log_file
            _logger.debug(f"Starting LogTailer for {log_path}")
            tailer = LogTailer(log_path, lambda line, n=log_name: self._append_log(n, line))
            tailer.start()
            self.tailers.append(tailer)

            self.root.after(0, lambda: self.status_var.set(f"Running ({mode})"))
            _logger.info(f"_launch_single complete for {mode}")
        except Exception as e:
            _logger.error(f"_launch_single FAILED: {traceback.format_exc()}")
            self._append_log("All", f"ERROR launching {mode}: {e}")

    def _wait_for_ready(self, log_file: Path, tab_name: str, timeout: int = 120) -> bool:
        """Block until a worker's log file contains 'Signaling ready', or timeout."""
        _logger.info(f"_wait_for_ready: {tab_name}, file={log_file}, timeout={timeout}s")
        self._append_log(tab_name, f"Waiting for worker to signal ready (up to {timeout}s)...")
        start = time.time()
        last_status_log = start
        while time.time() - start < timeout:
            if not self.running:
                _logger.info(f"_wait_for_ready: {tab_name} aborted (self.running=False)")
                return False
            try:
                if log_file.exists():
                    text = log_file.read_text(encoding="utf-8", errors="replace")
                    if "Signaling ready" in text:
                        elapsed = time.time() - start
                        _logger.info(f"_wait_for_ready: {tab_name} ready after {elapsed:.1f}s")
                        self._append_log(tab_name, f"Worker signaled ready! ({elapsed:.0f}s)")
                        return True
                    now = time.time()
                    if now - last_status_log >= 15.0:
                        elapsed = now - start
                        lines = len(text.splitlines())
                        _logger.debug(f"_wait_for_ready: {tab_name} still waiting "
                                      f"({elapsed:.0f}s elapsed, log has {lines} lines)")
                        self._dbg(tab_name, f"Still waiting for ready signal... "
                                  f"({elapsed:.0f}s, {lines} log lines)")
                        last_status_log = now
                else:
                    now = time.time()
                    if now - last_status_log >= 15.0:
                        _logger.debug(f"_wait_for_ready: {tab_name} log file doesn't exist yet "
                                      f"({now - start:.0f}s elapsed)")
                        last_status_log = now
            except OSError as e:
                _logger.warning(f"_wait_for_ready: OSError reading {log_file}: {e}")
            time.sleep(1.0)
        elapsed = time.time() - start
        _logger.warning(f"_wait_for_ready: {tab_name} TIMED OUT after {elapsed:.1f}s")
        self._append_log(tab_name, f"Timed out waiting for worker to be ready ({elapsed:.0f}s).")
        return False

    def _launch_parallel(self, n_workers: int, with_trainer: bool = True):
        _logger.info(f"_launch_parallel: n_workers={n_workers}, with_trainer={with_trainer}")
        try:
            self.worker_launcher_pids.clear()
            self.worker_sts_pids.clear()

            for i in range(1, n_workers + 1):
                log_file = ROOT / "logs" / f"worker_{i}_debug.log"
                try:
                    if log_file.exists():
                        _logger.debug(f"Deleting stale log: {log_file} "
                                      f"(size={log_file.stat().st_size})")
                        log_file.unlink()
                except OSError as e:
                    _logger.warning(f"Failed to delete stale log {log_file}: {e}")

            if with_trainer:
                self._append_log("All", "Starting offline trainer...")
                self.root.after(0, lambda: self._add_log_tab("Trainer"))
                time.sleep(0.1)

                trainer_cmd = [
                    str(VENV_PYTHON), str(SCRIPTS / "train_offline.py"),
                    "--model", str(ROOT / "models" / "ppo_sts.pt"),
                    "--data", str(ROOT / "rollouts_shared"),
                    "--delete-consumed",
                ]
                _logger.info(f"Trainer command: {trainer_cmd}")
                self.trainer_proc = subprocess.Popen(
                    trainer_cmd, cwd=str(ROOT),
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                _logger.info(f"Offline trainer started: PID {self.trainer_proc.pid}")
                self._append_log("Trainer", f"Offline trainer started (PID {self.trainer_proc.pid})")

                tailer = LogTailer(ROOT / "logs" / "train_offline_debug.log",
                                   lambda line: self._append_log("Trainer", line))
                tailer.start()
                self.tailers.append(tailer)
            else:
                _logger.info("Collect-only mode: no trainer launched")
                self._append_log("All",
                    "Collect-only mode: no local trainer. "
                    "Rollouts will accumulate in rollouts_shared/ — "
                    "zip and send to the trainer when done.")

            for i in range(1, n_workers + 1):
                if not self.running:
                    _logger.info(f"_launch_parallel: aborted at worker {i} (self.running=False)")
                    break

                tab_name = f"Worker {i}"
                self.root.after(0, lambda tn=tab_name: self._add_log_tab(tn))
                time.sleep(0.1)

                cmd = build_command("worker", worker_id=i)
                write_config(cmd)
                self._append_log(tab_name, f"Config written for worker {i}, launching STS...")
                self._append_log(tab_name, "Click 'Play' in ModTheSpire when it appears.")

                proc = launch_sts()
                self.processes.append(proc)
                self.worker_launcher_pids[i] = proc.pid
                self.worker_sts_pids[i] = set()
                _logger.info(f"Worker {i}: launcher PID={proc.pid}, "
                             f"total processes tracked={len(self.processes)}")
                self._append_log(tab_name, f"STS instance launched (PID {proc.pid})")

                threading.Thread(
                    target=self._capture_sts_pids_for_worker,
                    args=(proc.pid, i),
                    daemon=True,
                ).start()
                _logger.debug(f"Worker {i}: PID capture thread started")

                log_file = ROOT / "logs" / f"worker_{i}_debug.log"
                tailer = LogTailer(log_file, lambda line, n=tab_name: self._append_log(n, line))
                tailer.start()
                self.tailers.append(tailer)

                self.root.after(0, lambda idx=i: self.status_var.set(
                    f"Waiting for worker {idx}/{n_workers} to be ready..."))

                if i < n_workers:
                    ready = self._wait_for_ready(log_file, tab_name, timeout=120)
                    _logger.info(f"Worker {i} ready={ready}, proceeding to next worker")

            if self.running:
                suffix = "workers + trainer" if with_trainer else "collectors (no trainer)"
                _logger.info(f"All {n_workers} workers launched ({suffix})")
                _logger.info(f"Launcher PIDs: {self.worker_launcher_pids}")
                _logger.info(f"STS PIDs (captured so far): {dict(self.worker_sts_pids)}")
                self.root.after(0, lambda: self.status_var.set(
                    f"Running ({n_workers} {suffix})"))
        except Exception as e:
            _logger.error(f"_launch_parallel FAILED: {traceback.format_exc()}")
            self._append_log("All", f"ERROR in parallel launch: {e}")

    def _graceful_stop(self):
        """Wait for all current games to finish, save transitions, then stop."""
        if self._graceful_stopping:
            _logger.debug("_graceful_stop: already in progress, ignoring")
            return
        self._graceful_stopping = True
        self.graceful_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        _logger.info("Graceful stop requested")
        self._append_log("All", "Graceful stop requested — waiting for current games to finish...")
        self.status_var.set("Finishing current games...")
        threading.Thread(target=self._wait_games_then_stop, daemon=True).start()

    def _wait_games_then_stop(self):
        """Background thread: monitor logs for game-end, then call _stop."""
        mode = getattr(self, "_current_mode", "worker")
        _logger.info(f"_wait_games_then_stop: mode={mode}")

        if mode in ("worker", "collect"):
            self._wait_workers_then_stop()
        else:
            self._wait_single_then_stop(mode)

    def _wait_single_then_stop(self, mode: str):
        """Wait for single-instance mode to finish its current game."""
        log_map = {
            "train":   "logs/train_debug.log",
            "bc_ppo":  "logs/train_bc_ppo_debug.log",
            "bc":      "logs/bc_debug.log",
            "eval":    "logs/eval_debug.log",
            "logger":  "logs/game_logger_debug.log",
        }
        log_file = ROOT / log_map.get(mode, "logs/train_debug.log")
        _logger.info(f"_wait_single_then_stop: mode={mode}, log={log_file}, "
                     f"exists={log_file.exists()}")

        baseline_ended = 0
        if log_file.exists():
            try:
                text = log_file.read_text(encoding="utf-8", errors="replace")
                baseline_ended = text.count(" ended")
                _logger.info(f"Baseline ' ended' count: {baseline_ended} "
                             f"(log size={len(text)} bytes)")
            except OSError as e:
                _logger.warning(f"Failed to read log for baseline: {e}")

        self._append_log("All", "Waiting for current game to finish...")

        timeout = 300
        start = time.time()
        finished = False
        last_poll_log = start
        while time.time() - start < timeout:
            if log_file.exists():
                try:
                    text = log_file.read_text(encoding="utf-8", errors="replace")
                    current_ended = text.count(" ended")
                    if current_ended > baseline_ended:
                        elapsed = time.time() - start
                        _logger.info(f"Game ended detected: count {baseline_ended}->{current_ended} "
                                     f"after {elapsed:.1f}s")
                        finished = True
                        break
                    now = time.time()
                    if now - last_poll_log >= 30.0:
                        _logger.debug(f"Still waiting for game end ({now - start:.0f}s, "
                                      f"ended_count={current_ended})")
                        last_poll_log = now
                except OSError as e:
                    _logger.warning(f"OSError polling log: {e}")
            time.sleep(2.0)

        if finished:
            self._append_log("All", "Game finished — stopping.")
        else:
            elapsed = time.time() - start
            _logger.warning(f"_wait_single_then_stop: TIMED OUT after {elapsed:.1f}s")
            self._append_log("All", f"Timed out waiting ({elapsed:.0f}s) — stopping anyway.")

        self.root.after(0, self._stop)

    def _wait_workers_then_stop(self):
        """Wait for parallel workers to finish their current games."""
        game_counts: dict[int, int] = {}
        log_dir = ROOT / "logs"

        for i in range(1, 9):
            log_file = log_dir / f"worker_{i}_debug.log"
            if log_file.exists():
                try:
                    text = log_file.read_text(encoding="utf-8", errors="replace")
                    ended = text.count(" ended:")
                    game_counts[i] = ended
                    _logger.info(f"Worker {i} baseline: {ended} games ended "
                                 f"(log size={len(text)} bytes)")
                except OSError as e:
                    _logger.warning(f"Failed to read worker {i} log: {e}")

        if not game_counts:
            _logger.info("_wait_workers_then_stop: no active worker logs found, stopping immediately")
            self.root.after(0, self._stop)
            return

        _logger.info(f"Monitoring {len(game_counts)} workers: baselines={game_counts}")
        self._append_log("All", f"Monitoring {len(game_counts)} workers for game completion...")

        timeout = 300
        start = time.time()
        workers_done = set()
        last_poll_log = start

        while time.time() - start < timeout:
            all_done = True
            for worker_id, baseline_ended in game_counts.items():
                if worker_id in workers_done:
                    continue
                log_file = log_dir / f"worker_{worker_id}_debug.log"
                try:
                    if log_file.exists():
                        text = log_file.read_text(encoding="utf-8", errors="replace")
                        current_ended = text.count(" ended:")
                        if current_ended > baseline_ended:
                            elapsed = time.time() - start
                            _logger.info(f"Worker {worker_id} game ended: "
                                         f"count {baseline_ended}->{current_ended} "
                                         f"after {elapsed:.1f}s")
                            workers_done.add(worker_id)
                            n_killed = self._stop_single_worker(worker_id)
                            if n_killed:
                                self._append_log(
                                    "All",
                                    f"Worker {worker_id} finished — closed "
                                    f"{n_killed} process(es).")
                            else:
                                self._append_log(
                                    "All",
                                    f"Worker {worker_id} finished (no processes to close).")
                            continue
                except OSError as e:
                    _logger.warning(f"OSError polling worker {worker_id}: {e}")
                all_done = False

            if all_done:
                break

            now = time.time()
            if now - last_poll_log >= 30.0:
                still_waiting = [w for w in game_counts if w not in workers_done]
                _logger.debug(f"Graceful stop: {now - start:.0f}s elapsed, "
                              f"done={sorted(workers_done)}, waiting={still_waiting}")
                last_poll_log = now
            time.sleep(2.0)

        if not all_done:
            still_running = [w for w in game_counts if w not in workers_done]
            elapsed = time.time() - start
            _logger.warning(f"_wait_workers_then_stop: TIMED OUT after {elapsed:.1f}s, "
                            f"still running: {still_running}")
            self._append_log(
                "All",
                f"Timed out waiting for workers {still_running} — stopping anyway.")
        else:
            elapsed = time.time() - start
            _logger.info(f"All workers finished gracefully in {elapsed:.1f}s")
            self._append_log("All", "All workers finished their current games.")

        self.root.after(0, self._stop)

    def _find_pids_for_worker(self, worker_id: int) -> tuple[set[int], set[int]]:
        """Return (java_pids, python_pids) belonging to a specific worker."""
        java_pids: set[int] = set()
        py_pids: set[int] = set()

        try:
            import psutil
        except ImportError:
            _logger.warning("_find_pids_for_worker: psutil not available")
            return java_pids, py_pids

        worker_script = "rollout_worker.py"
        scanned = 0

        for p in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
            try:
                name = (p.info.get("name") or "").lower()
                if not name.startswith("python"):
                    continue
                scanned += 1
                cmdline = " ".join(p.info.get("cmdline") or [])
                if worker_script not in cmdline:
                    continue
                tokens = (p.info.get("cmdline") or [])
                if "--id" not in tokens:
                    continue
                try:
                    if str(tokens[tokens.index("--id") + 1]) != str(worker_id):
                        continue
                except (IndexError, ValueError):
                    continue

                py_pids.add(p.info["pid"])
                ppid = p.info.get("ppid")
                if ppid:
                    try:
                        parent = psutil.Process(ppid)
                        parent_name = parent.name().lower()
                        if parent_name in ("java.exe", "javaw.exe"):
                            java_pids.add(ppid)
                        else:
                            _logger.debug(f"Worker {worker_id} python PID {p.info['pid']} "
                                          f"parent is {parent_name} (PID {ppid}), not java")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        _logger.debug(f"Worker {worker_id}: parent PID {ppid} inaccessible")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        _logger.debug(f"_find_pids_for_worker({worker_id}): scanned {scanned} python procs, "
                      f"found java={java_pids}, py={py_pids}")
        return java_pids, py_pids

    def _capture_sts_pids_for_worker(self, launcher_pid: int, worker_id: int):
        """Best-effort snapshot of worker N's STS java PID once it's running."""
        _logger.debug(f"_capture_sts_pids: worker {worker_id}, launcher PID {launcher_pid}")
        deadline = time.time() + 180.0
        polls = 0
        while time.time() < deadline and self.running:
            java_pids, _ = self._find_pids_for_worker(worker_id)
            polls += 1
            if java_pids:
                self.worker_sts_pids.setdefault(worker_id, set()).update(java_pids)
                _logger.info(f"Captured STS PIDs for worker {worker_id}: {java_pids} "
                             f"(after {polls} polls)")
                return
            time.sleep(2.0)
        _logger.warning(f"_capture_sts_pids: worker {worker_id} timed out after {polls} polls "
                        f"(launcher PID {launcher_pid})")

    def _stop_single_worker(self, worker_id: int) -> int:
        """Kill one worker's STS instance(s) + launcher. Returns count killed."""
        _logger.info(f"_stop_single_worker: worker {worker_id}")
        killed = 0

        java_pids, py_pids = self._find_pids_for_worker(worker_id)
        cached_pids = self.worker_sts_pids.pop(worker_id, set())
        java_pids |= cached_pids
        _logger.info(f"Worker {worker_id} PIDs to kill: java={java_pids} "
                     f"(cached={cached_pids}), py={py_pids}")

        try:
            import psutil
        except ImportError:
            psutil = None

        if psutil is not None:
            for pid in java_pids:
                try:
                    proc = psutil.Process(pid)
                    proc.kill()
                    _logger.info(f"Killed java PID {pid} (worker {worker_id})")
                    killed += 1
                except psutil.NoSuchProcess:
                    _logger.debug(f"Java PID {pid} already dead")
                except psutil.AccessDenied:
                    _logger.warning(f"Access denied killing java PID {pid}")
            for pid in py_pids:
                try:
                    psutil.Process(pid).kill()
                    _logger.info(f"Killed python PID {pid} (worker {worker_id})")
                except psutil.NoSuchProcess:
                    _logger.debug(f"Python PID {pid} already dead")
                except psutil.AccessDenied:
                    _logger.warning(f"Access denied killing python PID {pid}")

        launcher_pid = self.worker_launcher_pids.pop(worker_id, None)
        if launcher_pid:
            tree_ok = self._kill_process_tree(launcher_pid)
            _logger.info(f"Worker {worker_id} launcher tree kill "
                         f"(PID {launcher_pid}): {'ok' if tree_ok else 'failed/dead'}")
            if tree_ok:
                killed += 1

        _logger.info(f"_stop_single_worker: worker {worker_id} done, killed {killed} process(es)")
        return killed

    def _kill_process_tree(self, pid: int) -> bool:
        """Forcefully close a Windows process and all its descendants."""
        _logger.debug(f"_kill_process_tree: PID {pid}")
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                timeout=10, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            stdout = result.stdout.decode(errors="replace").strip() if result.stdout else ""
            stderr = result.stderr.decode(errors="replace").strip() if result.stderr else ""
            ok = result.returncode == 0
            _logger.debug(f"taskkill PID {pid}: rc={result.returncode}, "
                          f"stdout={stdout!r}, stderr={stderr!r}")
            return ok
        except Exception as e:
            _logger.error(f"_kill_process_tree PID {pid} exception: {e}")
            return False

    def _kill_orphan_sts_instances(self) -> int:
        """Sweep for STS java.exe processes our Popen handles no longer cover."""
        _logger.info("Sweeping for orphan STS java processes...")
        try:
            import psutil
        except ImportError:
            _logger.warning("psutil not available for orphan sweep")
            return 0

        killed = 0
        scanned = 0
        sts_dir_lower = str(STS_DIR).lower()
        sts_keywords = ("slaythespire", "desktop-1.0.jar",
                        "mts-launcher", "modthespire")

        for p in psutil.process_iter(["pid", "name", "cwd", "cmdline"]):
            try:
                name = (p.info.get("name") or "").lower()
                if name not in ("java.exe", "javaw.exe"):
                    continue
                scanned += 1

                cwd = (p.info.get("cwd") or "").lower()
                cmdline = " ".join(p.info.get("cmdline") or []).lower()
                matched_cwd = sts_dir_lower in cwd
                matched_kw = any(kw in cmdline for kw in sts_keywords)

                if matched_cwd or matched_kw:
                    reason = f"cwd_match={matched_cwd}, kw_match={matched_kw}"
                    _logger.info(f"Killing orphan java PID {p.info['pid']}: {reason}, "
                                 f"cmdline={cmdline[:200]}")
                    p.kill()
                    killed += 1
                else:
                    _logger.debug(f"Skipping java PID {p.info['pid']}: no STS match "
                                  f"(cwd={cwd[:100]})")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                _logger.debug(f"Orphan sweep: PID access issue: {e}")
            except Exception as e:
                _logger.warning(f"Orphan sweep unexpected error: {e}")

        _logger.info(f"Orphan sweep done: scanned {scanned} java procs, killed {killed}")
        return killed

    def _stop(self):
        _logger.info("=" * 40)
        _logger.info("STOP sequence starting")
        self.running = False
        self._graceful_stopping = False
        self.status_var.set("Stopping...")
        self._append_log("All", "Stopping all processes...")

        _logger.info(f"Stopping {len(self.tailers)} log tailer(s)")
        for tailer in self.tailers:
            tailer.stop()
        self.tailers.clear()

        if self.trainer_proc:
            poll = self.trainer_proc.poll()
            _logger.info(f"Trainer proc PID {self.trainer_proc.pid}: "
                         f"poll={poll} ({'alive' if poll is None else 'exited'})")
            if poll is None:
                try:
                    self.trainer_proc.terminate()
                    self.trainer_proc.wait(timeout=5)
                    _logger.info("Trainer terminated gracefully")
                    self._append_log("All", "Trainer stopped")
                except Exception as e:
                    _logger.warning(f"Trainer terminate failed ({e}), force killing")
                    self.trainer_proc.kill()
                    self._append_log("All", "Trainer force-killed")
        else:
            _logger.debug("No trainer process to stop")
        self.trainer_proc = None

        closed = 0
        _logger.info(f"Killing {len(self.processes)} tracked launcher process(es)")
        for proc in self.processes:
            poll = proc.poll()
            if poll is None:
                _logger.info(f"Killing launcher PID {proc.pid} (still alive)")
                if self._kill_process_tree(proc.pid):
                    closed += 1
            else:
                _logger.debug(f"Launcher PID {proc.pid} already exited (rc={poll})")
        self.processes.clear()

        orphans = self._kill_orphan_sts_instances()
        closed += orphans
        if closed:
            self._append_log("All", f"Closed {closed} Slay the Spire instance(s).")
        _logger.info(f"Stop complete: {closed} process(es) closed "
                     f"({closed - orphans} tracked + {orphans} orphans)")

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.graceful_btn.configure(state="disabled")
        self.status_var.set("Idle")
        self._append_log("All", "All processes stopped.")
        _logger.info("STOP sequence finished")

    # ----- Validation -----

    def _validate(self) -> str:
        _logger.info("Running pre-launch validation...")
        issues = []
        _logger.debug(f"VENV_PYTHON: {VENV_PYTHON}  exists={VENV_PYTHON.exists()}")
        if not VENV_PYTHON.exists():
            issues.append(f"Python venv not found:\n  {VENV_PYTHON}\n  Run: python -m venv .venv && pip install -r requirements.txt")
        _logger.debug(f"JAVA_EXE: {JAVA_EXE}  exists={JAVA_EXE.exists()}")
        if not JAVA_EXE.exists():
            issues.append(f"STS Java runtime not found:\n  {JAVA_EXE}\n  Is Slay the Spire installed via Steam?")
        _logger.debug(f"MTS_LAUNCHER: {MTS_LAUNCHER}  exists={MTS_LAUNCHER.exists()}")
        if not MTS_LAUNCHER.exists():
            issues.append(f"ModTheSpire not found:\n  {MTS_LAUNCHER}\n  Install Mod the Spire via Steam Workshop.")
        if issues:
            _logger.warning(f"Validation failed: {len(issues)} issue(s)")
        else:
            _logger.info("Validation passed")
        return "\n\n".join(issues)

    # ----- Lifecycle -----

    def _on_close(self):
        _logger.info(f"Window close requested (running={self.running})")
        if self.running:
            if not messagebox.askyesno("Quit", "Training is running. Stop all processes and quit?"):
                _logger.info("User cancelled close")
                return
            self._stop()
        _logger.info("AscensionAI Control Panel shutting down")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = AscensionApp()
    app.run()
