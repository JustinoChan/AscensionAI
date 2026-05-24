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
import json
import logging
import re
import shutil
import time
import traceback
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = _ROOT
SCRIPTS = ROOT / "scripts"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STS_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire")
JAVA_EXE = STS_DIR / "jre" / "bin" / "java.exe"
JAVAW_EXE = STS_DIR / "jre" / "bin" / "javaw.exe"
# Use javaw.exe for GUI launches by default so ModTheSpire does not open a
# duplicate console window. Set ASCENSIONAI_SHOW_MTS_CONSOLE=1 to debug the
# raw ModTheSpire console with java.exe.
SHOW_MTS_CONSOLE = os.environ.get("ASCENSIONAI_SHOW_MTS_CONSOLE", "").strip().lower() in {"1", "true", "yes", "on"}
GAME_JAVA_EXE = JAVA_EXE if SHOW_MTS_CONSOLE or not JAVAW_EXE.exists() else JAVAW_EXE

# Use the real Steam Workshop ModTheSpire.jar when available. The common
# SlayTheSpire\mts-launcher.jar wrapper can still show the launcher even when
# --skip-launcher / --profile are passed, while ModTheSpire.jar accepts those
# flags directly. Override with ASCENSIONAI_MTS_JAR if your Workshop path differs.
_DEFAULT_MTS_JAR = Path(
    r"C:\Program Files (x86)\Steam\steamapps\workshop\content\646570\1605060445\ModTheSpire.jar"
)
_MTS_JAR_ENV = os.environ.get("ASCENSIONAI_MTS_JAR", "").strip()
if _MTS_JAR_ENV:
    MTS_LAUNCHER = Path(_MTS_JAR_ENV)
elif _DEFAULT_MTS_JAR.exists():
    MTS_LAUNCHER = _DEFAULT_MTS_JAR
else:
    # Fallback for non-Workshop installs. This may show the launcher on some setups.
    MTS_LAUNCHER = STS_DIR / "mts-launcher.jar"

CONFIG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "ModTheSpire" / "CommunicationMod"
CONFIG_FILE = CONFIG_DIR / "config.properties"

# Safety: do not synthesize global mouse/keyboard input.
# The ModTheSpire auto-Play fallback used SetForegroundWindow, keybd_event,
# SetCursorPos, and mouse_event. With multiple workers/restarts, that can steal
# focus, teleport the cursor, and click outside the game. Keep this False for
# overnight runs.
AUTO_PRESS_MTS_PLAY = False

# Optional deterministic launcher controls. Default to your saved ModTheSpire
# profile so every GUI launch mode skips the launcher and goes straight into STS.
# Override with ASCENSIONAI_MTS_PROFILE="" to disable profile launching, or use
# ASCENSIONAI_MTS_MODS="basemod,CommunicationMod,SuperFastMode" for exact mod IDs.
MTS_PROFILE = os.environ.get("ASCENSIONAI_MTS_PROFILE", "AscensionAI").strip()
MTS_MODS = os.environ.get("ASCENSIONAI_MTS_MODS", "").strip()

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
_logger.info(f"JAVAW_EXE: {JAVAW_EXE}  exists={JAVAW_EXE.exists()}")
_logger.info(f"GAME_JAVA_EXE: {GAME_JAVA_EXE}  exists={GAME_JAVA_EXE.exists()} show_console={SHOW_MTS_CONSOLE}")
_logger.info(f"MTS_LAUNCHER: {MTS_LAUNCHER}  exists={MTS_LAUNCHER.exists()}")
_logger.info(f"CONFIG_FILE: {CONFIG_FILE}")
_logger.info(f"STS_DIR: {STS_DIR}  exists={STS_DIR.exists()}")

MODES = {
    "Parallel Workers": "worker",
    "Collect Rollouts (No Training)": "collect",
    "Consume Rollouts (Trainer Only)": "consume",
    "Single-Instance Training": "train",
    "BC \u2192 PPO (End-to-End)": "bc_ppo",
    "Behavior Cloning": "bc",
    "BC Demo Collection": "bc_collect",
    "BC Train From Demos": "bc_train",
    "Evaluation (Greedy)": "eval",
    "Evaluate on Seed Set": "eval_set",
    "Game Logger (Passive)": "logger",
    "Play Game (No AI)": "play",
}

# (label_text, min, max, default) — None label = spinner hidden for that mode
SPINNER_CONFIG = {
    "worker":  ("Workers:", 1, 8, None),
    "collect": ("Workers:", 1, 8, None),
    "consume": (None, 0, 0, 0),
    "train":   (None, 0, 0, 0),
    "bc_ppo":  ("BC Games:", 10, 1000, 400),
    "bc":      ("BC Games:", 10, 1000, 400),
    "bc_collect": ("BC Games:", 10, 5000, 400),
    "bc_train": (None, 0, 0, 0),
    "eval":    ("Games:", 1, 100, 20),
    "eval_set": ("Games:", 1, 1000, 200),
    "logger":  (None, 0, 0, 0),
    "play":    (None, 0, 0, 0),
}

EVAL_POLICY_OPTIONS = {
    "PPO Current only": "ppo_current",
    "Full comparison": "full",
    "BC only": "bc",
    "Heuristic only": "heuristic",
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


def write_config(command: str | None = None, *, verbose: bool = True):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%a %b %d %H:%M:%S %z %Y")
    verbose_str = "true" if verbose else "false"
    if command:
        content = f"#{ts}\nverbose={verbose_str}\ncommand={command}\nrunAtGameStart=true\n"
    else:
        content = f"#{ts}\nverbose={verbose_str}\n"
    CONFIG_FILE.write_text(content, encoding="ascii")
    _logger.info(f"Config written to {CONFIG_FILE}")
    _logger.debug(f"Config content:\n{content.rstrip()}")


def build_command(mode: str, worker_id: int = 1, games: int = 20,
                  bc_games: int = 400, ppo_games: int = 200,
                  bc_epochs: int = 50, ent_coef: float = 0.001,
                  worker_games: int = 0, verbose: bool = False,
                  restart_every: int = 0) -> str:
    py = escape_properties_path(VENV_PYTHON)
    root = escape_properties_path(ROOT)
    verbose_flag = " --verbose" if verbose else ""
    if mode == "worker":
        cmd = (
            f"{py} {root}/scripts/rollout_worker.py "
            f"--model models/ppo_sts.pt --out rollouts_shared --id {worker_id}"
        )
        if int(worker_games) > 0:
            cmd += f" --games {int(worker_games)}"
        if int(restart_every) > 0:
            cmd += f" --restart-every {int(restart_every)}"
        cmd += verbose_flag
    elif mode == "train":
        cmd = f"{py} {root}/scripts/train_ppo.py --save models/ppo_sts.pt --resume models/ppo_sts.pt --save-every 5 --ent-coef {ent_coef}{verbose_flag}"
    elif mode == "bc_ppo":
        cmd = (
            f"{py} {root}/scripts/train_bc_ppo.py --bc-games {bc_games} "
            f"--bc-epochs {bc_epochs} --bc-lr 5e-4 "
            f"--ppo-games {ppo_games} --save models/ppo_sts.pt "
            f"--ent-start {ent_coef}{verbose_flag}"
        )
    elif mode == "bc":
        cmd = (
            f"{py} {root}/scripts/behavior_clone.py --games {bc_games} "
            f"--save models/ppo_sts_bc.pt --epochs {bc_epochs} "
            f"--lr 5e-4 --batch-size 256 --val-split 0.10 "
            f"--patience 12 --weight-decay 1e-5 --label-smoothing 0.02{verbose_flag}"
        )
    elif mode == "eval":
        cmd = f"{py} {root}/scripts/eval_model.py --model models/ppo_sts.pt --games {games}{verbose_flag}"
    elif mode == "logger":
        cmd = f"{py} {root}/scripts/game_logger.py{verbose_flag}"
    else:
        _logger.error(f"build_command: unknown mode '{mode}'")
        cmd = ""
    _logger.info(f"build_command(mode={mode}, worker_id={worker_id}, games={games}, "
                 f"bc_games={bc_games}, bc_epochs={bc_epochs}, ppo_games={ppo_games}, "
                 f"ent_coef={ent_coef}, worker_games={worker_games}, "
                 f"verbose={verbose}) -> {cmd}")
    return cmd


def build_bc_collect_command(worker_id: int, games: int, *,
                             demo_save: str, checkpoint: str,
                             verbose: bool = False) -> str:
    """Build a CommunicationMod command for one parallel BC demo collector."""
    py = escape_properties_path(VENV_PYTHON)
    root = escape_properties_path(ROOT)
    verbose_flag = " --verbose" if verbose else ""
    demo_arg = str(demo_save).replace("\\", "/")
    checkpoint_arg = str(checkpoint).replace("\\", "/")
    cmd = (
        f"{py} {root}/scripts/behavior_clone.py --games {int(games)} "
        f"--save models/ppo_sts_bc.pt --collect-only "
        f"--worker-id {int(worker_id)} "
        f"--demo-save {demo_arg} "
        f"--bc-checkpoint {checkpoint_arg}{verbose_flag}"
    )
    _logger.info(
        "build_bc_collect_command("
        f"worker_id={worker_id}, games={games}, demo_save={demo_save}, "
        f"checkpoint={checkpoint}, verbose={verbose}) -> {cmd}"
    )
    return cmd


def build_eval_command(*, policy: str, games: int, seed_file: str, run_tag: str,
                       model: str | None = None, top_actions: int = 0,
                       verbose: bool = False, resume_run: bool = False,
                       log_file: str | None = None,
                       restart_every: int = 0) -> str:
    """Build a CommunicationMod command for one fixed-seed eval run."""
    py = escape_properties_path(VENV_PYTHON)
    root = escape_properties_path(ROOT)
    verbose_flag = " --verbose" if verbose else ""
    safe_seed_file = str(seed_file).replace("\\", "/")
    cmd = (
        f"{py} {root}/scripts/eval_model.py "
        f"--games {int(games)} "
        f"--seed-file {safe_seed_file} "
        f"--run-tag {run_tag}"
    )
    if resume_run:
        cmd += " --resume-run"
    if int(restart_every) > 0:
        cmd += f" --restart-every {int(restart_every)}"
    if log_file:
        safe_log_file = str(log_file).replace("\\", "/")
        cmd += f" --log-file {safe_log_file}"
    if policy == "heuristic":
        cmd += " --policy heuristic"
    else:
        safe_model = str(model or "models/ppo_sts.pt").replace("\\", "/")
        cmd += f" --model {safe_model}"
        if int(top_actions) > 0:
            cmd += f" --top-actions {int(top_actions)}"
    cmd += verbose_flag
    _logger.info(
        "build_eval_command("
        f"policy={policy}, games={games}, seed_file={seed_file}, "
        f"run_tag={run_tag}, model={model}, top_actions={top_actions}, "
        f"verbose={verbose}, resume_run={resume_run}, log_file={log_file}) -> {cmd}"
    )
    return cmd


JVM_HEAP_OPTIONS = ["Default", "512m", "1g", "1536m", "2g", "2560m", "3g", "4g"]


def launch_sts(skip_launcher: bool = False,
               jvm_heap: str = "Default") -> subprocess.Popen:
    args = [str(GAME_JAVA_EXE)]
    if jvm_heap and jvm_heap != "Default":
        args.append(f"-Xmx{jvm_heap}")
    args.extend(["-jar", str(MTS_LAUNCHER)])

    # Prefer ModTheSpire's real CLI flags over any GUI automation.
    # --mods implies launcher skipping on supported ModTheSpire versions.
    # If a profile is configured, use it for every launch mode, not only worker
    # relaunches, so the GUI never pauses on the ModTheSpire menu.
    if MTS_MODS:
        args.extend(["--mods", MTS_MODS])
    elif MTS_PROFILE:
        args.extend(["--skip-launcher", "--profile", MTS_PROFILE])
    elif skip_launcher:
        args.append("--skip-launcher")

    _logger.info(f"Launching STS: {args}  cwd={STS_DIR}")
    popen_kwargs = {
        "cwd": str(STS_DIR),
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
    }

    # The GUI already tails AscensionAI logs. Hiding ModTheSpire's raw console
    # prevents the extra cmd window that duplicates the launcher output.
    if not SHOW_MTS_CONSOLE:
        popen_kwargs["creationflags"] |= subprocess.CREATE_NO_WINDOW
        popen_kwargs["stdin"] = subprocess.DEVNULL
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.DEVNULL

    proc = subprocess.Popen(args, **popen_kwargs)
    _logger.info(f"STS process spawned: PID {proc.pid}")

    if skip_launcher and AUTO_PRESS_MTS_PLAY:
        threading.Thread(
            target=_auto_press_mts_play,
            args=(proc.pid,),
            daemon=True,
        ).start()
    elif skip_launcher:
        _logger.debug(
            "ModTheSpire auto-Play fallback disabled; no synthetic "
            "mouse/keyboard input will be sent"
        )
    return proc


def _get_window_title(user32, hwnd) -> str:
    """Return a Win32 top-level window title, or an empty string on failure."""
    try:
        import ctypes
        from ctypes import wintypes
        length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
        return buf.value
    except Exception:
        return ""


def _find_mts_launcher_windows() -> list[int]:
    """Find visible ModTheSpire launcher windows for the auto-Play fallback."""
    if platform.system() != "Windows":
        return []
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            title = _get_window_title(user32, hwnd)
            if title.startswith("ModTheSpire"):
                found.append(int(hwnd))
            return True

        user32.EnumWindows(enum_proc, 0)
        return found
    except Exception as e:
        _logger.debug(f"Could not enumerate ModTheSpire windows: {e}")
        return []


def _click_mts_play(hwnd: int) -> bool:
    """Disabled safety stub: never send global keyboard or mouse input."""
    _logger.warning(
        "ModTheSpire auto-Play was requested but is disabled to prevent "
        "cursor teleporting, focus stealing, and accidental clicks."
    )
    return False


def _auto_press_mts_play(proc_pid: int, timeout_sec: float = 90.0):
    """Disabled: do not interact with the desktop to press ModTheSpire Play.

    The RL workers communicate through CommunicationMod and do not need mouse or
    keyboard control. If --skip-launcher fails, the GUI monitor should kill/restart
    the stuck process instead of sending global input.
    """
    _logger.warning(
        f"ModTheSpire launcher fallback disabled for PID {proc_pid}; "
        "no mouse/keyboard input was sent"
    )
    return


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
        self.root.geometry("900x820")
        self.root.minsize(760, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.processes: list[subprocess.Popen] = []
        self.trainer_proc: subprocess.Popen | None = None
        self._trainer_cmd: list[str] | None = None
        self._with_trainer = False
        self.tailers: list[LogTailer] = []
        self.running = False
        self.worker_launcher_pids: dict[int, int] = {}
        self.worker_sts_pids: dict[int, set[int]] = {}
        self._worker_commands: dict[int, str] = {}
        self._worker_modes: dict[int, str] = {}
        self._worker_demo_paths: dict[int, Path] = {}
        self._worker_checkpoint_paths: dict[int, Path] = {}
        self._worker_restarts: dict[int, int] = {}
        self._worker_launch_time: dict[int, float] = {}
        self._worker_game_targets: dict[int, int] = {}
        self._worker_games_at_launch: dict[int, int] = {}
        self._parallel_launch_started_at: float = 0.0
        self._n_workers: int = 0
        self._eval_set_started_at: float = 0.0
        self._ent_value = 0.001
        self._ent_manual_override = False
        self._ent_updating_ui = False
        self._latest_trainer_ent_coef: float | None = None
        self._latest_normalized_entropy: float | None = None
        self._latest_auto_tune_action = ""

        self.hw = detect_hardware()

        self._build_ui()
        self._on_mode_change()
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
        self.workers_entry = ttk.Entry(row1, textvariable=self.workers_var, width=6,
                                       justify="center", font=("Consolas", 11, "bold"))
        self.workers_entry.pack(side="left", padx=2)
        self.workers_entry.bind("<Return>", lambda _e: self._normalize_spinner_value())
        self.workers_entry.bind("<FocusOut>", lambda _e: self._normalize_spinner_value())
        self.spin_plus = ttk.Button(row1, text="+", width=3, command=self._inc_workers)
        self.spin_plus.pack(side="left")
        self.spin_step_var = tk.StringVar(value="")
        self.spin_step_label = ttk.Label(row1, textvariable=self.spin_step_var,
                                         foreground="gray")
        self.spin_step_label.pack(side="left", padx=(6, 0))

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

        self.use_bc_var = tk.BooleanVar(value=False)
        self.use_bc_chk = ttk.Checkbutton(row2, text="Use BC Checkpoint", variable=self.use_bc_var)
        self.use_bc_chk.pack(side="left", padx=(0, 10))

        self.gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="GPU Trainer", variable=self.gpu_var).pack(side="left", padx=(0, 10))
        
        self.auto_tune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Auto-Tune", variable=self.auto_tune_var).pack(side="left", padx=(0, 10))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(row2, textvariable=self.status_var, font=("Segoe UI", 10, "italic")).pack(side="left")

        # Model selection + archive controls (always visible).
        self.model_row = ttk.Frame(ctrl_frame)
        self.model_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.model_row, text="Model:").pack(side="left", padx=(0, 5))
        self._archive_options: dict[str, Path | None] = {}
        self.model_choice_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.model_row,
            textvariable=self.model_choice_var,
            state="readonly",
            width=52,
        )
        self.model_combo.pack(side="left", padx=(0, 8))
        ttk.Button(self.model_row, text="Refresh",
                   command=self._refresh_models_and_stats).pack(side="left", padx=(0, 6))
        ttk.Button(self.model_row, text="Archive Current...",
                   command=self._prompt_archive_now).pack(side="left", padx=(0, 6))
        self._refresh_archive_list()

        # Entropy control row (visible only for modes with a trainer)
        self.ent_row = ttk.Frame(ctrl_frame)
        self.ent_row.pack(fill="x", pady=(6, 0))

        self.ent_label = ttk.Label(self.ent_row, text="Start Entropy:")
        self.ent_label.pack(side="left", padx=(0, 5))
        self.ent_var = tk.StringVar(value=f"{self._ent_value:.6f}")
        self.ent_minus = ttk.Button(self.ent_row, text="-", width=3, command=self._dec_ent)
        self.ent_minus.pack(side="left")
        self.ent_display = ttk.Label(self.ent_row, textvariable=self.ent_var, width=9,
                                     anchor="center", font=("Consolas", 11, "bold"))
        self.ent_display.pack(side="left", padx=2)
        self.ent_plus = ttk.Button(self.ent_row, text="+", width=3, command=self._inc_ent)
        self.ent_plus.pack(side="left")
        self.ent_scale = ttk.Scale(
            self.ent_row,
            from_=0.0,
            to=0.01,
            orient="horizontal",
            length=160,
            command=self._on_ent_scale,
        )
        self._ent_updating_ui = True
        try:
            self.ent_scale.set(self._ent_value)
        finally:
            self._ent_updating_ui = False
        self.ent_scale.pack(side="left", padx=(8, 0))
        self.ent_sync_btn = ttk.Button(
            self.ent_row,
            text="Sync latest",
            command=lambda: self._sync_ent_from_latest(manual=True),
        )
        self.ent_sync_btn.pack(side="left", padx=(8, 0))
        self.ent_current_var = tk.StringVar(value="Current: no trainer entropy yet")
        ttk.Label(self.ent_row, textvariable=self.ent_current_var,
                  font=("Consolas", 9)).pack(side="left", padx=(8, 0))

        self.ent_help = ttk.Label(self.ent_row, text="(?)", foreground="gray",
                                  font=("Segoe UI", 10, "bold"), cursor="hand2")
        self.ent_help.pack(side="left", padx=(6, 0))
        self._ent_tooltip = None
        self.ent_help.bind("<Enter>", self._show_ent_tooltip)
        self.ent_help.bind("<Leave>", self._hide_ent_tooltip)

        # Behavior-cloning training controls.
        self.bc_epochs_row = ttk.Frame(ctrl_frame)
        self.bc_epochs_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.bc_epochs_row, text="BC Epochs:").pack(side="left", padx=(0, 5))
        self.bc_epochs_var = tk.IntVar(value=50)
        self.bc_epochs_spin = ttk.Spinbox(
            self.bc_epochs_row,
            from_=1,
            to=200,
            increment=5,
            textvariable=self.bc_epochs_var,
            width=7,
        )
        self.bc_epochs_spin.pack(side="left")
        ttk.Label(self.bc_epochs_row, text="default 50; early stopping may stop sooner").pack(side="left", padx=(8, 0))

        # Parallel BC collection controls.
        self.bc_workers_row = ttk.Frame(ctrl_frame)
        self.bc_workers_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.bc_workers_row, text="BC Workers:").pack(side="left", padx=(0, 5))
        self.bc_workers_var = tk.IntVar(value=self.hw["recommended_workers"])
        self.bc_workers_spin = ttk.Spinbox(
            self.bc_workers_row,
            from_=1,
            to=8,
            increment=1,
            textvariable=self.bc_workers_var,
            width=7,
        )
        self.bc_workers_spin.pack(side="left")
        ttk.Label(self.bc_workers_row, text="total games are split across worker slots").pack(side="left", padx=(8, 0))

        self.worker_ids_row = ttk.Frame(ctrl_frame)
        self.worker_ids_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.worker_ids_row, text="Launch Worker IDs:").pack(side="left", padx=(0, 5))
        self.worker_ids_var = tk.StringVar(value="")
        self.worker_ids_entry = ttk.Entry(self.worker_ids_row, textvariable=self.worker_ids_var, width=18)
        self.worker_ids_entry.pack(side="left")
        ttk.Label(self.worker_ids_row, text="blank = all; use 2 or 1,3 to relaunch specific slots").pack(side="left", padx=(8, 0))

        # BC -> PPO run-length controls.
        self.ppo_games_row = ttk.Frame(ctrl_frame)
        self.ppo_games_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.ppo_games_row, text="PPO Games After BC:").pack(side="left", padx=(0, 5))
        self.ppo_games_var = tk.IntVar(value=200)
        self.ppo_games_spin = ttk.Spinbox(
            self.ppo_games_row,
            from_=0,
            to=5000,
            increment=25,
            textvariable=self.ppo_games_var,
            width=7,
        )
        self.ppo_games_spin.pack(side="left")
        ttk.Label(self.ppo_games_row, text="0 = keep running").pack(side="left", padx=(8, 0))

        # Parallel Workers game-count target (total across slots).
        self.ppo_workers_games_row = ttk.Frame(ctrl_frame)
        self.ppo_workers_games_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.ppo_workers_games_row, text="Parallel PPO Games:").pack(side="left", padx=(0, 5))
        self.ppo_workers_games_var = tk.IntVar(value=0)
        self.ppo_workers_games_spin = ttk.Spinbox(
            self.ppo_workers_games_row,
            from_=0,
            to=100000,
            increment=25,
            textvariable=self.ppo_workers_games_var,
            width=8,
        )
        self.ppo_workers_games_spin.pack(side="left")
        ttk.Label(
            self.ppo_workers_games_row,
            text="total across all worker slots; 0 = keep running",
        ).pack(side="left", padx=(8, 0))

        # Restart STS every N games (RAM leak mitigation)
        self.restart_sts_row = ttk.Frame(ctrl_frame)
        self.restart_sts_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.restart_sts_row, text="Restart STS Every:").pack(side="left", padx=(0, 5))
        self.restart_sts_var = tk.IntVar(value=0)
        self.restart_sts_spin = ttk.Spinbox(
            self.restart_sts_row,
            from_=0, to=10000, increment=25,
            textvariable=self.restart_sts_var, width=8,
        )
        self.restart_sts_spin.pack(side="left")
        ttk.Label(
            self.restart_sts_row,
            text="games per instance; 0 = disabled",
        ).pack(side="left", padx=(8, 0))

        # JVM heap cap per STS instance (RAM management)
        self.jvm_heap_row = ttk.Frame(ctrl_frame)
        self.jvm_heap_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.jvm_heap_row, text="STS RAM Limit:").pack(side="left", padx=(0, 5))
        self.jvm_heap_var = tk.StringVar(value="Default")
        self.jvm_heap_combo = ttk.Combobox(
            self.jvm_heap_row,
            textvariable=self.jvm_heap_var,
            values=JVM_HEAP_OPTIONS,
            state="readonly",
            width=10,
        )
        self.jvm_heap_combo.pack(side="left")
        ttk.Label(
            self.jvm_heap_row,
            text="max heap per Java instance (-Xmx)",
        ).pack(side="left", padx=(8, 0))

        # Fixed-seed evaluation controls. Visible only for Evaluate on Seed Set.
        self.eval_policy_row = ttk.Frame(ctrl_frame)
        self.eval_policy_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.eval_policy_row, text="Eval Policy:").pack(side="left", padx=(0, 5))
        self.eval_policy_var = tk.StringVar(value="PPO Current only")
        self.eval_policy_combo = ttk.Combobox(
            self.eval_policy_row,
            textvariable=self.eval_policy_var,
            values=list(EVAL_POLICY_OPTIONS.keys()),
            state="readonly",
            width=22,
        )
        self.eval_policy_combo.pack(side="left", padx=(0, 8))
        ttk.Label(
            self.eval_policy_row,
            text="use PPO-only for routine checkpoint checks",
        ).pack(side="left")

        self.eval_seed_row = ttk.Frame(ctrl_frame)
        self.eval_seed_row.pack(fill="x", pady=(6, 0))
        ttk.Label(self.eval_seed_row, text="Seed File:").pack(side="left", padx=(0, 5))
        self.eval_seed_file_var = tk.StringVar(value="seeds/eval_200.txt")
        self.eval_seed_combo = ttk.Combobox(
            self.eval_seed_row,
            textvariable=self.eval_seed_file_var,
            values=self._seed_file_options(),
            width=34,
        )
        self.eval_seed_combo.pack(side="left", padx=(0, 6))
        ttk.Button(self.eval_seed_row, text="Refresh", command=self._refresh_seed_file_options).pack(side="left", padx=(0, 8))
        self.eval_generate_seed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.eval_seed_row,
            text="Generate/overwrite before run",
            variable=self.eval_generate_seed_var,
        ).pack(side="left")

        # Progress panel
        stats_frame = ttk.LabelFrame(self.root, text="  Progress  ", padding=10)
        stats_frame.pack(fill="x", **pad)

        self.stats_vars = {
            "train_line": tk.StringVar(value="Training:  no data yet"),
            "bc_line": tk.StringVar(value="BC:        no baseline yet"),
            "detail_line": tk.StringVar(value=""),
            "combat_line": tk.StringVar(value=""),
            "eval_line": tk.StringVar(value="Eval:  no data yet"),
        }
        ttk.Label(stats_frame, textvariable=self.stats_vars["train_line"],
                  font=("Consolas", 10)).pack(anchor="w")
        ttk.Label(stats_frame, textvariable=self.stats_vars["bc_line"],
                  font=("Consolas", 10)).pack(anchor="w", pady=(2, 0))
        ttk.Label(stats_frame, textvariable=self.stats_vars["detail_line"],
                  font=("Consolas", 10)).pack(anchor="w", pady=(2, 0))
        ttk.Label(stats_frame, textvariable=self.stats_vars["combat_line"],
                  font=("Consolas", 10)).pack(anchor="w", pady=(2, 0))
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

    def _spinner_bounds(self):
        mode = MODES.get(self.mode_var.get(), "worker")
        _label, lo, hi, default = SPINNER_CONFIG.get(mode, (None, 1, 8, 1))
        return mode, int(lo), int(hi), default

    def _spinner_step(self, mode: str) -> int:
        """Return the +/- step for the main numeric control.

        Worker counts should move one at a time, but game-count modes become
        painful with a step of 1.  The entry box remains editable for exact
        values like 8 games.
        """
        if mode in ("bc_ppo", "bc", "bc_collect"):
            return 10
        if mode == "eval":
            return 5
        if mode == "eval_set":
            return 25
        return 1

    def _normalize_spinner_value(self) -> int:
        mode, lo, hi, default = self._spinner_bounds()
        try:
            value = int(self.workers_var.get())
        except (tk.TclError, ValueError):
            value = int(default if default is not None else max(1, lo))
        lo = max(1, lo)
        value = max(lo, min(hi, value))
        self.workers_var.set(value)
        return value

    def _update_spinner_step_hint(self):
        mode = MODES.get(self.mode_var.get(), "worker")
        step = self._spinner_step(mode)
        if mode in ("bc_ppo", "bc", "bc_collect", "eval", "eval_set"):
            self.spin_step_var.set(f"+/- {step}; type exact value")
        else:
            self.spin_step_var.set("")

    def _inc_workers(self):
        mode, _lo, hi, _default = self._spinner_bounds()
        step = self._spinner_step(mode)
        v = self._normalize_spinner_value()
        if v < hi:
            self.workers_var.set(min(hi, v + step))

    def _dec_workers(self):
        mode, lo, _hi, _default = self._spinner_bounds()
        step = self._spinner_step(mode)
        v = self._normalize_spinner_value()
        floor = max(1, lo)
        if v > floor:
            self.workers_var.set(max(floor, v - step))

    def _get_ppo_games(self) -> int:
        try:
            value = int(self.ppo_games_var.get())
        except (tk.TclError, ValueError):
            value = 200
        value = max(0, min(5000, value))
        self.ppo_games_var.set(value)
        return value

    def _get_ppo_workers_games(self) -> int:
        try:
            value = int(self.ppo_workers_games_var.get())
        except (tk.TclError, ValueError):
            value = 0
        value = max(0, min(100000, value))
        self.ppo_workers_games_var.set(value)
        return value

    def _get_restart_sts_every(self) -> int:
        try:
            value = int(self.restart_sts_var.get())
        except (tk.TclError, ValueError):
            value = 0
        value = max(0, min(10000, value))
        self.restart_sts_var.set(value)
        return value

    def _get_jvm_heap(self) -> str:
        try:
            return str(self.jvm_heap_var.get()).strip()
        except (tk.TclError, ValueError):
            return "Default"

    def _compute_worker_game_targets(self, total_games: int,
                                     n_workers: int) -> dict[int, int]:
        """Split a total game goal across worker slots 1..n_workers.

        Example: total=503, n_workers=4 -> {1: 126, 2: 126, 3: 126, 4: 125}.
        Returns an empty dict when no target is set (workers run until stopped).
        """
        total_games = int(total_games)
        n_workers = int(n_workers)
        if total_games <= 0 or n_workers <= 0:
            return {}
        base, remainder = divmod(total_games, n_workers)
        return {slot: base + (1 if slot <= remainder else 0)
                for slot in range(1, n_workers + 1)}

    def _count_worker_completed_games(self, worker_id: int) -> int:
        """Count games logged by a worker in training_stats.csv since the current run started."""
        stats_path = ROOT / "logs" / "training_stats.csv"
        if not stats_path.exists():
            return 0
        run_start = getattr(self, "_parallel_launch_started_at", None)
        if run_start is None:
            return 0
        from datetime import datetime as _dt
        run_start_iso = _dt.fromtimestamp(run_start).isoformat()
        count = 0
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                import csv as _csv
                reader = _csv.DictReader(f)
                wid_str = str(worker_id)
                for row in reader:
                    if row.get("worker") == wid_str and row.get("game"):
                        ts = row.get("timestamp", "")
                        if ts >= run_start_iso:
                            count += 1
        except Exception:
            pass
        return count

    def _get_bc_epochs(self) -> int:
        try:
            value = int(self.bc_epochs_var.get())
        except (tk.TclError, ValueError):
            value = 50
        value = max(1, min(200, value))
        self.bc_epochs_var.set(value)
        return value

    def _get_bc_workers(self) -> int:
        try:
            value = int(self.bc_workers_var.get())
        except (tk.TclError, ValueError):
            value = int(self.hw.get("recommended_workers", 1) or 1)
        value = max(1, min(8, value))
        self.bc_workers_var.set(value)
        return value

    def _selected_worker_ids(self, total_slots: int) -> list[int]:
        raw = self.worker_ids_var.get().strip()
        if not raw:
            return list(range(1, total_slots + 1))
        selected: list[int] = []
        for part in re.split(r"[,\s]+", raw):
            if not part:
                continue
            try:
                worker_id = int(part)
            except ValueError as e:
                raise ValueError(f"Worker ID must be numeric: {part!r}") from e
            if worker_id < 1 or worker_id > total_slots:
                raise ValueError(f"Worker ID {worker_id} must be between 1 and {total_slots}")
            if worker_id not in selected:
                selected.append(worker_id)
        if not selected:
            return list(range(1, total_slots + 1))
        return selected

    def _seed_file_options(self) -> list[str]:
        seeds_dir = ROOT / "seeds"
        options: list[str] = []
        try:
            if seeds_dir.exists():
                for path in sorted(seeds_dir.glob("*.txt")):
                    try:
                        options.append(str(path.relative_to(ROOT)).replace("\\", "/"))
                    except ValueError:
                        options.append(str(path))
        except OSError as e:
            _logger.warning(f"Failed to scan seed files: {e}")
        if "seeds/eval_200.txt" not in options:
            options.insert(0, "seeds/eval_200.txt")
        return options

    def _refresh_seed_file_options(self):
        values = self._seed_file_options()
        self.eval_seed_combo.configure(values=values)
        if not self.eval_seed_file_var.get().strip() and values:
            self.eval_seed_file_var.set(values[0])

    # ----- Archive controls -----

    _ARCHIVE_LOG_FILES = (
        "training_stats.csv",
        "eval_stats.csv",
        "elite_stats.csv",
        "fight_stats.csv",
        "training_plot.png",
    )

    def _archives_dir(self) -> Path:
        return ROOT / "archives"

    def _archive_source_files(self) -> tuple[dict[str, Path], dict[str, Path]]:
        """Return (model_files, log_files) keyed by dest-relative archive path.

        Model files are kept in place after archiving; log files are removed
        from logs/ so the Progress panel resets for the next run.
        """
        models: dict[str, Path] = {}
        logs: dict[str, Path] = {}
        model_path = ROOT / "models" / "ppo_sts.pt"
        if model_path.exists():
            models["models/ppo_sts.pt"] = model_path
        for name in self._ARCHIVE_LOG_FILES:
            p = ROOT / "logs" / name
            if p.exists():
                logs[f"logs/{name}"] = p
        return models, logs

    def _scan_archives(self) -> list[tuple[Path, datetime | None]]:
        arch_dir = self._archives_dir()
        if not arch_dir.exists():
            return []
        found: list[tuple[Path, datetime | None]] = []
        try:
            entries = list(arch_dir.iterdir())
        except OSError as e:
            _logger.warning(f"Could not read archives dir: {e}")
            return []
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
                model_path = entry / "models" / "ppo_sts.pt"
                if not model_path.exists():
                    continue
                try:
                    mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                except OSError:
                    mtime = None
                found.append((entry, mtime))
            except OSError:
                continue
        found.sort(key=lambda t: (t[1] or datetime.min), reverse=True)
        return found

    def _active_model_label(self) -> str:
        path = ROOT / "models" / "ppo_sts.pt"
        if not path.exists():
            return "Active (models/ppo_sts.pt — not present yet)"
        try:
            st = path.stat()
            when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = st.st_size / (1024 * 1024)
            return f"Active (models/ppo_sts.pt — {when}, {size_mb:.1f} MB)"
        except OSError:
            return "Active (models/ppo_sts.pt)"

    def _archive_label(self, archive_dir: Path,
                       mtime: datetime | None) -> str:
        suffix = f" — {mtime.strftime('%Y-%m-%d %H:%M')}" if mtime else ""
        return f"{archive_dir.name}{suffix}"

    def _refresh_models_and_stats(self):
        self._refresh_archive_list()
        self._refresh_stats_now()

    def _refresh_archive_list(self):
        options: dict[str, Path | None] = {}
        active_label = self._active_model_label()
        options[active_label] = None
        for arch_dir, mtime in self._scan_archives():
            options[self._archive_label(arch_dir, mtime)] = arch_dir
        self._archive_options = options
        if hasattr(self, "model_combo"):
            self.model_combo.configure(values=list(options.keys()))
            current = self.model_choice_var.get()
            if current not in options:
                self.model_choice_var.set(active_label)

    def _resolve_selected_archive(self) -> Path | None:
        return self._archive_options.get(self.model_choice_var.get())

    _ARCHIVE_RELEVANT_MODES = {
        "worker", "collect", "consume", "train", "bc_ppo", "eval", "eval_set",
    }

    def _prompt_archive_now(self):
        if self.running:
            messagebox.showwarning(
                "Cannot Archive While Running",
                "Stop training/eval before archiving so the model file is "
                "not being written.",
            )
            return
        model_files, log_files = self._archive_source_files()
        if "models/ppo_sts.pt" not in model_files:
            messagebox.showinfo(
                "Nothing to Archive",
                "models/ppo_sts.pt does not exist yet — nothing to archive.",
            )
            return
        default_name = "archive_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_name = simpledialog.askstring(
            "Archive Current Model",
            "Name this archive (used as the folder name under archives/):",
            initialvalue=default_name,
            parent=self.root,
        )
        if raw_name is None:
            return
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._-")
        if not slug:
            slug = default_name
        target_dir = self._archives_dir() / slug
        if target_dir.exists():
            messagebox.showerror(
                "Archive Exists",
                f"An archive named '{slug}' already exists at:\n  {target_dir}",
            )
            return

        model_lines = "\n".join(f"  {rel}" for rel in model_files)
        log_lines = "\n".join(f"  {rel}" for rel in log_files) or "  (none)"
        ok = messagebox.askyesno(
            "Confirm Archive",
            f"Create archive 'archives/{slug}/'?\n\n"
            f"Model (kept in place after copy):\n{model_lines}\n\n"
            f"Logs (REMOVED from logs/ after copy):\n{log_lines}\n\n"
            "The Progress panel will reset so the next run starts with "
            "fresh stats. The active model file stays as-is.",
        )
        if not ok:
            return
        try:
            removed = self._perform_archive(target_dir, model_files, log_files)
        except Exception as e:
            _logger.error(f"Archive failed: {traceback.format_exc()}")
            messagebox.showerror("Archive Failed",
                                 f"Could not create archive:\n{e}")
            return
        total = len(model_files) + len(log_files)
        _logger.info(
            f"Archive created at {target_dir}: {total} file(s) copied, "
            f"{removed} log file(s) removed from logs/"
        )
        self._append_log(
            "All",
            f"Archived {total} file(s) -> archives/{slug}/; "
            f"cleared {removed} log file(s) from logs/.",
        )
        self._refresh_archive_list()
        self._refresh_stats_now()
        messagebox.showinfo(
            "Archive Complete",
            f"Archived to:\n  archives/{slug}/\n\n"
            f"Cleared {removed} log file(s) from logs/.",
        )

    def _perform_archive(self, target_dir: Path,
                         model_files: dict[str, Path],
                         log_files: dict[str, Path]) -> int:
        """Copy all sources to the archive, then delete log sources.

        Returns the number of log files successfully removed from logs/.
        """
        target_dir.mkdir(parents=True, exist_ok=False)
        copied: list[dict] = []
        for rel, src in {**model_files, **log_files}.items():
            dst = target_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            try:
                size = dst.stat().st_size
            except OSError:
                size = None
            copied.append({
                "path": rel,
                "source": str(src),
                "size_bytes": size,
                "moved": rel.startswith("logs/"),
            })

        removed = 0
        for rel, src in log_files.items():
            try:
                src.unlink()
                removed += 1
                _logger.debug(f"Cleared archived log source: {src}")
            except OSError as e:
                _logger.warning(
                    f"Archive copied {rel} but could not remove source "
                    f"{src}: {e}"
                )

        manifest = {
            "name": target_dir.name,
            "created_at": datetime.now().isoformat(),
            "source_root": str(ROOT),
            "files": copied,
            "logs_removed_from_source": removed,
        }
        manifest_path = target_dir / "manifest.json"
        tmp = manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(str(tmp), str(manifest_path))
        return removed

    def _maybe_restore_archive_before_start(self) -> bool:
        """If an archived model is selected, confirm + copy it over models/ppo_sts.pt.

        Returns True if the launch should proceed, False if the user cancelled
        or the restore failed.
        """
        archive_dir = self._resolve_selected_archive()
        if archive_dir is None:
            return True
        src = archive_dir / "models" / "ppo_sts.pt"
        if not src.exists():
            messagebox.showerror(
                "Archive Missing Model",
                f"Selected archive has no model at:\n  {src}",
            )
            return False
        dst = ROOT / "models" / "ppo_sts.pt"
        dst_info = ""
        if dst.exists():
            try:
                st = dst.stat()
                when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                size_mb = st.st_size / (1024 * 1024)
                dst_info = (
                    f"\n\nThe current active model "
                    f"({when}, {size_mb:.1f} MB) will be overwritten."
                )
            except OSError:
                pass
        ok = messagebox.askyesno(
            "Restore Archived Model",
            f"Restore archive '{archive_dir.name}' to "
            f"models/ppo_sts.pt and start?{dst_info}\n\n"
            "Click No and use 'Archive Current...' first if you want to "
            "keep the current model.",
        )
        if not ok:
            return False
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        except Exception as e:
            _logger.error(f"Restore failed: {traceback.format_exc()}")
            messagebox.showerror("Restore Failed",
                                 f"Could not restore archive:\n{e}")
            return False
        _logger.info(f"Restored archive {archive_dir.name} -> {dst}")
        self._append_log(
            "All",
            f"Restored archive '{archive_dir.name}' to models/ppo_sts.pt",
        )
        self._refresh_archive_list()
        self.model_choice_var.set(self._active_model_label())
        return True

    def _selected_seed_file(self, games: int) -> str:
        value = self.eval_seed_file_var.get().strip()
        if not value:
            value = f"seeds/eval_{games}.txt"
            self.eval_seed_file_var.set(value)
        value = value.replace("\\", "/")
        return value

    def _normalize_eval_policy_key(self, key: str | None) -> str:
        key = str(key or "").strip()
        if key in EVAL_POLICY_OPTIONS.values():
            return key
        return "ppo_current"

    def _selected_eval_policy(self) -> str:
        return self._normalize_eval_policy_key(
            EVAL_POLICY_OPTIONS.get(self.eval_policy_var.get().strip())
        )

    def _eval_policy_label(self, key: str | None = None) -> str:
        normalized = self._selected_eval_policy() if key is None else self._normalize_eval_policy_key(key)
        for label, candidate in EVAL_POLICY_OPTIONS.items():
            if candidate == normalized:
                return label
        return "PPO Current only"

    def _eval_runs_for_policy(self, key: str, games: int, ts: str) -> list[dict]:
        normalized = self._normalize_eval_policy_key(key)
        run_templates = {
            "heuristic": {
                "label": "Heuristic",
                "key": "heuristic",
                "policy": "heuristic",
                "model": None,
                "tag": f"heuristic_{games}_{ts}",
                "top_actions": 0,
            },
            "bc": {
                "label": "BC",
                "key": "bc",
                "policy": "model",
                "model": "models/ppo_sts_bc.pt",
                "tag": f"bc_{games}_{ts}",
                "top_actions": 5,
            },
            "ppo_current": {
                "label": "PPO Current",
                "key": "ppo_current",
                "policy": "model",
                "model": "models/ppo_sts.pt",
                "tag": f"ppo_current_{games}_{ts}",
                "top_actions": 5,
            },
        }
        run_keys = ["heuristic", "bc", "ppo_current"] if normalized == "full" else [normalized]
        return [dict(run_templates[run_key]) for run_key in run_keys]

    def _set_ent_value(self, value: float, *, manual: bool = False):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.001
        v = round(max(0.0, min(0.01, v)), 6)
        self._ent_value = v
        if manual:
            self._ent_manual_override = True
        if hasattr(self, "ent_var"):
            self.ent_var.set(f"{v:.6f}")
        if hasattr(self, "ent_scale"):
            self._ent_updating_ui = True
            try:
                self.ent_scale.set(v)
            finally:
                self._ent_updating_ui = False

    def _on_ent_scale(self, value):
        if self._ent_updating_ui:
            return
        self._set_ent_value(float(value), manual=True)

    def _sync_ent_from_latest(self, *, manual: bool = True):
        if self._latest_trainer_ent_coef is not None:
            self._set_ent_value(self._latest_trainer_ent_coef, manual=manual)

    def _set_ent_current_text(self, ent_coef, normalized_entropy, action):
        if ent_coef is None:
            self.ent_current_var.set("Current: no trainer entropy yet")
            return
        parts = [f"Current: ent_coef {ent_coef:.6f}"]
        if normalized_entropy is not None:
            parts.append(f"norm_ent {normalized_entropy:.3f}")
        if action:
            parts.append(str(action))
        self.ent_current_var.set(" | ".join(parts))

    def _inc_ent(self):
        self._set_ent_value(self._ent_value + 0.0001, manual=True)

    def _dec_ent(self):
        self._set_ent_value(self._ent_value - 0.0001, manual=True)

    _ENT_TOOLTIP_TEXT = (
        "Controls how much the agent explores vs. exploits.\n"
        "\n"
        "Higher (0.01-0.10): More random actions, but can\n"
        "destabilize a behavior-cloned warm start.\n"
        "\n"
        "Lower (0.00-0.005): More deterministic, better for\n"
        "testing whether PPO improves the BC policy.\n"
        "\n"
        "The slider and +/- buttons set the starting entropy for a\n"
        "new trainer launch. Auto-Tune may adjust it after that.\n"
        "Sync latest copies the most recent logged trainer value.\n"
        "\n"
        "Recommended after BC: 0.001000-0.005000. Raise only if\n"
        "entropy collapses too early."
    )

    def _show_ent_tooltip(self, event):
        if self._ent_tooltip is not None:
            return
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 20
        tw = tk.Toplevel(self.root)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        lbl = tk.Label(tw, text=self._ENT_TOOLTIP_TEXT, justify="left",
                       background="#ffffe0", relief="solid", borderwidth=1,
                       font=("Segoe UI", 9), padx=8, pady=6)
        lbl.pack()
        self._ent_tooltip = tw

    def _hide_ent_tooltip(self, event):
        if self._ent_tooltip is not None:
            self._ent_tooltip.destroy()
            self._ent_tooltip = None

    def _on_mode_change(self, event=None):
        mode = MODES.get(self.mode_var.get(), "worker")
        label, lo, hi, default = SPINNER_CONFIG.get(mode, (None, 0, 0, 0))
        if label:
            self.spinner_label_var.set(label)
            self.spin_minus.configure(state="normal")
            self.spin_plus.configure(state="normal")
            self.workers_entry.configure(state="normal")
            if default is not None:
                self.workers_var.set(default)
            elif mode in ("worker", "collect"):
                self.workers_var.set(self.hw["recommended_workers"])
            self._normalize_spinner_value()
        else:
            self.spinner_label_var.set("")
            self.spin_minus.configure(state="disabled")
            self.spin_plus.configure(state="disabled")
            self.workers_entry.configure(state="disabled")
        self._update_spinner_step_hint()

        bc_path = ROOT / "models" / "ppo_sts_bc.pt"
        if mode in ("worker", "collect", "train", "eval") and bc_path.exists():
            self.use_bc_chk.pack(side="left", padx=(0, 10))
        else:
            self.use_bc_chk.pack_forget()
            self.use_bc_var.set(False)

        if mode in ("worker", "train", "bc_ppo", "consume"):
            self.ent_row.pack(fill="x", pady=(6, 0))
        else:
            self.ent_row.pack_forget()

        if mode in ("bc", "bc_ppo", "bc_train"):
            self.bc_epochs_row.pack(fill="x", pady=(6, 0))
        else:
            self.bc_epochs_row.pack_forget()

        if mode == "bc_collect":
            self.bc_workers_row.pack(fill="x", pady=(6, 0))
        else:
            self.bc_workers_row.pack_forget()

        if mode in ("worker", "collect", "bc_collect"):
            self.worker_ids_row.pack(fill="x", pady=(6, 0))
        else:
            self.worker_ids_row.pack_forget()

        if mode == "bc_ppo":
            self.ppo_games_row.pack(fill="x", pady=(6, 0))
        else:
            self.ppo_games_row.pack_forget()

        if mode in ("worker", "collect"):
            self.ppo_workers_games_row.pack(fill="x", pady=(6, 0))
        else:
            self.ppo_workers_games_row.pack_forget()

        if mode in ("worker", "collect", "eval_set"):
            self.restart_sts_row.pack(fill="x", pady=(6, 0))
            self.jvm_heap_row.pack(fill="x", pady=(6, 0))
        else:
            self.restart_sts_row.pack_forget()
            self.jvm_heap_row.pack_forget()

        if mode == "eval_set":
            self._refresh_seed_file_options()
            self.eval_policy_row.pack(fill="x", pady=(6, 0))
            self.eval_seed_row.pack(fill="x", pady=(6, 0))
        else:
            self.eval_policy_row.pack_forget()
            self.eval_seed_row.pack_forget()

    # ----- Logging -----

    _LOG_MAX_LINES = 5000
    _LOG_TRIM_TO = 4000

    def _append_log(self, tab_name: str, text: str):
        def _do():
            for name in (tab_name, "All"):
                widget = self.log_tabs.get(name)
                if widget is None:
                    continue
                widget.configure(state="normal")
                prefix = f"[{tab_name}] " if name == "All" else ""
                widget.insert("end", prefix + text + "\n")
                line_count = int(widget.index("end-1c").split(".")[0])
                if line_count > self._LOG_MAX_LINES:
                    widget.delete("1.0", f"{line_count - self._LOG_TRIM_TO}.0")
                widget.see("end")
                widget.configure(state="disabled")
        self.root.after(0, _do)

    def _clear_logs(self):
        for widget in self.log_tabs.values():
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")

    # ----- Progress stats -----

    def _load_fight_stats(self, *, include_eval: bool = False) -> dict:
        csv_path = ROOT / "logs" / "fight_stats.csv"
        result = {
            "has_data": False,
            "elites_fought": 0,
            "elites_won": 0,
            "bosses_fought": 0,
            "bosses_won": 0,
        }
        if not csv_path.exists():
            return result
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(_csv.DictReader(f))
        except Exception as e:
            _logger.warning(f"Failed to parse fight_stats.csv: {e}")
            return result

        for r in rows:
            source = str(r.get("source") or "")
            if not include_eval and source in {"bc", "eval"}:
                continue
            fight_type = str(r.get("fight_type") or "").lower()
            room_type = str(r.get("room_type") or "")
            if not fight_type:
                if "Boss" in room_type:
                    fight_type = "boss"
                elif "Elite" in room_type:
                    fight_type = "elite"
            if fight_type not in ("elite", "boss"):
                continue
            try:
                won = int(float(r.get("won") or 0))
            except Exception:
                won = 0
            result["has_data"] = True
            if fight_type == "elite":
                result["elites_fought"] += 1
                result["elites_won"] += won
            else:
                result["bosses_fought"] += 1
                result["bosses_won"] += won
        return result

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

        game_rows = [r for r in rows if r.get("final_floor") not in (None, "")]
        floors, rewards, wins, best_floor, best_act = [], [], 0, 0, 0
        updates, total_transitions, total_steps = 0, 0, 0
        trainer_transitions = 0
        trainer_transition_rows = 0
        latest_trainer_ent_coef = None
        latest_normalized_entropy = None
        latest_auto_tune_action = ""
        prev_cumulative_transitions: dict[str, int] = {}
        prev_cumulative_steps: dict[str, int] = {}
        acts, victories = [], []
        parse_errors = 0
        for r in game_rows:
            try:
                fl = float(r.get("final_floor") or 0)
                floors.append(fl)
                if fl > best_floor:
                    best_floor = fl
            except Exception:
                parse_errors += 1
            try:
                rewards.append(float(r.get("total_reward") or 0))
            except Exception:
                parse_errors += 1
            try:
                v_raw = int(float(r.get("victory") or 0))
                fl = float(r.get("final_floor") or 0)
                act = int(float(r.get("final_act") or 0))
                # Older logs incorrectly treated CommunicationMod's COMPLETE
                # screen as a real run victory. A genuine Ironclad win should
                # be late Act 3/4, so suppress stale impossible win rows.
                v = 1 if v_raw and (act >= 3 or fl >= 50) else 0
                victories.append(v)
                if v:
                    wins += 1
            except Exception:
                parse_errors += 1
            try:
                u = int(float(r.get("total_updates") or 0))
                if u > updates:
                    updates = u
            except Exception:
                parse_errors += 1
            try:
                raw_transitions = int(float(r.get("transitions") or 0))
                worker = str(r.get("worker") or "single")
                series_key = worker
                # Older BC/single rows wrote cumulative demo counts here.
                # Worker rollout rows are normally per-game and much smaller.
                looks_cumulative = (
                    series_key in prev_cumulative_transitions
                    and raw_transitions >= prev_cumulative_transitions[series_key]
                    and raw_transitions > 1000
                ) or (worker == "single" and raw_transitions > 1000)
                if looks_cumulative:
                    prev = prev_cumulative_transitions.get(series_key)
                    if prev is not None and raw_transitions >= prev:
                        transitions = raw_transitions - prev
                    else:
                        transitions = raw_transitions
                    prev_cumulative_transitions[series_key] = raw_transitions
                else:
                    transitions = raw_transitions
                    if (series_key in prev_cumulative_transitions
                            and raw_transitions < prev_cumulative_transitions[series_key]):
                        prev_cumulative_transitions.pop(series_key, None)
                total_transitions += max(0, transitions)
            except Exception:
                pass
            try:
                raw_steps = int(float(r.get("steps") or 0))
                transitions = int(float(r.get("transitions") or 0))
                worker = str(r.get("worker") or "single")
                game_no = str(r.get("game") or "")
                series_key = worker
                # Older worker/BC rows wrote lifetime step counters while
                # transitions were per-game. Diff monotonic counters so the
                # progress panel does not count game 1 + game 2 + ... totals.
                looks_cumulative = (
                    raw_steps > max(1000, transitions * 8)
                    or (
                        series_key in prev_cumulative_steps
                        and raw_steps >= prev_cumulative_steps[series_key]
                        and transitions > 0
                        and raw_steps > transitions * 3
                    )
                )
                if looks_cumulative:
                    prev = prev_cumulative_steps.get(series_key)
                    if prev is not None and raw_steps >= prev:
                        steps = raw_steps - prev
                    else:
                        steps = max(transitions, min(raw_steps, transitions * 3))
                    prev_cumulative_steps[series_key] = raw_steps
                    _logger.debug(
                        f"Corrected cumulative steps row worker={worker} "
                        f"game={game_no}: raw={raw_steps} -> {steps}"
                    )
                else:
                    steps = raw_steps
                    if series_key in prev_cumulative_steps and raw_steps < prev_cumulative_steps[series_key]:
                        prev_cumulative_steps.pop(series_key, None)
                total_steps += max(0, steps)
            except Exception:
                pass
            try:
                act = int(float(r.get("final_act") or 0))
                acts.append(act)
                if act > best_act:
                    best_act = act
            except Exception:
                pass
        if parse_errors:
            _logger.debug(f"training_stats.csv: {parse_errors} field parse error(s) "
                          f"across {len(game_rows)} game rows")

        for r in rows:
            if str(r.get("worker") or "") != "trainer":
                continue
            try:
                trainer_transitions += int(float(r.get("transitions") or 0))
                trainer_transition_rows += 1
            except Exception:
                pass
            try:
                u = int(float(r.get("total_updates") or 0))
                if u > updates:
                    updates = u
            except Exception:
                pass
            try:
                ent_raw = r.get("ent_coef")
                if ent_raw not in (None, ""):
                    latest_trainer_ent_coef = float(ent_raw)
            except Exception:
                pass
            try:
                norm_raw = r.get("normalized_entropy")
                if norm_raw not in (None, ""):
                    latest_normalized_entropy = float(norm_raw)
            except Exception:
                pass
            action = str(r.get("auto_tune_action") or "").strip()
            if action:
                latest_auto_tune_action = action

        display_transitions = trainer_transitions if trainer_transitions else total_transitions

        recent_n = 100
        recent_floors = floors[-recent_n:]
        recent_rewards = rewards[-recent_n:]
        recent_wins = sum(victories[-recent_n:])
        lifetime_avg_floor = sum(floors) / len(floors) if floors else 0.0
        lifetime_avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        rollouts_pending = 0
        rollouts_dir = ROOT / "rollouts_shared"
        if rollouts_dir.exists():
            rollouts_pending = len(list(rollouts_dir.glob("*.npz")))

        model_path = ROOT / "models" / "ppo_sts.pt"
        model_age = ""
        if model_path.exists():
            delta = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            mins = int(delta.total_seconds() // 60)
            if mins < 60:
                model_age = f"{mins}m ago"
            elif mins < 1440:
                model_age = f"{mins // 60}h {mins % 60}m ago"
            else:
                model_age = f"{mins // 1440}d {(mins % 1440) // 60}h ago"

        fight_stats = self._load_fight_stats(include_eval=False)

        return {
            "total": len(game_rows),
            "wins": wins,
            "win_rate": wins / len(game_rows) if game_rows else 0.0,
            "recent_win_rate": recent_wins / len(victories[-recent_n:]) if victories[-recent_n:] else 0.0,
            "best_floor": int(best_floor),
            "best_act": best_act,
            "avg_floor": sum(recent_floors) / len(recent_floors) if recent_floors else 0.0,
            "avg_reward": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0,
            "lifetime_avg_floor": lifetime_avg_floor,
            "lifetime_avg_reward": lifetime_avg_reward,
            "updates": updates,
            "total_transitions": display_transitions,
            "episode_transitions": total_transitions,
            "trainer_transition_rows": trainer_transition_rows,
            "total_steps": total_steps,
            "rollouts_pending": rollouts_pending,
            "model_age": model_age,
            "latest_ent_coef": latest_trainer_ent_coef,
            "latest_normalized_entropy": latest_normalized_entropy,
            "latest_auto_tune_action": latest_auto_tune_action,
            "fight_stats_available": fight_stats["has_data"],
            "elites_fought": fight_stats["elites_fought"],
            "elites_won": fight_stats["elites_won"],
            "bosses_fought": fight_stats["bosses_fought"],
            "bosses_won": fight_stats["bosses_won"],
        }

    def _load_bc_stats(self) -> dict | None:
        csv_path = ROOT / "logs" / "bc_stats.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(_csv.DictReader(f))
        except Exception as e:
            _logger.warning(f"Failed to parse bc_stats.csv: {e}")
            return None

        rows = [r for r in rows if r.get("final_floor") not in (None, "")]
        if not rows:
            return None

        latest_run = ""
        for r in reversed(rows):
            latest_run = str(r.get("run_id") or "")
            if latest_run:
                break
        run_rows = [r for r in rows if str(r.get("run_id") or "") == latest_run] if latest_run else rows

        floors, victories = [], []
        best_floor, best_act = 0, 0
        samples, steps, skipped = 0, 0, 0
        elites_fought = elites_won = bosses_fought = bosses_won = 0
        target_games = 0
        sources: set[str] = set()
        for r in run_rows:
            try:
                fl = float(r.get("final_floor") or 0)
                floors.append(fl)
                best_floor = max(best_floor, int(fl))
            except Exception:
                pass
            try:
                act = int(float(r.get("final_act") or 0))
                best_act = max(best_act, act)
                v_raw = int(float(r.get("victory") or 0))
                fl = float(r.get("final_floor") or 0)
                victories.append(1 if v_raw and (act >= 3 or fl >= 50) else 0)
            except Exception:
                victories.append(0)
            for key, total_name in (
                ("samples", "samples"),
                ("steps", "steps"),
                ("skipped_samples", "skipped"),
            ):
                try:
                    value = int(float(r.get(key) or 0))
                except Exception:
                    value = 0
                if total_name == "samples":
                    samples += value
                elif total_name == "steps":
                    steps += value
                else:
                    skipped += value
            try:
                target_games = max(target_games, int(float(r.get("target_games") or 0)))
            except Exception:
                pass
            try:
                elites_fought += int(float(r.get("elites_fought") or 0))
                elites_won += int(float(r.get("elites_won") or 0))
                bosses_fought += int(float(r.get("bosses_fought") or 0))
                bosses_won += int(float(r.get("bosses_won") or 0))
            except Exception:
                pass
            source = str(r.get("source") or "")
            if source:
                sources.add(source)

        recent_floors = floors[-20:]
        return {
            "run_id": latest_run,
            "sources": ",".join(sorted(sources)) if sources else "bc",
            "games": len(run_rows),
            "target_games": target_games,
            "wins": sum(victories),
            "win_rate": sum(victories) / len(victories) if victories else 0.0,
            "avg_floor": sum(floors) / len(floors) if floors else 0.0,
            "recent_avg_floor": sum(recent_floors) / len(recent_floors) if recent_floors else 0.0,
            "best_floor": best_floor,
            "best_act": best_act,
            "samples": samples,
            "steps": steps,
            "skipped": skipped,
            "elites_fought": elites_fought,
            "elites_won": elites_won,
            "bosses_fought": bosses_fought,
            "bosses_won": bosses_won,
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
            elites_fought = sum(int(float(r.get("elites_fought") or 0)) for r in run_rows)
            elites_won = sum(int(float(r.get("elites_won") or 0)) for r in run_rows)
            bosses_fought = sum(int(float(r.get("bosses_fought") or 0)) for r in run_rows)
            bosses_won = sum(int(float(r.get("bosses_won") or 0)) for r in run_rows)
        except Exception as e:
            _logger.warning(f"Error parsing eval_stats.csv run '{last_tag}': {e}")
            return None
        return {
            "games": len(run_rows),
            "win_rate": wins / len(run_rows) if run_rows else 0.0,
            "avg_floor": sum(floors) / len(floors) if floors else 0.0,
            "run": last_tag,
            "elites_fought": elites_fought,
            "elites_won": elites_won,
            "bosses_fought": bosses_fought,
            "bosses_won": bosses_won,
        }

    def _refresh_stats(self):
        self._refresh_stats_now()
        self.root.after(5000, self._refresh_stats)

    def _refresh_stats_now(self):
        try:
            ts = self._load_training_stats()
            if ts:
                self._latest_trainer_ent_coef = ts["latest_ent_coef"]
                self._latest_normalized_entropy = ts["latest_normalized_entropy"]
                self._latest_auto_tune_action = ts["latest_auto_tune_action"]
                self._set_ent_current_text(
                    self._latest_trainer_ent_coef,
                    self._latest_normalized_entropy,
                    self._latest_auto_tune_action,
                )
                self.stats_vars["train_line"].set(
                    f"Training:  {ts['total']} games  |  "
                    f"Wins: {ts['wins']} ({ts['win_rate']:.0%})  |  "
                    f"Avg100 Wins: {ts['recent_win_rate']:.0%}  |  "
                    f"Best: Floor {ts['best_floor']} Act {ts['best_act']}  |  "
                    f"Avg100 Floor: {ts['avg_floor']:.1f}  |  "
                    f"Life Floor: {ts['lifetime_avg_floor']:.1f}"
                )
                detail_parts = [
                    f"Transitions: {ts['total_transitions']:,}",
                    f"Steps: {ts['total_steps']:,}",
                    f"Updates: {ts['updates']}",
                    f"Avg100 Reward: {ts['avg_reward']:.1f}",
                    f"Life Reward: {ts['lifetime_avg_reward']:.1f}",
                ]
                if ts["latest_ent_coef"] is not None:
                    detail_parts.append(f"Ent: {ts['latest_ent_coef']:.6f}")
                    if ts["latest_normalized_entropy"] is not None:
                        detail_parts.append(f"NormEnt: {ts['latest_normalized_entropy']:.3f}")
                    if not self.running and not self._ent_manual_override:
                        self._sync_ent_from_latest(manual=False)
                if ts['rollouts_pending']:
                    detail_parts.append(f"Rollouts Queued: {ts['rollouts_pending']}")
                if ts['model_age']:
                    detail_parts.append(f"Model: {ts['model_age']}")
                self.stats_vars["detail_line"].set("  " + "  |  ".join(detail_parts))

                combat_parts = []
                if ts["elites_fought"]:
                    ew = ts["elites_won"] / ts["elites_fought"]
                    combat_parts.append(
                        f"Elites: {ts['elites_won']}/{ts['elites_fought']} ({ew:.0%})"
                    )
                if ts["bosses_fought"]:
                    bw = ts["bosses_won"] / ts["bosses_fought"]
                    combat_parts.append(
                        f"Bosses: {ts['bosses_won']}/{ts['bosses_fought']} ({bw:.0%})"
                    )
                if combat_parts:
                    self.stats_vars["combat_line"].set(
                        "Fights:  " + "  |  ".join(combat_parts)
                    )
                elif ts["fight_stats_available"]:
                    self.stats_vars["combat_line"].set(
                        "Fights:  no elite/boss fights tracked yet"
                    )
                else:
                    self.stats_vars["combat_line"].set(
                        "Fights:  waiting for post-fix fight data"
                    )
            else:
                self.stats_vars["train_line"].set("Training:  no data yet")
                self.stats_vars["detail_line"].set("")
                self.stats_vars["combat_line"].set("")
                self._set_ent_current_text(None, None, "")

            bs = self._load_bc_stats()
            if bs:
                target = f"/{bs['target_games']}" if bs["target_games"] else ""
                bc_parts = [
                    f"BC:        {bs['games']}{target} games",
                    f"Avg Floor: {bs['avg_floor']:.1f}",
                    f"Last20: {bs['recent_avg_floor']:.1f}",
                    f"Best: Floor {bs['best_floor']} Act {bs['best_act']}",
                    f"Samples: {bs['samples']:,}",
                ]
                if bs["wins"]:
                    bc_parts.append(f"Wins: {bs['wins']} ({bs['win_rate']:.0%})")
                if bs["skipped"]:
                    bc_parts.append(f"Skipped: {bs['skipped']:,}")
                self.stats_vars["bc_line"].set("  |  ".join(bc_parts))
            else:
                self.stats_vars["bc_line"].set("BC:        no baseline yet")

            es = self._load_eval_stats()
            if es:
                eval_parts = [
                    f"Eval ({es['run']}):  {es['games']} games",
                    f"Win Rate: {es['win_rate']:.0%}",
                    f"Avg Floor: {es['avg_floor']:.1f}",
                ]
                if es["elites_fought"]:
                    eval_parts.append(
                        f"Elites: {es['elites_won']}/{es['elites_fought']}"
                    )
                if es["bosses_fought"]:
                    eval_parts.append(
                        f"Bosses: {es['bosses_won']}/{es['bosses_fought']}"
                    )
                self.stats_vars["eval_line"].set(
                    "  |  ".join(eval_parts)
                )
            else:
                self.stats_vars["eval_line"].set("Eval:  no data yet")
            if self.running and self.trainer_proc is not None:
                rc = self.trainer_proc.poll()
                if rc is not None:
                    _logger.warning(f"Trainer process died (exit code {rc})")
                    self._append_log("All", f"WARNING: Trainer crashed (exit code {rc}) — no PPO updates are happening!")
                    self.trainer_proc = None
        except Exception as e:
            _logger.error(f"_refresh_stats error: {e}")

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

    def _start_validation_failed(self, message: str):
        _logger.error(f"Start input validation failed: {message}")
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.graceful_btn.configure(state="disabled")
        self.status_var.set("Idle")
        messagebox.showerror("Cannot Start", message)

    def _start(self):
        _logger.info("=" * 40)
        _logger.info("START pressed")
        errors = self._validate()
        if errors:
            _logger.error(f"Validation failed:\n{errors}")
            messagebox.showerror("Cannot Start", errors)
            return

        pending_mode = MODES.get(self.mode_var.get(), "worker")
        if pending_mode in self._ARCHIVE_RELEVANT_MODES:
            if not self._maybe_restore_archive_before_start():
                _logger.info("Start cancelled: archive restore declined or failed")
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
        n = self._normalize_spinner_value()
        bc_epochs = self._get_bc_epochs()

        model_path = ROOT / "models" / "ppo_sts.pt"
        _logger.info(f"Mode: {mode} | Spinner: {n} | Verbose: {verbose}")
        _logger.info(f"Model checkpoint: {model_path}  exists={model_path.exists()}")
        if model_path.exists():
            _logger.info(f"Model size: {model_path.stat().st_size} bytes, "
                         f"modified: {datetime.fromtimestamp(model_path.stat().st_mtime)}")

        (ROOT / "models").mkdir(exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)

        if self.use_bc_var.get():
            bc_path = ROOT / "models" / "ppo_sts_bc.pt"
            if bc_path.exists():
                shutil.copy2(str(bc_path), str(model_path))
                _logger.info(f"Copied BC checkpoint -> {model_path}")
                self._append_log("All", "Initialized model from BC checkpoint")
                bc_demo_target = ROOT / "models" / "ppo_sts_bc_demos.npz"
                for bc_demo_source in (
                    ROOT / "models" / "ppo_sts_bc_demos.npz",
                    ROOT / "models" / "ppo_sts_bc_bc_demos.npz",
                ):
                    if not bc_demo_source.exists():
                        continue
                    if bc_demo_source != bc_demo_target:
                        shutil.copy2(str(bc_demo_source), str(bc_demo_target))
                    _logger.info(f"BC demo anchor available at {bc_demo_target}")
                    break
            else:
                _logger.warning("Use BC checked but ppo_sts_bc.pt not found")
                self._append_log("All", "WARNING: BC checkpoint not found, starting fresh")

        if mode == "worker":
            (ROOT / "rollouts_shared").mkdir(exist_ok=True)
            rollout_count = len(list((ROOT / "rollouts_shared").glob("*.npz")))
            _logger.info(f"rollouts_shared/ has {rollout_count} existing .npz files")
            try:
                worker_ids = self._selected_worker_ids(n)
            except ValueError as e:
                self._start_validation_failed(str(e))
                return
            total_target = self._get_ppo_workers_games()
            worker_targets = self._compute_worker_game_targets(total_target, n)
            target_desc = (
                f", target {total_target} games total ({worker_targets})"
                if worker_targets else ""
            )
            _logger.info(f"Parallel worker game target: total={total_target}, "
                         f"per-slot={worker_targets}")
            self.status_var.set(
                f"Launching {len(worker_ids)}/{n} workers + trainer{target_desc}..."
            )
            threading.Thread(target=self._launch_parallel,
                             args=(n,),
                             kwargs={"worker_ids": worker_ids,
                                     "worker_targets": worker_targets},
                             daemon=True).start()
        elif mode == "collect":
            (ROOT / "rollouts_shared").mkdir(exist_ok=True)
            rollout_count = len(list((ROOT / "rollouts_shared").glob("*.npz")))
            _logger.info(f"rollouts_shared/ has {rollout_count} existing .npz files")
            try:
                worker_ids = self._selected_worker_ids(n)
            except ValueError as e:
                self._start_validation_failed(str(e))
                return
            total_target = self._get_ppo_workers_games()
            worker_targets = self._compute_worker_game_targets(total_target, n)
            target_desc = (
                f", target {total_target} games total ({worker_targets})"
                if worker_targets else ""
            )
            _logger.info(f"Collect-only worker game target: total={total_target}, "
                         f"per-slot={worker_targets}")
            self.status_var.set(
                f"Launching {len(worker_ids)}/{n} collectors (no trainer){target_desc}..."
            )
            threading.Thread(target=self._launch_parallel,
                             args=(n,),
                             kwargs={"with_trainer": False,
                                     "worker_ids": worker_ids,
                                     "worker_targets": worker_targets},
                             daemon=True).start()
        elif mode == "consume":
            (ROOT / "rollouts_shared").mkdir(exist_ok=True)
            rollout_count = len(list((ROOT / "rollouts_shared").glob("*.npz")))
            _logger.info(f"rollouts_shared/ has {rollout_count} existing .npz files")
            if rollout_count == 0:
                self._start_validation_failed(
                    "No rollout files in rollouts_shared/.\n"
                    "Use Parallel Workers or Collect Rollouts first."
                )
                return
            self.status_var.set(
                f"Consuming {rollout_count} rollouts (trainer only, no game workers)..."
            )
            threading.Thread(target=self._launch_consume,
                             daemon=True).start()
        elif mode == "train":
            self.status_var.set("Launching single-instance training...")
            threading.Thread(target=self._launch_single, args=("train",), daemon=True).start()
        elif mode == "bc_ppo":
            ppo_games = self._get_ppo_games()
            ppo_desc = "continuous PPO" if ppo_games == 0 else f"{ppo_games} PPO games"
            self.status_var.set(f"Launching BC\u2192PPO ({n} BC games, {ppo_desc})...")
            threading.Thread(target=self._launch_single,
                             args=("bc_ppo", n, ppo_games, bc_epochs),
                             daemon=True).start()
        elif mode == "bc":
            self.status_var.set(f"Launching behavior cloning ({n} games)...")
            threading.Thread(target=self._launch_single, args=("bc", n, 200, bc_epochs), daemon=True).start()
        elif mode == "bc_collect":
            total_slots = self._get_bc_workers()
            try:
                worker_ids = self._selected_worker_ids(total_slots)
            except ValueError as e:
                self._start_validation_failed(str(e))
                return
            self.status_var.set(
                f"Launching BC collection {len(worker_ids)}/{total_slots} workers "
                f"({n} total games)..."
            )
            threading.Thread(target=self._launch_bc_collect,
                             args=(total_slots, n, worker_ids),
                             kwargs={"reset_existing": not self.worker_ids_var.get().strip()},
                             daemon=True).start()
        elif mode == "bc_train":
            self.status_var.set("Training BC from collected demos...")
            threading.Thread(target=self._launch_bc_train_from_demos,
                             args=(bc_epochs,),
                             daemon=True).start()
        elif mode == "eval":
            self.status_var.set(f"Launching evaluation ({n} games)...")
            threading.Thread(target=self._launch_single, args=("eval", n), daemon=True).start()
        elif mode == "eval_set":
            resume_manifest = self._prompt_eval_set_resume(n)
            if resume_manifest is False:
                self.running = False
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.graceful_btn.configure(state="disabled")
                self.status_var.set("Eval set launch cancelled.")
                return
            launch_label = "Resuming" if resume_manifest else "Launching"
            policy_label = (
                self._eval_policy_label(str(resume_manifest.get("eval_policy") or "full"))
                if resume_manifest else self._eval_policy_label()
            )
            runs_count = len(resume_manifest.get("runs") or []) if resume_manifest else (
                3 if self._selected_eval_policy() == "full" else 1
            )
            games_desc = f"{n} games each" if runs_count > 1 else f"{n} games"
            self.status_var.set(f"{launch_label} {policy_label} fixed-seed eval ({games_desc})...")
            threading.Thread(
                target=self._launch_eval_set,
                args=(n,),
                kwargs={"resume_manifest": resume_manifest},
                daemon=True,
            ).start()
        elif mode == "logger":
            self.status_var.set("Launching passive game logger...")
            threading.Thread(target=self._launch_single, args=("logger",), daemon=True).start()
        elif mode == "play":
            self.status_var.set("Launching game (no AI)...")
            threading.Thread(target=self._launch_play, daemon=True).start()
        _logger.info(f"Launch thread started for mode={mode}")

    def _generate_seed_file(self, seed_file: str, games: int) -> bool:
        out_path = ROOT / seed_file if not os.path.isabs(seed_file) else Path(seed_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(VENV_PYTHON), str(SCRIPTS / "make_eval_seeds.py"),
            "--count", str(int(games)),
            "--out", str(out_path),
        ]
        _logger.info(f"Generating eval seed file: {cmd}")
        self._append_log("Eval Set", f"Generating seed file: {out_path}")
        try:
            result = subprocess.run(
                cmd, cwd=str(ROOT), check=True, timeout=60,
                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.stdout:
                self._append_log("Eval Set", result.stdout.decode(errors="replace").strip())
            if result.stderr:
                _logger.debug(f"Seed generation stderr: {result.stderr.decode(errors='replace')[:500]}")
            self._refresh_seed_file_options()
            return True
        except Exception as e:
            _logger.error(f"Seed generation failed: {traceback.format_exc()}")
            self._append_log("Eval Set", f"ERROR generating seed file: {e}")
            return False

    def _count_eval_rows_for_run(self, run_tag: str) -> int:
        csv_path = ROOT / "logs" / "eval_stats.csv"
        if not csv_path.exists():
            return 0
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                return sum(1 for r in _csv.DictReader(f) if r.get("run") == run_tag)
        except Exception as e:
            _logger.warning(f"Failed to count eval rows for {run_tag}: {e}")
            return 0

    def _eval_set_manifest_path(self) -> Path:
        return ROOT / "Eval" / "eval_set_state.json"

    def _load_eval_set_manifest(self) -> dict | None:
        path = self._eval_set_manifest_path()
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _logger.warning(f"Failed to read eval set manifest: {e}")
            return None
        if not isinstance(data, dict) or "ts" not in data or "runs" not in data:
            _logger.warning("Eval set manifest is malformed; ignoring.")
            return None
        return data

    def _save_eval_set_manifest(self, manifest: dict) -> None:
        path = self._eval_set_manifest_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(str(tmp), str(path))
        except Exception as e:
            _logger.warning(f"Failed to write eval set manifest: {e}")

    def _clear_eval_set_manifest(self) -> None:
        path = self._eval_set_manifest_path()
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            _logger.warning(f"Failed to clear eval set manifest: {e}")

    def _eval_set_manifest_progress(self, manifest: dict) -> list[tuple[str, int, int]]:
        games = int(manifest.get("games", 0))
        out: list[tuple[str, int, int]] = []
        for run in manifest.get("runs", []) or []:
            tag = run.get("tag") or ""
            label = run.get("label") or tag
            rows = self._count_eval_rows_for_run(tag) if tag else 0
            out.append((label, rows, games))
        return out

    def _eval_set_has_resumable_progress(self, manifest: dict) -> bool:
        progress = self._eval_set_manifest_progress(manifest)
        if not progress:
            return False
        has_rows = any(rows > 0 for _, rows, _ in progress)
        any_incomplete = any(rows < games for _, rows, games in progress)
        return has_rows and any_incomplete

    def _prompt_eval_set_resume(self, games: int):
        """Return a manifest dict to resume from, None to start fresh, or False to abort.

        Called on the UI thread before the eval-set worker is launched. If a manifest
        from a previous (interrupted) eval set is on disk with any recorded rows,
        ask the user whether to resume those tags or discard and start over.
        """
        manifest = self._load_eval_set_manifest()
        if manifest is None:
            return None
        if not self._eval_set_has_resumable_progress(manifest):
            self._clear_eval_set_manifest()
            return None

        current_seed_file = self._selected_seed_file(games)
        current_eval_policy = self._selected_eval_policy()
        manifest_seed_file = str(manifest.get("seed_file") or "")
        manifest_games = int(manifest.get("games") or 0)
        manifest_eval_policy = self._normalize_eval_policy_key(manifest.get("eval_policy") or "full")
        progress_lines = "\n".join(
            f"  {label}: {rows}/{total}"
            for label, rows, total in self._eval_set_manifest_progress(manifest)
        )
        started_at = manifest.get("started_at") or "?"

        if (
            manifest_seed_file == current_seed_file
            and manifest_games == int(games)
            and manifest_eval_policy == current_eval_policy
        ):
            choice = messagebox.askyesnocancel(
                "Resume Eval Set?",
                "An incomplete eval set is on disk from "
                f"{started_at} ({self._eval_policy_label(manifest_eval_policy)}):\n\n"
                f"{progress_lines}\n\n"
                "Yes  — resume and reuse the recorded games.\n"
                "No   — discard previous progress and start fresh.\n"
                "Cancel — abort the launch.",
            )
            if choice is None:
                return False
            if choice:
                return manifest
            self._clear_eval_set_manifest()
            return None

        choice = messagebox.askyesnocancel(
            "Previous Eval Set Found",
            "An incomplete eval set is on disk, but its parameters differ "
            "from this launch:\n\n"
            f"  Previous: {manifest_seed_file or '?'} × {manifest_games} games "
            f"({self._eval_policy_label(manifest_eval_policy)}, started {started_at})\n"
            f"{progress_lines}\n\n"
            f"  Current:  {current_seed_file} × {int(games)} games "
            f"({self._eval_policy_label(current_eval_policy)})\n\n"
            "Yes  — discard previous progress and start fresh with current settings.\n"
            "No   — keep the previous manifest on disk for later (abort this launch).\n"
            "Cancel — abort the launch.",
        )
        if choice is None or choice is False:
            return False
        self._clear_eval_set_manifest()
        return None

    def _eval_python_running_for_run(self, run_tag: str) -> bool | None:
        try:
            import psutil
        except ImportError:
            return None

        try:
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if not name.startswith("python"):
                        continue
                    tokens = [str(t) for t in (p.info.get("cmdline") or [])]
                    cmdline = " ".join(tokens)
                    if "eval_model.py" not in cmdline or run_tag not in tokens:
                        continue
                    return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            _logger.debug(f"Failed to scan eval python process for {run_tag}: {e}")
            return None
        return False

    def _wait_for_eval_complete(self, run_tag: str, games: int, proc: subprocess.Popen,
                                timeout_sec: int, restart_every: int = 0) -> str:
        start = time.time()
        last_progress_at = start
        last_count = -1
        count_at_launch = self._count_eval_rows_for_run(run_tag)
        exit_seen_at = None
        stall_timeout_sec = max(900, min(timeout_sec, 3600))
        exit_grace_sec = 120
        while self.running:
            count = self._count_eval_rows_for_run(run_tag)
            if count != last_count:
                _logger.info(f"Eval run {run_tag}: {count}/{games} games recorded")
                self._append_log("Eval Set", f"{run_tag}: {count}/{games} games recorded")
                last_count = count
                last_progress_at = time.time()
            if count >= games:
                return "complete"

            rc = proc.poll()
            if rc is not None:
                if exit_seen_at is None:
                    exit_seen_at = time.time()
                    _logger.warning(
                        f"Eval launcher process exited for {run_tag} before completion "
                        f"(rc={rc}, rows={count}/{games}); waiting for progress grace period"
                    )
                eval_running = self._eval_python_running_for_run(run_tag)
                games_this_launch = count - count_at_launch
                if (
                    eval_running is False
                    and restart_every > 0
                    and games_this_launch >= restart_every
                    and count < games
                ):
                    _logger.info(
                        f"Eval run {run_tag}: planned restart-every exit "
                        f"({games_this_launch} games this launch, {count}/{games} total)"
                    )
                    self._append_log(
                        "Eval Set",
                        f"RAM cleanup restart: {run_tag} at {count}/{games} games",
                    )
                    return "exited"
                if (
                    eval_running is False
                    and time.time() - exit_seen_at > exit_grace_sec
                    and time.time() - last_progress_at > exit_grace_sec
                ):
                    _logger.warning(
                        f"Eval run {run_tag} stopped progressing after launcher exit "
                        f"(rc={rc}, rows={count}/{games})"
                    )
                    self._append_log(
                        "Eval Set",
                        f"CRASH/EXIT: {run_tag} exited with rc={rc} at {count}/{games} games",
                    )
                    return "exited"

            if restart_every > 0 and rc is None:
                games_this_launch = count - count_at_launch
                if games_this_launch >= restart_every and count < games:
                    eval_running = self._eval_python_running_for_run(run_tag)
                    if eval_running is False:
                        _logger.info(
                            f"Eval run {run_tag}: planned restart-every exit "
                            f"(launcher still alive, {games_this_launch} games, {count}/{games} total)"
                        )
                        self._append_log(
                            "Eval Set",
                            f"RAM cleanup restart: {run_tag} at {count}/{games} games",
                        )
                        return "exited"

            elapsed = time.time() - start
            if elapsed > timeout_sec:
                _logger.warning(f"Eval run {run_tag} timed out after {timeout_sec}s with {count}/{games} games")
                self._append_log("Eval Set", f"TIMEOUT: {run_tag} only reached {count}/{games} games")
                return "timeout"
            if time.time() - last_progress_at > stall_timeout_sec:
                _logger.warning(
                    f"Eval run {run_tag} stalled for {stall_timeout_sec}s "
                    f"with {count}/{games} games recorded"
                )
                self._append_log("Eval Set", f"STALLED: {run_tag} stuck at {count}/{games} games")
                return "stalled"
            time.sleep(5.0)
        return "stopped"

    def _cleanup_eval_instance(self, proc: subprocess.Popen | None, label: str):
        try:
            if proc is not None:
                if proc in self.processes:
                    self.processes.remove(proc)
                if proc.poll() is None:
                    _logger.info(f"Closing eval STS process for {label}: PID {proc.pid}")
                    self._kill_process_tree(proc.pid)
        except Exception as e:
            _logger.warning(f"Failed to close eval STS process for {label}: {e}")
        # ModTheSpire can detach the actual game JVM from the Popen handle. Sweep
        # after each eval so the next policy gets a clean CommunicationMod launch.
        try:
            self._kill_orphan_sts_instances()
            self._kill_orphan_python_scripts()
        except Exception as e:
            _logger.warning(f"Eval cleanup sweep failed for {label}: {e}")

    def _launch_eval_set(self, games: int, *, resume_manifest: dict | None = None):
        _logger.info(f"_launch_eval_set: games={games} resume={bool(resume_manifest)}")
        completed_all = False
        try:
            verbose = bool(self.verbose_var.get())
            seed_file = self._selected_seed_file(games)
            seed_path = ROOT / seed_file if not os.path.isabs(seed_file) else Path(seed_file)
            self.root.after(0, lambda: self._add_log_tab("Eval Set"))
            time.sleep(0.1)

            if resume_manifest is None:
                if self.eval_generate_seed_var.get() or not seed_path.exists():
                    if not self._generate_seed_file(seed_file, games):
                        self.root.after(0, lambda: messagebox.showerror(
                            "Seed Generation Failed",
                            "Could not generate the eval seed file. Check Eval Set logs."
                        ))
                        return
                else:
                    self._append_log("Eval Set", f"Using existing seed file: {seed_file}")
            else:
                if not seed_path.exists():
                    self._append_log(
                        "Eval Set",
                        f"ERROR: resume requested but seed file {seed_file} is missing; aborting."
                    )
                    self.root.after(0, lambda: messagebox.showerror(
                        "Resume Failed",
                        f"Cannot resume: seed file {seed_file} is missing.\n"
                        "Uncheck 'Generate/overwrite before run' if disabled, or start fresh."
                    ))
                    return
                self._append_log("Eval Set", f"Resuming previous eval set; seed file: {seed_file}")

            eval_policy = self._selected_eval_policy()
            runs: list[dict] = []
            if resume_manifest is not None:
                ts = str(resume_manifest.get("ts") or datetime.now().strftime("%Y%m%d_%H%M%S"))
                eval_policy = self._normalize_eval_policy_key(resume_manifest.get("eval_policy") or "full")
                runs = [dict(r) for r in (resume_manifest.get("runs") or []) if isinstance(r, dict)]
                if not runs:
                    self._append_log("Eval Set", "Resume manifest had no runs; starting fresh.")
                    resume_manifest = None
                    eval_policy = self._selected_eval_policy()
            if resume_manifest is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                runs = self._eval_runs_for_policy(eval_policy, int(games), ts)

            started_at = (
                resume_manifest.get("started_at") if resume_manifest else None
            ) or datetime.now().isoformat(timespec="seconds")
            manifest_state = {
                "ts": ts,
                "games": int(games),
                "seed_file": seed_file,
                "eval_policy": eval_policy,
                "eval_policy_label": self._eval_policy_label(eval_policy),
                "started_at": started_at,
                "runs": runs,
            }
            self._save_eval_set_manifest(manifest_state)

            self._eval_set_started_at = time.time()
            games_desc = f"{games} games each" if len(runs) > 1 else f"{games} games"
            if resume_manifest is not None:
                progress_summary = ", ".join(
                    f"{label}={rows}/{total}"
                    for label, rows, total in self._eval_set_manifest_progress(manifest_state)
                )
                self._append_log(
                    "Eval Set",
                    f"Resuming fixed-seed eval set ts={ts}: {games_desc}, "
                    f"seed_file={seed_file}, policy={self._eval_policy_label(eval_policy)} "
                    f"({progress_summary})"
                )
            else:
                self._append_log(
                    "Eval Set",
                    f"Starting fixed-seed eval set: {games_desc}, seed_file={seed_file}, "
                    f"policy={self._eval_policy_label(eval_policy)}"
                )
            if len(runs) > 1:
                self._append_log(
                    "Eval Set",
                    "Running selected eval policies sequentially; concurrent eval is deferred "
                    "because CommunicationMod config, process cleanup, and log/stat isolation need "
                    "separate instance ownership first."
                )
            else:
                self._append_log("Eval Set", f"Running {runs[0]['label']} eval only.")

            restart_every = self._get_restart_sts_every()
            max_restarts = 3
            if restart_every > 0 and games > 0:
                max_restarts = max(3, (games // restart_every) + 3)
            completed_all = True
            for run in runs:
                if not self.running:
                    completed_all = False
                    break
                model = run["model"]
                if model and not (ROOT / model).exists():
                    msg = f"Skipping {run['label']}: missing checkpoint {model}"
                    _logger.warning(msg)
                    self._append_log("Eval Set", msg)
                    continue

                log_rel = f"Eval/eval_{run['key']}_{run['tag']}.log"
                log_path = ROOT / log_rel
                log_path.parent.mkdir(parents=True, exist_ok=True)
                tab_name = f"Eval {run['label']}"
                self.root.after(0, lambda name=tab_name: self._add_log_tab(name))
                run_tailer = LogTailer(log_path, lambda line, name=tab_name: self._append_log(name, line))
                run_tailer.start()
                self.tailers.append(run_tailer)

                status = "stopped"
                attempt = 0
                try:
                    while self.running and attempt <= max_restarts:
                        completed_rows = self._count_eval_rows_for_run(run["tag"])
                        if completed_rows >= games:
                            status = "complete"
                            break

                        resume_run = attempt > 0 or completed_rows > 0
                        cmd = build_eval_command(
                            policy=run["policy"], games=games, seed_file=seed_file,
                            run_tag=run["tag"], model=model,
                            top_actions=int(run["top_actions"]), verbose=verbose,
                            resume_run=resume_run, log_file=log_rel,
                            restart_every=restart_every,
                        )
                        attempt_label = "resume" if resume_run else "fresh"
                        self._append_log(
                            "Eval Set",
                            f"Launching {run['label']} eval ({attempt_label} attempt "
                            f"{attempt + 1}/{max_restarts + 1}): {run['tag']}"
                        )
                        self._append_log("Eval Set", f"Log file: {log_rel}")
                        self._append_log("Eval Set", cmd)
                        write_config(cmd, verbose=verbose)
                        proc = None
                        try:
                            proc = launch_sts(skip_launcher=True, jvm_heap=self._get_jvm_heap())
                            self.processes.append(proc)
                            _logger.info(f"Eval set {run['label']} STS launched: PID {proc.pid}")
                            self.root.after(0, lambda label=run['label']: self.status_var.set(
                                f"Evaluating {label} ({games} games)..."
                            ))
                            timeout = max(900, int(games) * 180)
                            status = self._wait_for_eval_complete(run["tag"], games, proc, timeout, restart_every=restart_every)
                        finally:
                            self._cleanup_eval_instance(proc, run["label"])
                            time.sleep(3.0)

                        if status == "complete":
                            break
                        if status == "stopped" or not self.running:
                            break
                        if attempt >= max_restarts:
                            completed_rows = self._count_eval_rows_for_run(run["tag"])
                            msg = (
                                f"{run['label']} eval failed after {max_restarts} restart(s): "
                                f"status={status}, rows={completed_rows}/{games}"
                            )
                            _logger.error(msg)
                            self._append_log("Eval Set", msg)
                            break

                        attempt += 1
                        completed_rows = self._count_eval_rows_for_run(run["tag"])
                        msg = (
                            f"{run['label']} eval {status}; relaunching with --resume-run "
                            f"from {completed_rows}/{games} recorded games"
                        )
                        _logger.warning(msg)
                        self._append_log("Eval Set", msg)
                finally:
                    run_tailer.stop()
                    if run_tailer in self.tailers:
                        self.tailers.remove(run_tailer)

                self._append_log(
                    "Eval Set",
                    f"{run['label']} eval {'complete' if status == 'complete' else 'stopped/failed'} ({status})"
                )
                if status != "complete" and self.running:
                    completed_all = False
                    break
                if status != "complete":
                    completed_all = False

            self.running = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.graceful_btn.configure(state="disabled"))
            final_status = "Eval set complete" if completed_all else "Eval set stopped/failed"
            self.root.after(0, lambda status=final_status: self.status_var.set(status))
            if completed_all:
                self._clear_eval_set_manifest()
                self._append_log("Eval Set", "Fixed-seed eval set finished.")
            else:
                self._append_log(
                    "Eval Set",
                    "Fixed-seed eval set stopped/failed. "
                    "Progress saved to Eval/eval_set_state.json — relaunch to resume."
                )
            _logger.info(f"_launch_eval_set complete: completed_all={completed_all}")
        except Exception as e:
            _logger.error(f"_launch_eval_set FAILED: {traceback.format_exc()}")
            self._append_log("Eval Set", f"ERROR launching eval set: {e}")
            self.running = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.graceful_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.status_var.set("Eval set failed"))

    def _launch_single(self, mode: str, games: int = 20, ppo_games: int = 200, bc_epochs: int = 50):
        _logger.info(f"_launch_single: mode={mode}, games={games}, ppo_games={ppo_games}, bc_epochs={bc_epochs}")
        try:
            verbose = bool(self.verbose_var.get())
            cmd = build_command(mode, games=games, bc_games=games,
                                ppo_games=ppo_games, bc_epochs=bc_epochs,
                                ent_coef=self._ent_value, verbose=verbose)
            write_config(cmd, verbose=verbose)

            _mode_info = {
                "train":   ("Training",   "logs/train_debug.log"),
                "bc_ppo":  ("BC\u2192PPO",     "logs/train_bc_ppo_debug.log"),
                "bc":      ("BC",          "logs/bc_debug.log"),
                "eval":    ("Evaluation",  "Eval/eval_debug.log"),
                "logger":  ("Logger",      "logs/game_logger_debug.log"),
            }
            log_name, log_file = _mode_info.get(mode, ("Training", "logs/train_debug.log"))
            _logger.info(f"Log tab: {log_name}, log file: {log_file}")

            self.root.after(0, lambda: self._add_log_tab(log_name))
            time.sleep(0.1)

            self._append_log(log_name, f"Config written, launching STS ({mode})...")
            proc = launch_sts(jvm_heap=self._get_jvm_heap())
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

    def _launch_play(self):
        _logger.info("_launch_play: launching game with no AI command")
        try:
            write_config(None, verbose=bool(self.verbose_var.get()))
            self._append_log("All", "Config written (no AI command), launching STS...")
            proc = launch_sts()
            self.processes.append(proc)
            self._append_log("All", f"STS launched (PID {proc.pid}) — play/configure mods freely")
            self.root.after(0, lambda: self.status_var.set("Running (Play Game)"))
            _logger.info(f"_launch_play complete: PID {proc.pid}")
        except Exception as e:
            _logger.error(f"_launch_play FAILED: {traceback.format_exc()}")
            self._append_log("All", f"ERROR launching game: {e}")

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

    def _wait_for_worker_process(self, worker_id: int, tab_name: str,
                                 timeout: int = 120) -> bool:
        """Block until a worker-specific Python process appears."""
        _logger.info(f"_wait_for_worker_process: worker={worker_id}, timeout={timeout}s")
        self._append_log(tab_name, f"Waiting for worker {worker_id} process to start...")
        start = time.time()
        while time.time() - start < timeout:
            if not self.running:
                return False
            _java_pids, py_pids = self._find_pids_for_worker(worker_id)
            if py_pids:
                elapsed = time.time() - start
                self._append_log(tab_name, f"Worker process detected ({elapsed:.0f}s).")
                return True
            time.sleep(1.0)
        elapsed = time.time() - start
        self._append_log(tab_name, f"Timed out waiting for worker process ({elapsed:.0f}s).")
        return False

    def _launch_consume(self):
        """Launch the offline trainer only (no game workers) to consume accumulated rollouts."""
        try:
            self._with_trainer = True
            self._trainer_cmd = None
            self._n_workers = 0

            self._append_log("All", "Starting offline trainer (consume-only mode)...")
            self.root.after(0, lambda: self._add_log_tab("Trainer"))
            time.sleep(0.1)

            trainer_cmd = [
                str(VENV_PYTHON), str(SCRIPTS / "train_offline.py"),
                "--model", str(ROOT / "models" / "ppo_sts.pt"),
                "--data", str(ROOT / "rollouts_shared"),
                "--delete-consumed",
                "--batch-games", "8",
                "--lr", "3e-5",
                "--bc-coef", "0.10",
                "--max-rollout-lag", "4",
                "--ent-coef", str(self._ent_value),
                "--device", "gpu" if self.gpu_var.get() else "cpu",
            ]
            if self._ent_manual_override:
                trainer_cmd.append("--override-ent-coef")
            if self.auto_tune_var.get():
                trainer_cmd.append("--auto-tune")
            if self.verbose_var.get():
                trainer_cmd.append("--verbose")
            self._trainer_cmd = trainer_cmd
            _logger.info(f"Consume trainer command: {trainer_cmd}")
            self.trainer_proc = subprocess.Popen(
                trainer_cmd, cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            _logger.info(f"Offline trainer started (consume mode): PID {self.trainer_proc.pid}")
            self._append_log("Trainer", f"Offline trainer started (PID {self.trainer_proc.pid})")

            tailer = LogTailer(ROOT / "logs" / "train_offline_debug.log",
                               lambda line: self._append_log("Trainer", line))
            tailer.start()
            self.tailers.append(tailer)

            rollout_dir = ROOT / "rollouts_shared"
            while self.running:
                time.sleep(5.0)
                remaining = len(list(rollout_dir.glob("*.npz")))
                if self.trainer_proc is not None:
                    rc = self.trainer_proc.poll()
                    if rc is not None:
                        self._append_log("Trainer",
                            f"Trainer exited (code {rc}), {remaining} rollouts remaining")
                        if remaining > 0 and self.running:
                            _logger.info("Trainer exited with rollouts remaining; restarting")
                            self._append_log("Trainer", "Restarting trainer...")
                            self.trainer_proc = subprocess.Popen(
                                trainer_cmd, cwd=str(ROOT),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                            )
                            continue
                self.root.after(0, lambda r=remaining: self.status_var.set(
                    f"Consuming rollouts... {r} remaining"
                ))
                if remaining == 0:
                    self._append_log("All", "All rollouts consumed.")
                    self.root.after(0, lambda: self.status_var.set("All rollouts consumed"))
                    self.running = False
                    self.root.after(0, lambda: self.start_btn.configure(state="normal"))
                    self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
                    self.root.after(0, lambda: self.graceful_btn.configure(state="disabled"))
                    break
        except Exception as e:
            _logger.error(f"_launch_consume error: {e}", exc_info=True)
            self._append_log("All", f"ERROR: {e}")
            self.running = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.graceful_btn.configure(state="disabled"))

    def _launch_parallel(self, n_workers: int, with_trainer: bool = True,
                         worker_ids: list[int] | None = None,
                         worker_targets: dict[int, int] | None = None):
        selected_ids = list(worker_ids or range(1, n_workers + 1))
        targets = dict(worker_targets or {})
        _logger.info(f"_launch_parallel: n_workers={n_workers}, with_trainer={with_trainer}, "
                     f"worker_ids={selected_ids}, targets={targets}")
        try:
            self.worker_launcher_pids.clear()
            self.worker_sts_pids.clear()
            self._worker_commands.clear()
            self._worker_modes.clear()
            self._worker_demo_paths.clear()
            self._worker_checkpoint_paths.clear()
            self._worker_restarts.clear()
            self._worker_launch_time.clear()
            self._worker_games_at_launch.clear()
            self._worker_game_targets = dict(targets)
            self._parallel_launch_started_at = time.time()
            self._n_workers = 0
            self._with_trainer = with_trainer
            self._trainer_cmd = None

            for i in selected_ids:
                log_file = ROOT / "logs" / f"worker_{i}_debug.log"
                heartbeat_file = self._worker_heartbeat_path(i)
                done_file = self._worker_done_marker_path(i)
                try:
                    if log_file.exists():
                        _logger.debug(f"Deleting stale log: {log_file} "
                                      f"(size={log_file.stat().st_size})")
                        log_file.unlink()
                    if heartbeat_file.exists():
                        _logger.debug(f"Deleting stale heartbeat: {heartbeat_file}")
                        heartbeat_file.unlink()
                    if done_file.exists():
                        _logger.debug(f"Deleting stale done marker: {done_file}")
                        done_file.unlink()
                except OSError as e:
                    _logger.warning(f"Failed to delete stale worker files for {i}: {e}")

            if with_trainer:
                self._append_log("All", "Starting offline trainer...")
                self.root.after(0, lambda: self._add_log_tab("Trainer"))
                time.sleep(0.1)

                trainer_cmd = [
                    str(VENV_PYTHON), str(SCRIPTS / "train_offline.py"),
                    "--model", str(ROOT / "models" / "ppo_sts.pt"),
                    "--data", str(ROOT / "rollouts_shared"),
                    "--delete-consumed",
                    "--batch-games", "8",
                    "--lr", "3e-5",
                    "--bc-coef", "0.10",
                    "--max-rollout-lag", "4",
                    "--ent-coef", str(self._ent_value),
                    "--device", "gpu" if self.gpu_var.get() else "cpu",
                ]
                if self._ent_manual_override:
                    trainer_cmd.append("--override-ent-coef")
                if self.auto_tune_var.get():
                    trainer_cmd.append("--auto-tune")
                if self.verbose_var.get():
                    trainer_cmd.append("--verbose")
                self._trainer_cmd = trainer_cmd
                _logger.info(f"Trainer command: {trainer_cmd}")
                self.trainer_proc = subprocess.Popen(
                    trainer_cmd, cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
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

            for idx, i in enumerate(selected_ids, start=1):
                if not self.running:
                    _logger.info(f"_launch_parallel: aborted at worker {i} (self.running=False)")
                    break

                tab_name = f"Worker {i}"
                self.root.after(0, lambda tn=tab_name: self._add_log_tab(tn))
                time.sleep(0.1)

                target_games = int(self._worker_game_targets.get(i, 0) or 0)
                restart_every = self._get_restart_sts_every()
                cmd = build_command("worker", worker_id=i,
                                    worker_games=target_games,
                                    verbose=bool(self.verbose_var.get()),
                                    restart_every=restart_every)
                self._worker_commands[i] = cmd
                self._worker_modes[i] = "worker"
                write_config(cmd, verbose=bool(self.verbose_var.get()))
                target_msg = (
                    f" (target {target_games} games)" if target_games > 0 else ""
                )
                self._append_log(
                    tab_name,
                    f"Config written for worker {i}{target_msg}, launching STS...",
                )

                before_java_pids = self._sts_java_pids()
                proc = launch_sts(skip_launcher=True, jvm_heap=self._get_jvm_heap())
                self.processes.append(proc)
                self.worker_launcher_pids[i] = proc.pid
                self.worker_sts_pids[i] = set()
                self._worker_restarts[i] = 0
                self._worker_games_at_launch[i] = 0
                self._worker_launch_time[i] = time.time()
                _logger.info(f"Worker {i}: launcher PID={proc.pid}, "
                             f"total processes tracked={len(self.processes)}")
                self._append_log(tab_name, f"STS instance launched (PID {proc.pid})")

                threading.Thread(
                    target=self._capture_sts_pids_for_worker,
                    args=(proc.pid, i, before_java_pids),
                    daemon=True,
                ).start()
                _logger.debug(f"Worker {i}: PID capture thread started")

                log_file = ROOT / "logs" / f"worker_{i}_debug.log"
                tailer = LogTailer(log_file, lambda line, n=tab_name: self._append_log(n, line))
                tailer.start()
                self.tailers.append(tailer)

                self.root.after(0, lambda idx=i: self.status_var.set(
                    f"Waiting for worker {idx}/{n_workers} to be ready..."))

                if idx < len(selected_ids):
                    ready = self._wait_for_ready(log_file, tab_name, timeout=120)
                    _logger.info(f"Worker {i} ready={ready}, proceeding to next worker")

            if self.running:
                self._n_workers = n_workers
                suffix = "workers + trainer" if with_trainer else "collectors (no trainer)"
                _logger.info(f"Launched worker ids {selected_ids} of {n_workers} slots ({suffix})")
                _logger.info(f"Launcher PIDs: {self.worker_launcher_pids}")
                _logger.info(f"STS PIDs (captured so far): {dict(self.worker_sts_pids)}")
                self.root.after(0, lambda: self.status_var.set(
                    f"Running ({len(selected_ids)}/{n_workers} {suffix})"))
                threading.Thread(target=self._monitor_workers, daemon=True).start()
                _logger.info("Worker health monitor started")
        except Exception as e:
            _logger.error(f"_launch_parallel FAILED: {traceback.format_exc()}")
            self._append_log("All", f"ERROR in parallel launch: {e}")

    def _launch_bc_collect(self, total_slots: int, total_games: int,
                           worker_ids: list[int], *,
                           reset_existing: bool = True):
        _logger.info(f"_launch_bc_collect: total_slots={total_slots}, "
                     f"total_games={total_games}, worker_ids={worker_ids}, "
                     f"reset_existing={reset_existing}")
        try:
            self.worker_launcher_pids.clear()
            self.worker_sts_pids.clear()
            self._worker_commands.clear()
            self._worker_modes.clear()
            self._worker_demo_paths.clear()
            self._worker_checkpoint_paths.clear()
            self._worker_restarts.clear()
            self._worker_launch_time.clear()
            self._parallel_launch_started_at = time.time()
            self._n_workers = total_slots
            self._with_trainer = False
            self._trainer_cmd = None

            demo_dir = ROOT / "bc_demos_shared"
            demo_dir.mkdir(exist_ok=True)
            (ROOT / "logs").mkdir(exist_ok=True)
            games_per_worker = int((int(total_games) + int(total_slots) - 1) // int(total_slots))

            self.root.after(0, lambda: self._add_log_tab("BC Collect"))
            time.sleep(0.1)
            self._append_log(
                "BC Collect",
                f"Starting BC demo collection: slots={total_slots}, "
                f"launching={worker_ids}, games_per_slot={games_per_worker}"
            )

            bc_log = ROOT / "logs" / "bc_debug.log"
            tailer = LogTailer(bc_log, lambda line: self._append_log("BC Collect", line))
            tailer.start()
            self.tailers.append(tailer)

            for idx, worker_id in enumerate(worker_ids, start=1):
                if not self.running:
                    _logger.info(f"_launch_bc_collect aborted at worker {worker_id}")
                    break

                tab_name = f"BC {worker_id}"
                self.root.after(0, lambda tn=tab_name: self._add_log_tab(tn))
                time.sleep(0.1)

                demo_rel = f"bc_demos_shared/bc_demo_worker_{worker_id}.npz"
                checkpoint_rel = f"bc_demos_shared/bc_worker_{worker_id}_progress.npz"
                demo_path = ROOT / demo_rel
                checkpoint_path = ROOT / checkpoint_rel

                if reset_existing:
                    for path in (demo_path, checkpoint_path):
                        try:
                            if path.exists():
                                _logger.info(f"Deleting stale BC worker file before fresh launch: {path}")
                                path.unlink()
                        except OSError as e:
                            _logger.warning(f"Failed to delete stale BC file {path}: {e}")

                cmd = build_bc_collect_command(
                    worker_id,
                    games_per_worker,
                    demo_save=demo_rel,
                    checkpoint=checkpoint_rel,
                    verbose=bool(self.verbose_var.get()),
                )
                self._worker_commands[worker_id] = cmd
                self._worker_modes[worker_id] = "bc_collect"
                self._worker_demo_paths[worker_id] = demo_path
                self._worker_checkpoint_paths[worker_id] = checkpoint_path
                self._worker_restarts[worker_id] = 0

                write_config(cmd, verbose=bool(self.verbose_var.get()))
                self._append_log(tab_name, f"Config written for BC worker {worker_id}, launching STS...")

                before_java_pids = self._sts_java_pids()
                proc = launch_sts(skip_launcher=True, jvm_heap=self._get_jvm_heap())
                self.processes.append(proc)
                self.worker_launcher_pids[worker_id] = proc.pid
                self.worker_sts_pids[worker_id] = set()
                self._worker_launch_time[worker_id] = time.time()
                _logger.info(f"BC worker {worker_id}: launcher PID={proc.pid}")
                self._append_log(tab_name, f"STS instance launched (PID {proc.pid})")

                threading.Thread(
                    target=self._capture_sts_pids_for_worker,
                    args=(proc.pid, worker_id, before_java_pids),
                    daemon=True,
                ).start()

                self.root.after(0, lambda wid=worker_id: self.status_var.set(
                    f"Waiting for BC worker {wid}/{total_slots} to start..."))

                if idx < len(worker_ids):
                    ready = self._wait_for_worker_process(worker_id, tab_name, timeout=120)
                    _logger.info(f"BC worker {worker_id} process-ready={ready}, proceeding")

            if self.running:
                self.root.after(0, lambda: self.status_var.set(
                    f"Running BC collection ({len(worker_ids)}/{total_slots} workers)"))
                threading.Thread(target=self._monitor_workers, daemon=True).start()
                _logger.info("BC collection health monitor started")
        except Exception as e:
            _logger.error(f"_launch_bc_collect FAILED: {traceback.format_exc()}")
            self._append_log("All", f"ERROR in BC collection launch: {e}")

    def _launch_bc_train_from_demos(self, bc_epochs: int):
        _logger.info(f"_launch_bc_train_from_demos: bc_epochs={bc_epochs}")
        try:
            demo_dir = ROOT / "bc_demos_shared"
            demo_dir.mkdir(exist_ok=True)
            self.root.after(0, lambda: self._add_log_tab("BC Train"))
            time.sleep(0.1)
            tailer = LogTailer(ROOT / "logs" / "bc_debug.log",
                               lambda line: self._append_log("BC Train", line))
            tailer.start()
            self.tailers.append(tailer)

            cmd = [
                str(VENV_PYTHON), str(SCRIPTS / "behavior_clone.py"),
                "--train-demo-dir", str(demo_dir),
                "--save", "models/ppo_sts_bc.pt",
                "--demo-save", "models/ppo_sts_bc_demos.npz",
                "--epochs", str(int(bc_epochs)),
                "--lr", "5e-4",
                "--batch-size", "256",
                "--val-split", "0.10",
                "--patience", "12",
                "--weight-decay", "1e-5",
                "--label-smoothing", "0.02",
            ]
            if self.verbose_var.get():
                cmd.append("--verbose")
            self._append_log("BC Train", f"Training from demos in {demo_dir}")
            _logger.info(f"BC train command: {cmd}")
            self.trainer_proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            _logger.info(f"BC train started: PID {self.trainer_proc.pid}")
            self._append_log("BC Train", f"BC trainer started (PID {self.trainer_proc.pid})")
            rc = self.trainer_proc.wait()
            _logger.info(f"BC train exited: rc={rc}")
            self._append_log("BC Train", f"BC training exited with code {rc}")
            self.running = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.graceful_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.status_var.set(
                "BC demo training complete" if rc == 0 else "BC demo training failed"
            ))
        except Exception as e:
            _logger.error(f"_launch_bc_train_from_demos FAILED: {traceback.format_exc()}")
            self._append_log("BC Train", f"ERROR training from demos: {e}")

    # ----- Worker health monitor -----

    _MAX_RESTARTS = 20
    _WORKER_BOOT_GRACE_SEC = 180
    _WORKER_HEARTBEAT_TIMEOUT_SEC = 360
    _MONITOR_INTERVAL_SEC = 30

    def _monitor_workers(self):
        """Periodically check workers/trainer and relaunch unhealthy pieces."""
        _logger.info("_monitor_workers: starting health check loop")
        while self.running and not getattr(self, "_graceful_stopping", False):
            time.sleep(self._MONITOR_INTERVAL_SEC)
            if not self.running or getattr(self, "_graceful_stopping", False):
                break
            self._monitor_trainer()
            for worker_id in list(self._worker_commands.keys()):
                if not self.running:
                    break
                worker_mode = self._worker_modes.get(worker_id, "worker")
                if worker_mode != "bc_collect" and self._worker_done(worker_id):
                    target = self._worker_game_targets.get(worker_id, 0)
                    _logger.info(f"Worker {worker_id}: game target reached "
                                 f"(target={target}); closing worker")
                    self._append_log(
                        f"Worker {worker_id}",
                        f"Game target reached ({target} games); closing STS instance.",
                    )
                    self._stop_single_worker(worker_id)
                    self._worker_commands.pop(worker_id, None)
                    self._worker_modes.pop(worker_id, None)
                    self._worker_game_targets.pop(worker_id, None)
                    continue
                if worker_mode == "bc_collect" and self._bc_worker_complete(worker_id):
                    _logger.info(f"BC worker {worker_id}: demo file complete; stopping worker")
                    self._append_log(f"BC {worker_id}", "Demo file complete; closing STS instance.")
                    self._stop_single_worker(worker_id)
                    self._worker_commands.pop(worker_id, None)
                    self._worker_modes.pop(worker_id, None)
                    continue
                launch_t = self._worker_launch_time.get(worker_id, 0)
                age = time.time() - launch_t
                if age < self._WORKER_BOOT_GRACE_SEC:
                    continue
                java_pids, py_pids = self._find_pids_for_worker(worker_id)
                java_pids |= self._live_cached_java_pids(worker_id)
                heartbeat_age = self._worker_heartbeat_age(worker_id)
                missing_piece = not java_pids or not py_pids
                stale_heartbeat = False
                if worker_mode != "bc_collect":
                    stale_heartbeat = (
                        heartbeat_age is None
                        or heartbeat_age > self._WORKER_HEARTBEAT_TIMEOUT_SEC
                    )
                if missing_piece or stale_heartbeat:
                    restarts = self._worker_restarts.get(worker_id, 0)
                    if restarts >= self._MAX_RESTARTS:
                        if restarts == self._MAX_RESTARTS:
                            _logger.warning(f"Worker {worker_id}: max restarts ({self._MAX_RESTARTS}) reached, giving up")
                            self._append_log(f"BC {worker_id}" if worker_mode == "bc_collect" else f"Worker {worker_id}",
                                             f"Max restarts ({self._MAX_RESTARTS}) reached — not relaunching.")
                            self._worker_restarts[worker_id] = restarts + 1
                        continue
                    reason = (
                        f"java={sorted(java_pids)} py={sorted(py_pids)} "
                        f"heartbeat_age={heartbeat_age if heartbeat_age is not None else 'missing'}"
                    )
                    _logger.warning(f"Worker {worker_id}: unhealthy ({reason}), relaunching "
                                    f"(restart #{restarts + 1})")
                    self._append_log(f"BC {worker_id}" if worker_mode == "bc_collect" else f"Worker {worker_id}",
                                     f"Unhealthy worker detected — relaunching "
                                     f"(restart #{restarts + 1}).")
                    self._stop_single_worker(worker_id)
                    self._relaunch_worker(worker_id)
                    time.sleep(15.0)
            self._kill_excess_unowned_sts_instances("monitor")
            current_mode = getattr(self, "_current_mode", "")
            if self.running and not self._worker_commands and current_mode == "bc_collect":
                _logger.info("All BC collection workers completed")
                self._append_log("BC Collect", "All selected BC workers completed.")
                self.running = False
                self.root.after(0, lambda: self.start_btn.configure(state="normal"))
                self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
                self.root.after(0, lambda: self.graceful_btn.configure(state="disabled"))
                self.root.after(0, lambda: self.status_var.set("BC collection complete"))
                break
            if (
                self.running
                and not self._worker_commands
                and current_mode in ("worker", "collect")
                and self._n_workers > 0
            ):
                _logger.info("All parallel workers reached game target")
                self._append_log(
                    "All", "All workers reached game target. Stopping trainer."
                )
                self.root.after(
                    0, lambda: self.status_var.set("Game target reached; stopping...")
                )
                self.root.after(0, self._stop)
                break
        _logger.info("_monitor_workers: loop ended")

    def _bc_worker_complete(self, worker_id: int) -> bool:
        demo_path = self._worker_demo_paths.get(worker_id)
        if demo_path is None:
            return False
        try:
            return demo_path.exists() and demo_path.stat().st_size > 0
        except OSError:
            return False

    def _monitor_trainer(self):
        """Restart the offline trainer if it exits during parallel training."""
        if not self._with_trainer or self._trainer_cmd is None:
            return
        if self.trainer_proc is not None and self.trainer_proc.poll() is None:
            return
        rc = self.trainer_proc.poll() if self.trainer_proc is not None else None
        _logger.warning(f"Offline trainer is not running (rc={rc}); restarting")
        self._append_log("Trainer", "Offline trainer stopped — restarting...")
        try:
            self.trainer_proc = subprocess.Popen(
                self._trainer_cmd, cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            _logger.info(f"Offline trainer restarted: PID {self.trainer_proc.pid}")
            self._append_log("Trainer", f"Offline trainer restarted (PID {self.trainer_proc.pid})")
        except Exception as e:
            _logger.error(f"Failed to restart offline trainer: {e}")
            self._append_log("Trainer", f"ERROR restarting trainer: {e}")

    def _worker_heartbeat_path(self, worker_id: int) -> Path:
        return ROOT / "logs" / f"worker_{worker_id}_heartbeat.txt"

    def _worker_done_marker_path(self, worker_id: int) -> Path:
        return ROOT / "logs" / f"worker_{worker_id}_done.txt"

    def _worker_done(self, worker_id: int) -> bool:
        try:
            return self._worker_done_marker_path(worker_id).exists()
        except OSError:
            return False

    def _worker_heartbeat_age(self, worker_id: int) -> float | None:
        path = self._worker_heartbeat_path(worker_id)
        try:
            if not path.exists():
                return None
            return max(0.0, time.time() - path.stat().st_mtime)
        except OSError as e:
            _logger.warning(f"Could not stat heartbeat for worker {worker_id}: {e}")
            return None

    def _live_cached_java_pids(self, worker_id: int) -> set[int]:
        try:
            import psutil
        except ImportError:
            return set()
        live: set[int] = set()
        for pid in set(self.worker_sts_pids.get(worker_id, set())):
            try:
                p = psutil.Process(pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    live.add(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if live:
            self.worker_sts_pids[worker_id] = live
        return live

    def _relaunch_worker(self, worker_id: int):
        """Relaunch a single crashed worker using --skip-launcher."""
        try:
            if self._worker_done(worker_id):
                _logger.info(
                    f"_relaunch_worker: worker {worker_id} has done marker; "
                    "skipping relaunch"
                )
                return
            cmd = self._worker_commands.get(worker_id)
            if not cmd:
                _logger.error(f"_relaunch_worker: no stored command for worker {worker_id}")
                return
            original_target = self._worker_game_targets.get(worker_id, 0)
            completed = self._count_worker_completed_games(worker_id)
            last_launch_completed = self._worker_games_at_launch.get(worker_id, 0)
            games_since_launch = completed - last_launch_completed
            restart_every = self._get_restart_sts_every()
            is_planned_restart = (
                restart_every > 0 and games_since_launch >= restart_every
            )
            if original_target > 0:
                remaining = original_target - completed
                if remaining <= 0:
                    _logger.info(
                        f"_relaunch_worker: worker {worker_id} already completed "
                        f"{completed}/{original_target} games; writing done marker"
                    )
                    self._worker_done_marker_path(worker_id).write_text(
                        f"completed={completed}\ttarget={original_target}\n",
                        encoding="utf-8",
                    )
                    return
                cmd = build_command(
                    "worker", worker_id=worker_id,
                    worker_games=remaining,
                    verbose=bool(self.verbose_var.get()),
                    restart_every=restart_every,
                )
                self._worker_commands[worker_id] = cmd
                _logger.info(
                    f"_relaunch_worker: worker {worker_id} adjusted target: "
                    f"{completed} done, {remaining} remaining (was {original_target})"
                )
            elif restart_every > 0:
                cmd = build_command(
                    "worker", worker_id=worker_id,
                    verbose=bool(self.verbose_var.get()),
                    restart_every=restart_every,
                )
                self._worker_commands[worker_id] = cmd
            worker_mode = self._worker_modes.get(worker_id, "worker")
            tab_name = f"BC {worker_id}" if worker_mode == "bc_collect" else f"Worker {worker_id}"
            write_config(cmd, verbose=bool(self.verbose_var.get()))
            before_java_pids = self._sts_java_pids()
            proc = launch_sts(skip_launcher=True, jvm_heap=self._get_jvm_heap())
            self.processes.append(proc)
            self.worker_launcher_pids[worker_id] = proc.pid
            self.worker_sts_pids[worker_id] = set()
            if is_planned_restart:
                _logger.info(
                    f"Worker {worker_id}: planned RAM cleanup restart "
                    f"({games_since_launch} games since last launch)"
                )
            else:
                self._worker_restarts[worker_id] = self._worker_restarts.get(worker_id, 0) + 1
            self._worker_games_at_launch[worker_id] = completed
            self._worker_launch_time[worker_id] = time.time()
            _logger.info(f"Worker {worker_id} relaunched: PID {proc.pid}")
            self._append_log(
                tab_name,
                f"{'RAM cleanup restart' if is_planned_restart else 'Relaunched'} "
                f"STS instance (PID {proc.pid}, --skip-launcher)",
            )

            threading.Thread(
                target=self._capture_sts_pids_for_worker,
                args=(proc.pid, worker_id, before_java_pids),
                daemon=True,
            ).start()
        except Exception as e:
            _logger.error(f"_relaunch_worker {worker_id} FAILED: {e}")
            tab_name = f"BC {worker_id}" if self._worker_modes.get(worker_id) == "bc_collect" else f"Worker {worker_id}"
            self._append_log(tab_name, f"ERROR relaunching: {e}")

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

    _GAME_END_RE = re.compile(
        r"(?m)^\d{4}-\d{2}-\d{2}T.*"
        r"(?:Game #\d+|PPO game #\d+|BC game #\d+|Demo game #\d+) "
        r"ended\b"
    )
    _EVAL_GAME_RE = re.compile(
        r"(?m)^\d{4}-\d{2}-\d{2}T.*Game #\d+: floor="
    )

    def _count_completed_games_in_log(self, text: str, mode: str = "") -> int:
        """Count full-run completions without matching fight/act transitions."""
        if mode == "eval":
            return len(self._EVAL_GAME_RE.findall(text))
        return len(self._GAME_END_RE.findall(text))

    def _wait_single_then_stop(self, mode: str):
        """Wait for single-instance mode to finish its current game."""
        log_map = {
            "train":   "logs/train_debug.log",
            "bc_ppo":  "logs/train_bc_ppo_debug.log",
            "bc":      "logs/bc_debug.log",
            "eval":    "Eval/eval_debug.log",
            "logger":  "logs/game_logger_debug.log",
        }
        log_file = ROOT / log_map.get(mode, "logs/train_debug.log")
        _logger.info(f"_wait_single_then_stop: mode={mode}, log={log_file}, "
                     f"exists={log_file.exists()}")

        baseline_ended = 0
        if log_file.exists():
            try:
                text = log_file.read_text(encoding="utf-8", errors="replace")
                baseline_ended = self._count_completed_games_in_log(text, mode)
                _logger.info(f"Baseline completed-game count: {baseline_ended} "
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
                    current_ended = self._count_completed_games_in_log(text, mode)
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
        worker_ids = sorted(self._worker_commands.keys())
        if not worker_ids:
            worker_ids = list(range(1, int(getattr(self, "_n_workers", 0) or 0) + 1))

        for i in worker_ids:
            log_file = log_dir / f"worker_{i}_debug.log"
            if log_file.exists():
                try:
                    text = log_file.read_text(encoding="utf-8", errors="replace")
                    ended = self._count_completed_games_in_log(text, "worker")
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
                        current_ended = self._count_completed_games_in_log(text, "worker")
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

    def _is_sts_java_process(self, proc) -> bool:
        """Best-effort test for Java processes that belong to Slay the Spire/ModTheSpire."""
        try:
            name = proc.name().lower()
            if name not in ("java.exe", "javaw.exe"):
                return False
            sts_dir_lower = str(STS_DIR).lower()
            cwd = ""
            try:
                cwd = (proc.cwd() or "").lower()
            except Exception:
                cwd = ""
            cmdline = " ".join(proc.cmdline() or []).lower()
            sts_keywords = ("slaythespire", "desktop-1.0.jar",
                            "mts-launcher", "modthespire")
            return (sts_dir_lower in cwd) or any(kw in cmdline for kw in sts_keywords)
        except Exception:
            return False

    def _sts_java_pids(self) -> set[int]:
        """Return all visible STS/ModTheSpire java PIDs."""
        try:
            import psutil
        except ImportError:
            return set()
        out: set[int] = set()
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if self._is_sts_java_process(p):
                    out.add(int(p.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return out

    def _kill_excess_unowned_sts_instances(self, reason: str = "monitor") -> int:
        """Kill extra STS/MTS java processes not attached to live worker Python.

        ModTheSpire can detach from the launcher, so relaunching a worker after a
        crash used to leave the old game/menu alive. During parallel training we
        expect roughly one Java process per worker. If there are more, prefer the
        Java PIDs that are parents of live rollout_worker.py processes and kill
        newer unmatched Java processes from this training run.
        """
        if not getattr(self, "running", False) or int(getattr(self, "_n_workers", 0) or 0) <= 0:
            return 0
        try:
            import psutil
        except ImportError:
            return 0

        all_java = self._sts_java_pids()
        if len(all_java) <= int(getattr(self, "_n_workers", 0) or 0):
            return 0

        worker_parent_java: set[int] = set()
        for worker_id in list(self._worker_commands.keys()):
            java_pids, _py_pids = self._find_pids_for_worker(worker_id)
            worker_parent_java |= java_pids
            if java_pids:
                self.worker_sts_pids.setdefault(worker_id, set()).update(java_pids)

        candidates = sorted(all_java - worker_parent_java)
        killed = 0
        for pid in candidates:
            # Keep pre-existing manually-opened STS sessions if the user had one
            # before starting the parallel run. Kill only processes born during
            # this run unless we cannot read create_time.
            try:
                proc = psutil.Process(pid)
                started = float(proc.create_time())
                run_start = float(getattr(self, "_parallel_launch_started_at", 0.0) or 0.0)
                if run_start and started < run_start - 10.0:
                    _logger.info(f"Skipping pre-existing STS java PID {pid} during {reason}")
                    continue
                _logger.warning(f"Killing excess STS/MTS java PID {pid} during {reason}; "
                                f"all={sorted(all_java)} worker_parent_java={sorted(worker_parent_java)}")
                proc.kill()
                killed += 1
                # Remove from any stale cache entry.
                for cached in self.worker_sts_pids.values():
                    cached.discard(pid)
                if len(all_java) - killed <= int(getattr(self, "_n_workers", 0) or 0):
                    break
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                _logger.warning(f"Access denied killing excess STS java PID {pid}")
            except Exception as e:
                _logger.warning(f"Failed to kill excess STS java PID {pid}: {e}")
        return killed

    def _find_pids_for_worker(self, worker_id: int) -> tuple[set[int], set[int]]:
        """Return (java_pids, python_pids) belonging to a specific worker."""
        java_pids: set[int] = set()
        py_pids: set[int] = set()

        try:
            import psutil
        except ImportError:
            _logger.warning("_find_pids_for_worker: psutil not available")
            return java_pids, py_pids

        worker_mode = self._worker_modes.get(worker_id, "worker")
        if worker_mode == "bc_collect":
            worker_script = "behavior_clone.py"
            id_flag = "--worker-id"
        else:
            worker_script = "rollout_worker.py"
            id_flag = "--id"
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
                if worker_mode == "bc_collect" and "--collect-only" not in tokens:
                    continue
                if id_flag not in tokens:
                    continue
                try:
                    if str(tokens[tokens.index(id_flag) + 1]) != str(worker_id):
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

        _logger.debug(f"_find_pids_for_worker({worker_id}, mode={worker_mode}): "
                      f"scanned {scanned} python procs, found java={java_pids}, py={py_pids}")
        return java_pids, py_pids

    def _capture_sts_pids_for_worker(self, launcher_pid: int, worker_id: int,
                                     before_java_pids: set[int] | None = None):
        """Best-effort snapshot of worker N's STS/MTS java PIDs once launched.

        A worker can fail before rollout_worker.py starts, leaving only a
        ModTheSpire launcher/menu. Capture newly-created Java PIDs too, not only
        Java parents of Python worker processes, so restart/stop can clean them.
        """
        before_java_pids = set(before_java_pids or set())
        _logger.debug(f"_capture_sts_pids: worker {worker_id}, launcher PID {launcher_pid}, "
                      f"before_java={sorted(before_java_pids)}")
        deadline = time.time() + 180.0
        polls = 0
        captured_new = False
        while time.time() < deadline and self.running:
            java_pids, _ = self._find_pids_for_worker(worker_id)
            current_java = self._sts_java_pids()
            new_java = current_java - before_java_pids
            polls += 1
            if new_java:
                self.worker_sts_pids.setdefault(worker_id, set()).update(new_java)
                if not captured_new:
                    _logger.info(f"Captured newly spawned STS/MTS PIDs for worker {worker_id}: "
                                 f"{sorted(new_java)} (after {polls} polls)")
                    captured_new = True
            if java_pids:
                self.worker_sts_pids.setdefault(worker_id, set()).update(java_pids)
                _logger.info(f"Captured worker-parent STS PIDs for worker {worker_id}: "
                             f"{sorted(java_pids)} (after {polls} polls)")
                return
            if captured_new and polls >= 5:
                # We have at least the Java/menu PID. If Python never starts,
                # health monitoring can kill this cached PID and relaunch cleanly.
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

    _OUR_SCRIPTS = (
        "rollout_worker.py", "train_ppo.py", "train_bc_ppo.py",
        "behavior_clone.py", "eval_model.py", "game_logger.py",
        "train_offline.py",
    )

    def _kill_orphan_python_scripts(self) -> int:
        """Kill any Python processes still running our scripts."""
        _logger.info("Sweeping for orphan Python script processes...")
        try:
            import psutil
        except ImportError:
            _logger.warning("psutil not available for Python orphan sweep")
            return 0

        killed = 0
        scanned = 0
        our_pid = os.getpid()

        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if p.info["pid"] == our_pid:
                    continue
                name = (p.info.get("name") or "").lower()
                if not name.startswith("python"):
                    continue
                scanned += 1
                cmdline = " ".join(p.info.get("cmdline") or [])
                matched_script = None
                for script in self._OUR_SCRIPTS:
                    if script in cmdline:
                        matched_script = script
                        break
                if matched_script:
                    _logger.info(f"Killing orphan python PID {p.info['pid']}: "
                                 f"script={matched_script}, cmdline={cmdline[:200]}")
                    p.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                _logger.debug(f"Python orphan sweep: PID access issue: {e}")
            except Exception as e:
                _logger.warning(f"Python orphan sweep unexpected error: {e}")

        _logger.info(f"Python orphan sweep done: scanned {scanned} python procs, killed {killed}")
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
        self._trainer_cmd = None
        self._with_trainer = False

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
        self.worker_launcher_pids.clear()
        self.worker_sts_pids.clear()
        self._worker_commands.clear()
        self._worker_modes.clear()
        self._worker_demo_paths.clear()
        self._worker_checkpoint_paths.clear()
        self._worker_launch_time.clear()
        self._worker_game_targets.clear()
        self._worker_games_at_launch.clear()

        java_orphans = self._kill_orphan_sts_instances()
        py_orphans = self._kill_orphan_python_scripts()
        closed += java_orphans + py_orphans
        if closed:
            self._append_log("All", f"Closed {closed} process(es) "
                             f"({java_orphans} java, {py_orphans} python scripts).")
        _logger.info(f"Stop complete: {closed} process(es) closed "
                     f"(tracked launchers + {java_orphans} java orphans "
                     f"+ {py_orphans} python orphans)")

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
        mode = MODES.get(self.mode_var.get(), "worker")
        _logger.debug(f"VENV_PYTHON: {VENV_PYTHON}  exists={VENV_PYTHON.exists()}")
        if mode != "play" and not VENV_PYTHON.exists():
            issues.append(f"Python venv not found:\n  {VENV_PYTHON}\n  Run: python -m venv .venv && pip install -r requirements.txt")
        _logger.debug(f"JAVA_EXE: {JAVA_EXE}  exists={JAVA_EXE.exists()}")
        if mode not in ("bc_train", "consume") and not JAVA_EXE.exists():
            issues.append(f"STS Java runtime not found:\n  {JAVA_EXE}\n  Is Slay the Spire installed via Steam?")
        _logger.debug(f"MTS_LAUNCHER: {MTS_LAUNCHER}  exists={MTS_LAUNCHER.exists()}")
        if mode not in ("bc_train", "consume") and not MTS_LAUNCHER.exists():
            issues.append(
                f"ModTheSpire jar not found:\n  {MTS_LAUNCHER}\n"
                "Install Mod the Spire via Steam Workshop, or set "
                "ASCENSIONAI_MTS_JAR to the full ModTheSpire.jar path."
            )
        if mode == "eval_set":
            seed_file = self._selected_seed_file(self._normalize_spinner_value())
            if not seed_file.lower().endswith(".txt"):
                issues.append("Seed file must be a .txt file, for example seeds/eval_200.txt")
            eval_policy = self._selected_eval_policy()
            required_models: list[str] = []
            if eval_policy in {"full", "bc"}:
                required_models.append("models/ppo_sts_bc.pt")
            if eval_policy in {"full", "ppo_current"}:
                required_models.append("models/ppo_sts.pt")
            for rel in required_models:
                if not (ROOT / rel).exists():
                    issues.append(f"Required eval checkpoint not found:\n  {ROOT / rel}")
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
