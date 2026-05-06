<#
.SYNOPSIS
    Launches multiple Slay the Spire instances for parallel RL training.

.DESCRIPTION
    Each instance runs rollout_worker.py with a unique worker ID. The script
    swaps the CommunicationMod config.properties before each launch, waits for
    the process to start, then swaps in the next config.

    Run train_offline.py separately to consume the rollout data:
        python scripts\train_offline.py --model models\ppo_sts.pt --data rollouts_shared --delete-consumed

.PARAMETER NumWorkers
    Number of STS instances to launch (default: 3).

.PARAMETER Mode
    "worker" to run rollout_worker.py (default), "train" to run train_ppo.py,
    "bc" to run behavior_clone.py on a single instance.

.EXAMPLE
    .\launch_workers.ps1
    .\launch_workers.ps1 -NumWorkers 2
    .\launch_workers.ps1 -Mode bc
#>

param(
    [int]$NumWorkers = 3,
    [ValidateSet("worker", "train", "bc-ppo", "bc", "eval", "logger")]
    [string]$Mode = "worker",
    [int]$Games = 20,
    [int]$BCGames = 50
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$ProjectRoot   = $PSScriptRoot
$PythonExe     = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StsDir        = "C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire"
$MtsLauncher   = Join-Path $StsDir "mts-launcher.jar"
$JavaExe       = Join-Path $StsDir "jre\bin\java.exe"
$ConfigDir     = Join-Path $env:LOCALAPPDATA "ModTheSpire\CommunicationMod"
$ConfigFile    = Join-Path $ConfigDir "config.properties"

# --- Validation ---
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python venv not found at $PythonExe. Run: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $MtsLauncher)) {
    Write-Error "ModTheSpire not found at $MtsLauncher. Is STS installed via Steam?"
    exit 1
}
if (-not (Test-Path $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
}

# Ensure rollouts_shared exists for workers
$RolloutsDir = Join-Path $ProjectRoot "rollouts_shared"
if ($Mode -eq "worker" -and -not (Test-Path $RolloutsDir)) {
    New-Item -ItemType Directory -Path $RolloutsDir -Force | Out-Null
}

# Ensure models dir exists
$ModelsDir = Join-Path $ProjectRoot "models"
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
}

# Escape paths for Java properties file format (backslash → forward slash, colon escaped)
function Escape-PropertiesPath($path) {
    return $path.Replace('\', '/').Replace(':', '\:')
}

$PyEsc   = Escape-PropertiesPath $PythonExe
$RootEsc = Escape-PropertiesPath $ProjectRoot

function Write-Config($command) {
    $timestamp = Get-Date -Format "ddd MMM dd HH:mm:ss zzz yyyy"
    $content = @"
#$timestamp
verbose=true
command=$command
runAtGameStart=true
"@
    Set-Content -Path $ConfigFile -Value $content -Encoding ASCII
}

function Launch-STS {
    $args = @("-jar", "`"$MtsLauncher`"", "--skip-launcher")
    $proc = Start-Process -FilePath $JavaExe `
        -ArgumentList $args `
        -WorkingDirectory $StsDir `
        -PassThru
    return $proc
}

# --- Build command per mode ---
function Get-Command-For-Worker($workerId) {
    return "$PyEsc $RootEsc/scripts/rollout_worker.py --model models/ppo_sts.pt --out rollouts_shared --id $workerId"
}

function Get-Command-For-Train {
    return "$PyEsc $RootEsc/scripts/train_ppo.py --save models/ppo_sts.pt --save-every 5"
}

function Get-Command-For-BC {
    return "$PyEsc $RootEsc/scripts/behavior_clone.py --games 50 --save models/ppo_sts.pt"
}

function Get-Command-For-BCPPO {
    return "$PyEsc $RootEsc/scripts/train_bc_ppo.py --bc-games $BCGames --ppo-games 200 --save models/ppo_sts.pt"
}

function Get-Command-For-Eval {
    return "$PyEsc $RootEsc/scripts/eval_model.py --model models/ppo_sts.pt --games $Games"
}

function Get-Command-For-Logger {
    return "$PyEsc $RootEsc/scripts/game_logger.py"
}

# --- Launch ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AscensionAI Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$launched = @()

switch ($Mode) {
    "worker" {
        Write-Host "Mode: Parallel rollout workers ($NumWorkers instances)" -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: Start the offline trainer in a separate terminal:" -ForegroundColor Yellow
        Write-Host "  python scripts\train_offline.py --model models\ppo_sts.pt --data rollouts_shared --delete-consumed" -ForegroundColor Yellow
        Write-Host ""

        for ($i = 1; $i -le $NumWorkers; $i++) {
            $cmd = Get-Command-For-Worker $i
            Write-Host "[$i/$NumWorkers] Setting config for worker $i..." -ForegroundColor Cyan
            Write-Config $cmd

            Write-Host "[$i/$NumWorkers] Launching STS instance (worker $i)..." -ForegroundColor Cyan
            $proc = Launch-STS
            $launched += $proc
            Write-Host "[$i/$NumWorkers] PID: $($proc.Id)" -ForegroundColor DarkGray

            if ($i -lt $NumWorkers) {
                Write-Host "  Waiting 15 seconds for ModTheSpire to read config..." -ForegroundColor DarkGray
                Start-Sleep -Seconds 15
            }
        }
    }
    "train" {
        Write-Host "Mode: Single-instance PPO training" -ForegroundColor Green
        $NumWorkers = 1
        $cmd = Get-Command-For-Train
        Write-Config $cmd
        Write-Host "Launching STS..." -ForegroundColor Cyan
        $proc = Launch-STS
        $launched += $proc
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
    }
    "bc" {
        Write-Host "Mode: Behavior cloning (50 heuristic games)" -ForegroundColor Green
        $NumWorkers = 1
        $cmd = Get-Command-For-BC
        Write-Config $cmd
        Write-Host "Launching STS..." -ForegroundColor Cyan
        $proc = Launch-STS
        $launched += $proc
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
    }
    "bc-ppo" {
        Write-Host "Mode: BC -> PPO end-to-end ($BCGames BC games + 200 PPO games)" -ForegroundColor Green
        $NumWorkers = 1
        $cmd = Get-Command-For-BCPPO
        Write-Config $cmd
        Write-Host "Launching STS..." -ForegroundColor Cyan
        $proc = Launch-STS
        $launched += $proc
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
    }
    "eval" {
        Write-Host "Mode: Evaluation ($Games games, greedy)" -ForegroundColor Green
        $NumWorkers = 1
        $cmd = Get-Command-For-Eval
        Write-Config $cmd
        Write-Host "Launching STS..." -ForegroundColor Cyan
        $proc = Launch-STS
        $launched += $proc
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
    }
    "logger" {
        Write-Host "Mode: Passive game logger" -ForegroundColor Green
        $NumWorkers = 1
        $cmd = Get-Command-For-Logger
        Write-Config $cmd
        Write-Host "Launching STS..." -ForegroundColor Cyan
        $proc = Launch-STS
        $launched += $proc
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  $($launched.Count) instance(s) launched" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Log files:" -ForegroundColor DarkGray

switch ($Mode) {
    "worker" {
        for ($i = 1; $i -le $NumWorkers; $i++) {
            Write-Host "  Worker $i : worker_${i}_debug.log" -ForegroundColor DarkGray
        }
        Write-Host "  Trainer  : train_offline_debug.log" -ForegroundColor DarkGray
    }
    "train" {
        Write-Host "  Training : train_debug.log" -ForegroundColor DarkGray
    }
    "bc-ppo" {
        Write-Host "  BC->PPO  : train_bc_ppo_debug.log" -ForegroundColor DarkGray
    }
    "bc" {
        Write-Host "  BC       : bc_debug.log" -ForegroundColor DarkGray
    }
    "eval" {
        Write-Host "  Eval     : eval_debug.log" -ForegroundColor DarkGray
    }
    "logger" {
        Write-Host "  Logger   : game_logger_debug.log" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "To stop: close the STS game windows (or press Ctrl+C in trainer terminal)." -ForegroundColor DarkGray
