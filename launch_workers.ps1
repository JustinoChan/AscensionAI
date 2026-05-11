<#
.SYNOPSIS
    Launches multiple Slay the Spire instances for parallel RL training.

.DESCRIPTION
    Each instance runs rollout_worker.py with a unique worker ID. The script
    swaps the CommunicationMod config.properties before each launch, waits for
    the process to start, then swaps in the next config.

    Run train_offline.py separately to consume the rollout data:
        python scripts\train_offline.py --model models\ppo_sts.pt --data rollouts_shared --delete-consumed --batch-games 8 --lr 3e-5 --bc-coef 0.10 --max-rollout-lag 4 --ent-coef 0.001 --auto-tune

.PARAMETER NumWorkers
    Number of STS instances to launch (default: 3).

.PARAMETER Mode
    "worker" to run rollout_worker.py (default), "train" to run train_ppo.py,
    "bc-ppo" for end-to-end warm start, "bc" for behavior cloning,
    "bc-collect" for parallel BC demo collection, "bc-train" to train
    from collected demo files, "eval" for greedy evaluation, or "logger"
    for passive logging.

.EXAMPLE
    .\launch_workers.ps1
    .\launch_workers.ps1 -NumWorkers 2
    .\launch_workers.ps1 -Mode bc
    .\launch_workers.ps1 -Mode bc-collect -NumWorkers 4 -BCGames 400
    .\launch_workers.ps1 -Mode bc-train -BCEpochs 50
#>

param(
    [int]$NumWorkers = 3,
    [ValidateSet("worker", "train", "bc-ppo", "bc", "bc-collect", "bc-train", "eval", "logger")]
    [string]$Mode = "worker",
    [int]$Games = 20,
    [int]$BCGames = 400,
    [int]$BCEpochs = 50,
    [double]$BCLr = 5e-4,
    [int]$BCBatchSize = 256,
    [double]$BCValSplit = 0.10,
    [int]$BCPatience = 12,
    [double]$BCWeightDecay = 1e-5,
    [double]$BCLabelSmoothing = 0.02,
    [string]$SeedFile = "",
    [int]$TopActions = 0,
    [switch]$HeuristicEval
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$ProjectRoot   = $PSScriptRoot
$PythonExe     = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StsDir        = "C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire"
$JavaExe       = Join-Path $StsDir "jre\bin\java.exe"

# Use the real Steam Workshop ModTheSpire.jar instead of the common
# SlayTheSpire\mts-launcher.jar wrapper. The wrapper may still open the
# launcher menu even when --skip-launcher / --profile are passed.
$DefaultMtsJar = "C:\Program Files (x86)\Steam\steamapps\workshop\content\646570\1605060445\ModTheSpire.jar"
if ($env:ASCENSIONAI_MTS_JAR -and $env:ASCENSIONAI_MTS_JAR.Trim() -ne "") {
    $MtsLauncher = $env:ASCENSIONAI_MTS_JAR.Trim()
} elseif (Test-Path $DefaultMtsJar) {
    $MtsLauncher = $DefaultMtsJar
} else {
    $MtsLauncher = Join-Path $StsDir "mts-launcher.jar"
}

$MtsProfile = if ($env:ASCENSIONAI_MTS_PROFILE -and $env:ASCENSIONAI_MTS_PROFILE.Trim() -ne "") {
    $env:ASCENSIONAI_MTS_PROFILE.Trim()
} else {
    "AscensionAI"
}
$MtsMods = if ($env:ASCENSIONAI_MTS_MODS -and $env:ASCENSIONAI_MTS_MODS.Trim() -ne "") {
    $env:ASCENSIONAI_MTS_MODS.Trim()
} else {
    ""
}

$ConfigDir     = Join-Path $env:LOCALAPPDATA "ModTheSpire\CommunicationMod"
$ConfigFile    = Join-Path $ConfigDir "config.properties"

# --- Validation ---
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python venv not found at $PythonExe. Run: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $MtsLauncher)) {
    Write-Error "ModTheSpire jar not found at $MtsLauncher. Install Mod the Spire via Steam Workshop or set ASCENSIONAI_MTS_JAR."
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

# Shared directory for parallel BC demo collection.
$BCDemosDir = Join-Path $ProjectRoot "bc_demos_shared"
if (($Mode -eq "bc-collect" -or $Mode -eq "bc-train") -and -not (Test-Path $BCDemosDir)) {
    New-Item -ItemType Directory -Path $BCDemosDir -Force | Out-Null
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
    $args = @("-jar", "`"$MtsLauncher`"")

    if ($MtsMods -ne "") {
        # --mods implies skip-launcher on supported ModTheSpire versions.
        $args += @("--mods", $MtsMods)
    } elseif ($MtsProfile -ne "") {
        $args += @("--skip-launcher", "--profile", $MtsProfile)
    } else {
        $args += @("--skip-launcher")
    }

    Write-Host "Launching with ModTheSpire jar: $MtsLauncher" -ForegroundColor DarkGray
    if ($MtsMods -ne "") {
        Write-Host "Using ModTheSpire --mods: $MtsMods" -ForegroundColor DarkGray
    } elseif ($MtsProfile -ne "") {
        Write-Host "Using ModTheSpire profile: $MtsProfile" -ForegroundColor DarkGray
    }

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
    return "$PyEsc $RootEsc/scripts/behavior_clone.py --games $BCGames --save models/ppo_sts_bc.pt --epochs $BCEpochs --lr $BCLr --batch-size $BCBatchSize --val-split $BCValSplit --patience $BCPatience --weight-decay $BCWeightDecay --label-smoothing $BCLabelSmoothing"
}

function Get-Command-For-BCCollect($workerId, $gamesForWorker) {
    return "$PyEsc $RootEsc/scripts/behavior_clone.py --games $gamesForWorker --save models/ppo_sts_bc_worker_$workerId.pt --collect-only --worker-id $workerId --demo-dir bc_demos_shared --bc-checkpoint models/ppo_sts_bc_progress_worker_$workerId.npz --lr $BCLr --batch-size $BCBatchSize --val-split $BCValSplit --patience $BCPatience --weight-decay $BCWeightDecay --label-smoothing $BCLabelSmoothing"
}

function Get-Command-For-BCTrain {
    return "$PyEsc $RootEsc/scripts/behavior_clone.py --train-demo-dir bc_demos_shared --save models/ppo_sts_bc.pt --epochs $BCEpochs --lr $BCLr --batch-size $BCBatchSize --val-split $BCValSplit --patience $BCPatience --weight-decay $BCWeightDecay --label-smoothing $BCLabelSmoothing"
}

function Get-Command-For-BCPPO {
    return "$PyEsc $RootEsc/scripts/train_bc_ppo.py --bc-games $BCGames --bc-epochs $BCEpochs --bc-lr 5e-4 --ppo-games 200 --save models/ppo_sts.pt"
}

function Get-Command-For-Eval {
    $cmd = "$PyEsc $RootEsc/scripts/eval_model.py --model models/ppo_sts.pt --games $Games"
    if ($SeedFile -ne "") {
        $cmd += " --seed-file $SeedFile"
    }
    if ($TopActions -gt 0) {
        $cmd += " --top-actions $TopActions"
    }
    if ($HeuristicEval) {
        $cmd += " --policy heuristic"
    }
    return $cmd
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
        Write-Host "  python scripts\train_offline.py --model models\ppo_sts.pt --data rollouts_shared --delete-consumed --batch-games 8 --lr 3e-5 --bc-coef 0.10 --max-rollout-lag 4 --ent-coef 0.001 --auto-tune" -ForegroundColor Yellow
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
        Write-Host "Mode: Behavior cloning ($BCGames heuristic games, $BCEpochs BC epochs)" -ForegroundColor Green
        $NumWorkers = 1
        $cmd = Get-Command-For-BC
        Write-Config $cmd
        Write-Host "Launching STS..." -ForegroundColor Cyan
        $proc = Launch-STS
        $launched += $proc
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
    }
    "bc-collect" {
        $gamesPerWorker = [math]::Ceiling($BCGames / [double]$NumWorkers)
        $estimatedTotal = $gamesPerWorker * $NumWorkers
        Write-Host "Mode: Parallel BC demo collection" -ForegroundColor Green
        Write-Host "Workers: $NumWorkers | Requested total games: $BCGames | Per worker: $gamesPerWorker | Actual max total: $estimatedTotal" -ForegroundColor Yellow
        Write-Host "Demo files will be written to: $BCDemosDir" -ForegroundColor Yellow
        Write-Host "After all workers finish, run:" -ForegroundColor Yellow
        Write-Host "  .\launch_workers.ps1 -Mode bc-train -BCEpochs $BCEpochs" -ForegroundColor Yellow
        Write-Host ""

        for ($i = 1; $i -le $NumWorkers; $i++) {
            $cmd = Get-Command-For-BCCollect $i $gamesPerWorker
            Write-Host "[$i/$NumWorkers] Setting config for BC collector $i..." -ForegroundColor Cyan
            Write-Config $cmd

            Write-Host "[$i/$NumWorkers] Launching STS instance (BC collector $i)..." -ForegroundColor Cyan
            $proc = Launch-STS
            $launched += $proc
            Write-Host "[$i/$NumWorkers] PID: $($proc.Id)" -ForegroundColor DarkGray

            if ($i -lt $NumWorkers) {
                Write-Host "  Waiting 15 seconds for ModTheSpire to read config..." -ForegroundColor DarkGray
                Start-Sleep -Seconds 15
            }
        }
    }
    "bc-train" {
        Write-Host "Mode: Train BC model from saved demo files" -ForegroundColor Green
        Write-Host "Demo dir: $BCDemosDir" -ForegroundColor Yellow
        Write-Host "Epochs: $BCEpochs | LR: $BCLr | Batch size: $BCBatchSize" -ForegroundColor Yellow
        $cmd = Get-Command-For-BCTrain
        Write-Host "Running: $cmd" -ForegroundColor DarkGray
        & $PythonExe (Join-Path $ProjectRoot "scripts\behavior_clone.py") --train-demo-dir "bc_demos_shared" --save "models/ppo_sts_bc.pt" --epochs $BCEpochs --lr $BCLr --batch-size $BCBatchSize --val-split $BCValSplit --patience $BCPatience --weight-decay $BCWeightDecay --label-smoothing $BCLabelSmoothing
    }
    "bc-ppo" {
        Write-Host "Mode: BC -> PPO end-to-end ($BCGames BC games, $BCEpochs BC epochs + 200 PPO games)" -ForegroundColor Green
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
    "bc-collect" {
        for ($i = 1; $i -le $NumWorkers; $i++) {
            Write-Host "  BC $i    : bc_debug.log" -ForegroundColor DarkGray
        }
        Write-Host "  Demos    : bc_demos_shared\*.npz" -ForegroundColor DarkGray
    }
    "bc-train" {
        Write-Host "  BC train : bc_debug.log" -ForegroundColor DarkGray
        Write-Host "  Stats    : bc_train_stats.csv" -ForegroundColor DarkGray
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
