# AscensionAI VM Sync — Windows PowerShell version
#
# Usage (from project root):
#   .\vm\sync.ps1 push           # Push model + code to VM
#   .\vm\sync.ps1 pull           # Pull model + logs from VM
#   .\vm\sync.ps1 push-code      # Push only code after changes
#   .\vm\sync.ps1 pull-model     # Pull only the .pt checkpoint
#   .\vm\sync.ps1 status         # Check VM training status
#
# Prerequisites: gcloud CLI installed and configured
#   gcloud compute ssh INSTANCE_NAME -- command
#   gcloud compute scp local:path INSTANCE_NAME:~/path

param(
    [Parameter(Position=0)]
    [ValidateSet("push", "pull", "push-code", "pull-model", "push-model", "status", "help")]
    [string]$Action = "help"
)

# ─── Configuration ──────────────────────────────────────────────────────────
# Option 1: Use gcloud (recommended for GCP)
$VM_INSTANCE = "ascension-vm"          # Your GCP VM instance name
$VM_ZONE     = "us-west1-a"            # Your GCP zone
$VM_PROJECT_PATH = "~/ascension"

# Option 2: Use direct SSH (uncomment and set if preferred)
# $VM_SSH = "username@34.xx.xx.xx"

$LOCAL_PROJECT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# ─── Helper ─────────────────────────────────────────────────────────────────
function GCP-SCP {
    param([string]$Local, [string]$Remote, [switch]$Upload)
    if ($Upload) {
        gcloud compute scp --recurse $Local "${VM_INSTANCE}:${Remote}" --zone $VM_ZONE
    } else {
        gcloud compute scp --recurse "${VM_INSTANCE}:${Remote}" $Local --zone $VM_ZONE
    }
}

function GCP-SSH {
    param([string]$Command)
    gcloud compute ssh $VM_INSTANCE --zone $VM_ZONE --command $Command
}

# ─── Commands ───────────────────────────────────────────────────────────────
switch ($Action) {
    "push" {
        Write-Host "=== Pushing model + code + seeds to VM ===" -ForegroundColor Green
        GCP-SCP -Local "$LOCAL_PROJECT\scripts" -Remote "$VM_PROJECT_PATH/scripts/" -Upload
        GCP-SCP -Local "$LOCAL_PROJECT\external" -Remote "$VM_PROJECT_PATH/external/" -Upload
        GCP-SCP -Local "$LOCAL_PROJECT\models\ppo_sts.pt" -Remote "$VM_PROJECT_PATH/models/" -Upload
        GCP-SCP -Local "$LOCAL_PROJECT\seeds" -Remote "$VM_PROJECT_PATH/seeds/" -Upload
        GCP-SCP -Local "$LOCAL_PROJECT\vm" -Remote "$VM_PROJECT_PATH/vm/" -Upload
        Write-Host "Done. SSH in and run: cd ~/ascension && ./vm/run_training.sh" -ForegroundColor Green
    }
    "pull" {
        Write-Host "=== Pulling model + logs from VM ===" -ForegroundColor Green
        GCP-SCP -Local "$LOCAL_PROJECT\models\" -Remote "$VM_PROJECT_PATH/models/ppo_sts.pt"
        try { GCP-SCP -Local "$LOCAL_PROJECT\logs\" -Remote "$VM_PROJECT_PATH/logs/training_stats.csv" } catch {}
        try { GCP-SCP -Local "$LOCAL_PROJECT\logs\" -Remote "$VM_PROJECT_PATH/logs/eval_stats.csv" } catch {}
        Write-Host "Done. Model and logs synced to local." -ForegroundColor Green
    }
    "push-code" {
        Write-Host "=== Pushing code only ===" -ForegroundColor Green
        GCP-SCP -Local "$LOCAL_PROJECT\scripts" -Remote "$VM_PROJECT_PATH/scripts/" -Upload
        GCP-SCP -Local "$LOCAL_PROJECT\external" -Remote "$VM_PROJECT_PATH/external/" -Upload
        GCP-SCP -Local "$LOCAL_PROJECT\vm" -Remote "$VM_PROJECT_PATH/vm/" -Upload
        Write-Host "Done. Restart workers on VM to pick up changes." -ForegroundColor Green
    }
    "pull-model" {
        Write-Host "=== Pulling model checkpoint only ===" -ForegroundColor Green
        GCP-SCP -Local "$LOCAL_PROJECT\models\" -Remote "$VM_PROJECT_PATH/models/ppo_sts.pt"
        Write-Host "Done. Checkpoint at: models\ppo_sts.pt" -ForegroundColor Green
    }
    "push-model" {
        Write-Host "=== Pushing model to VM ===" -ForegroundColor Green
        GCP-SCP -Local "$LOCAL_PROJECT\models\ppo_sts.pt" -Remote "$VM_PROJECT_PATH/models/" -Upload
        Write-Host "Done." -ForegroundColor Green
    }
    "status" {
        Write-Host "=== VM Training Status ===" -ForegroundColor Cyan
        GCP-SSH -Command "pgrep -c -f DesktopLauncher 2>/dev/null || echo 0; echo '---'; pgrep -c -f train_offline 2>/dev/null || echo 0; echo '---'; wc -l < ~/ascension/logs/training_stats.csv 2>/dev/null || echo 0; echo '---'; ls ~/ascension/rollouts_shared/*.npz 2>/dev/null | wc -l"
    }
    default {
        Write-Host @"
Usage: .\vm\sync.ps1 <action>

Actions:
  push        Push model + code + seeds to VM
  pull        Pull model + logs from VM
  push-code   Push only code (after making changes)
  pull-model  Pull only the .pt checkpoint
  push-model  Push only the .pt checkpoint to VM
  status      Check VM training status remotely

Configuration:
  Edit the top of this file to set your VM instance name and zone.
  Requires: gcloud CLI (install from https://cloud.google.com/sdk)

Workflow:
  1. Make code changes locally
  2. .\vm\sync.ps1 push-code     # send changes to VM
  3. SSH to VM, stop/restart workers
  4. Let it run for N hours
  5. .\vm\sync.ps1 pull           # get model + stats back
"@
    }
}
