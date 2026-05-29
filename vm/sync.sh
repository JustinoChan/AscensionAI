#!/bin/bash
# Sync model and data between local machine and GCP VM
#
# Usage (run from your LOCAL machine):
#   ./vm/sync.sh push              # Push model + code to VM
#   ./vm/sync.sh pull              # Pull model + logs from VM
#   ./vm/sync.sh push-code         # Push only code (after changes)
#   ./vm/sync.sh pull-model        # Pull only the .pt checkpoint
#   ./vm/sync.sh status            # Check VM training status
#
# Configure VM_HOST below or set environment variable:
#   export VM_HOST="username@34.xx.xx.xx"

set -e

# ─── Configuration ──────────────────────────────────────────────────────────
# Set your VM's SSH address here (or export VM_HOST env var)
VM_HOST="${VM_HOST:-your-username@YOUR_VM_IP}"
VM_PROJECT="~/ascension"
LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Validation ─────────────────────────────────────────────────────────────
if [[ "$VM_HOST" == *"YOUR_VM_IP"* ]]; then
    echo "ERROR: Set VM_HOST first."
    echo "  export VM_HOST=\"username@34.xx.xx.xx\""
    echo "  or edit vm/sync.sh and change the VM_HOST line."
    exit 1
fi

# ─── Commands ───────────────────────────────────────────────────────────────
case "${1:-help}" in

push)
    echo "=== Pushing model + code + seeds to VM ==="
    # Code
    rsync -avz --progress \
        "$LOCAL_PROJECT/scripts/" \
        "$VM_HOST:$VM_PROJECT/scripts/"
    rsync -avz --progress \
        "$LOCAL_PROJECT/external/" \
        "$VM_HOST:$VM_PROJECT/external/"
    # Model
    rsync -avz --progress \
        "$LOCAL_PROJECT/models/ppo_sts.pt" \
        "$VM_HOST:$VM_PROJECT/models/"
    # Seeds
    rsync -avz --progress \
        "$LOCAL_PROJECT/seeds/" \
        "$VM_HOST:$VM_PROJECT/seeds/"
    # VM scripts
    rsync -avz --progress \
        "$LOCAL_PROJECT/vm/" \
        "$VM_HOST:$VM_PROJECT/vm/"
    echo "Done. SSH in and run: cd ~/ascension && ./vm/run_training.sh"
    ;;

pull)
    echo "=== Pulling model + logs from VM ==="
    # Model checkpoint
    rsync -avz --progress \
        "$VM_HOST:$VM_PROJECT/models/ppo_sts.pt" \
        "$LOCAL_PROJECT/models/"
    # Training stats
    rsync -avz --progress \
        "$VM_HOST:$VM_PROJECT/logs/training_stats.csv" \
        "$LOCAL_PROJECT/logs/" 2>/dev/null || echo "(no training_stats.csv)"
    # Eval stats
    rsync -avz --progress \
        "$VM_HOST:$VM_PROJECT/logs/eval_stats.csv" \
        "$LOCAL_PROJECT/logs/" 2>/dev/null || echo "(no eval_stats.csv)"
    # Worker debug logs
    rsync -avz --progress \
        "$VM_HOST:$VM_PROJECT/logs/worker_*_debug.log" \
        "$LOCAL_PROJECT/logs/" 2>/dev/null || echo "(no worker logs)"
    echo "Done. Model and logs synced to local."
    ;;

push-code)
    echo "=== Pushing code only ==="
    rsync -avz --progress \
        "$LOCAL_PROJECT/scripts/" \
        "$VM_HOST:$VM_PROJECT/scripts/"
    rsync -avz --progress \
        "$LOCAL_PROJECT/external/" \
        "$VM_HOST:$VM_PROJECT/external/"
    rsync -avz --progress \
        "$LOCAL_PROJECT/vm/" \
        "$VM_HOST:$VM_PROJECT/vm/"
    echo "Done. Restart workers to pick up changes."
    ;;

pull-model)
    echo "=== Pulling model checkpoint only ==="
    rsync -avz --progress \
        "$VM_HOST:$VM_PROJECT/models/ppo_sts.pt" \
        "$LOCAL_PROJECT/models/"
    echo "Done. Checkpoint at: models/ppo_sts.pt"
    ;;

push-model)
    echo "=== Pushing model checkpoint to VM ==="
    rsync -avz --progress \
        "$LOCAL_PROJECT/models/ppo_sts.pt" \
        "$VM_HOST:$VM_PROJECT/models/"
    echo "Done. Restart workers to use new checkpoint."
    ;;

status)
    echo "=== VM Training Status ==="
    ssh "$VM_HOST" bash -c "'
        echo \"--- Processes ---\"
        pgrep -c -f DesktopLauncher 2>/dev/null && echo \"STS instances: \$(pgrep -c -f DesktopLauncher)\" || echo \"STS instances: 0\"
        pgrep -c -f train_offline 2>/dev/null && echo \"Trainer: running\" || echo \"Trainer: stopped\"
        echo \"\"
        echo \"--- Latest training stats ---\"
        if [ -f ~/ascension/logs/training_stats.csv ]; then
            GAMES=\$(grep -c \",\" ~/ascension/logs/training_stats.csv 2>/dev/null || echo 0)
            echo \"Total rows: \$GAMES\"
            tail -1 ~/ascension/logs/training_stats.csv | cut -d, -f1-5
        fi
        echo \"\"
        echo \"--- Rollout queue ---\"
        ls ~/ascension/rollouts_shared/*.npz 2>/dev/null | wc -l | xargs -I{} echo \"Pending rollouts: {}\"
        echo \"\"
        echo \"--- Disk usage ---\"
        du -sh ~/ascension/models/ ~/ascension/rollouts_shared/ ~/ascension/logs/ 2>/dev/null
    '"
    ;;

*)
    echo "Usage: $0 {push|pull|push-code|pull-model|push-model|status}"
    echo ""
    echo "  push        Push model + code + seeds to VM"
    echo "  pull        Pull model + logs from VM"
    echo "  push-code   Push only code (after making changes)"
    echo "  pull-model  Pull only the .pt checkpoint"
    echo "  push-model  Push only the .pt checkpoint"
    echo "  status      Check VM training status remotely"
    ;;
esac
