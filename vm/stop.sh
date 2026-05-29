#!/bin/bash
# Stop all AscensionAI training/eval processes gracefully
# Usage: ./stop.sh

PROJECT_DIR="$HOME/ascension"
STOP_FILE="$PROJECT_DIR/logs/.stop_training"
PIDS_FILE="$PROJECT_DIR/logs/.training_pids"

echo "=== Stopping AscensionAI ==="

# Signal the training loop to stop after current games finish
touch "$STOP_FILE"
echo "Stop signal sent. Workers will finish current games and exit."

# Kill any Java/STS instances
STS_PIDS=$(pgrep -f "DesktopLauncher" 2>/dev/null || true)
if [ -n "$STS_PIDS" ]; then
    echo "Killing $(echo "$STS_PIDS" | wc -w) STS instances..."
    echo "$STS_PIDS" | xargs kill 2>/dev/null || true
fi

# Kill any Xvfb instances
XVFB_PIDS=$(pgrep -f "Xvfb" 2>/dev/null || true)
if [ -n "$XVFB_PIDS" ]; then
    echo "Killing Xvfb processes..."
    echo "$XVFB_PIDS" | xargs kill 2>/dev/null || true
fi

# Kill trainer
TRAINER_PIDS=$(pgrep -f "train_offline.py" 2>/dev/null || true)
if [ -n "$TRAINER_PIDS" ]; then
    echo "Stopping trainer..."
    echo "$TRAINER_PIDS" | xargs kill 2>/dev/null || true
fi

# Kill tracked PIDs from run_training.sh
if [ -f "$PIDS_FILE" ]; then
    while read -r pid; do
        kill "$pid" 2>/dev/null || true
    done < "$PIDS_FILE"
    rm -f "$PIDS_FILE"
fi

sleep 2

# Verify
REMAINING=$(pgrep -f "DesktopLauncher|train_offline|rollout_worker|eval_model" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "Force-killing remaining processes..."
    echo "$REMAINING" | xargs kill -9 2>/dev/null || true
fi

echo "All processes stopped."
echo ""
echo "Rollouts in:  $PROJECT_DIR/rollouts_shared/"
echo "Model at:     $PROJECT_DIR/models/ppo_sts.pt"
echo "Logs at:      $PROJECT_DIR/logs/"
echo ""
echo "To pull results: vm/sync.sh pull  (from local machine)"
