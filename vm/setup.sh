#!/bin/bash
# AscensionAI VM Setup — run once on a fresh Ubuntu 22.04+ VM
# Usage: bash setup.sh
set -e

echo "=== AscensionAI VM Setup ==="

# System packages
sudo apt-get update
sudo apt-get install -y \
    openjdk-17-jre-headless \
    xvfb \
    python3 python3-pip python3-venv \
    git unzip rsync tmux htop

# Project directory
PROJECT_DIR="$HOME/ascension"
mkdir -p "$PROJECT_DIR"/{models,rollouts_shared,logs,seeds,Eval}

echo ""
echo "=== Python environment ==="
python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy gymnasium

echo ""
echo "=== Directory structure ==="
echo "Project root: $PROJECT_DIR"
echo ""
echo "You still need to:"
echo "  1. Copy your STS game files:   scp -r 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\SlayTheSpire\\*' user@VM:~/ascension/game/"
echo "  2. Copy your project scripts:  scp -r scripts/ external/ user@VM:~/ascension/"
echo "  3. Copy ModTheSpire + CommunicationMod into game/mods/"
echo "  4. Copy your model checkpoint:  scp models/ppo_sts.pt user@VM:~/ascension/models/"
echo "  5. Copy your seed file:         scp seeds/eval_200.txt user@VM:~/ascension/seeds/"
echo ""
echo "Or use:  vm/sync.sh push   (from your local machine)"
echo ""
echo "Setup complete."
