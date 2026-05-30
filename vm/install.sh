#!/bin/bash
# AscensionAI VM Installer — one-shot setup for a fresh Ubuntu 22.04 VM.
#
# Run this ONCE on the VM after creating it. Idempotent: safe to re-run.
#
#     bash vm/install.sh
#
# What it does:
#   1. Installs system packages (Java 8, Xvfb, OpenAL, Python venv tooling)
#   2. Pins Java 8 as the default `java`  (ModTheSpire needs 8, NOT 17 — see note)
#   3. Creates the Python venv and installs torch (CPU) + numpy + gymnasium
#   4. Builds the ~/ascension directory layout
#   5. Links mod jars into game/mods/ and writes a low-res display config
#   6. Validates the install and prints exactly what to copy next
#
# It does NOT copy the Slay the Spire game files or your model — those are
# yours and are pushed from your local machine (see vm/sync.ps1 / sync.sh,
# or the printed instructions at the end).
#
# ─── Why these choices (hard-won; don't "simplify" them away) ────────────────
#   * Java 8        — ModTheSpire fails to load mods on Java 17+ (module access
#                     changes break BaseMod/SuperFastMode). The game ships for 8.
#   * libopenal1    — STS/LWJGL dlopen's libopenal.so.1 at startup; without it
#                     every instance crashes before reaching the menu.
#   * Xvfb + softGL — there's no GPU, so we render on a virtual display with a
#                     software GL stack. run_training.sh gives each worker its
#                     OWN Xvfb (shared display = ~100x slowdown) and its OWN
#                     java.io.tmpdir (shared tmpdir = LWJGL extraction SIGSEGV).
#   * 2 GB heap     — workers OOM (-Xmx512m) after ~35 games and silently die;
#                     run_training.sh uses -Xmx2048m + --restart-every 25.

set -eo pipefail

PROJECT_DIR="$HOME/ascension"
GAME_DIR="$PROJECT_DIR/game"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=========================================="
echo "  AscensionAI VM Installer"
echo "=========================================="
echo ""

# ─── 1. System packages ─────────────────────────────────────────────────────
echo "[1/6] Installing system packages (this can take a minute)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    openjdk-8-jre-headless \
    xvfb \
    libopenal1 \
    python3 python3-pip python3-venv \
    git unzip rsync tmux htop
echo "    Done."
echo ""

# ─── 2. Pin Java 8 as the default ───────────────────────────────────────────
echo "[2/6] Pinning Java 8 as the default 'java'..."
JAVA8="$(update-alternatives --list java 2>/dev/null | grep 'java-8' | head -1 || true)"
if [ -n "$JAVA8" ]; then
    sudo update-alternatives --set java "$JAVA8" >/dev/null
fi
JAVA_VER="$(java -version 2>&1 | head -1)"
echo "    Active java: $JAVA_VER"
if echo "$JAVA_VER" | grep -q '1\.8\.'; then
    echo "    OK — Java 8 active."
else
    echo "    !! WARNING: Java 8 is NOT the default. Mods will fail to load."
    echo "    !! Fix:  sudo update-alternatives --config java   (pick java-8)"
fi
echo ""

# ─── 3. Python environment ──────────────────────────────────────────────────
echo "[3/6] Creating Python venv + installing torch/numpy/gymnasium..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade -q pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy gymnasium
echo "    torch:     $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
echo "    numpy:     $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo MISSING)"
echo "    gymnasium: $(python3 -c 'import gymnasium; print(gymnasium.__version__)' 2>/dev/null || echo MISSING)"
echo ""

# ─── 4. Directory layout ────────────────────────────────────────────────────
echo "[4/6] Creating directory layout under $PROJECT_DIR ..."
mkdir -p "$PROJECT_DIR"/{game,scripts,external,models,rollouts_shared,logs,seeds,Eval}
echo "    Done."
echo ""

# ─── 5. Game-dependent setup (only if game files are already present) ────────
echo "[5/6] Configuring game directory..."
if [ -f "$GAME_DIR/desktop-1.0.jar" ]; then
    # Link mod jars into game/mods/ — ModTheSpire loads mods from there.
    mkdir -p "$GAME_DIR/mods"
    for jar in BaseMod.jar CommunicationMod.jar SuperFastMode.jar; do
        if [ -f "$GAME_DIR/$jar" ]; then
            ln -sf "$GAME_DIR/$jar" "$GAME_DIR/mods/$jar"
        else
            echo "    !! Missing mod jar: $GAME_DIR/$jar"
        fi
    done

    # Low-resolution display config (width/height/fps/fullscreen/vsync/...).
    # Resolution barely affects CPU on software GL, but small is harmless.
    if [ ! -f "$GAME_DIR/info.displayconfig" ]; then
        printf '1280\n720\n60\nfalse\nfalse\ntrue\n' > "$GAME_DIR/info.displayconfig"
        echo "    Wrote info.displayconfig"
    fi
    echo "    Game directory configured."
else
    echo "    (game files not copied yet — skipping mod links / display config)"
    echo "    Re-run this script after copying the game, or just run training;"
    echo "    run_training.sh links the mods itself on launch."
fi
echo ""

# ─── 6. Validation ──────────────────────────────────────────────────────────
echo "[6/6] Validating install..."
ok=true
check() {  # check <label> <test-cmd...>
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "    [ OK ] $label"
    else
        echo "    [FAIL] $label"
        ok=false
    fi
}
check "Java 8 active"           bash -c 'java -version 2>&1 | grep -q 1\.8\.'
check "Xvfb installed"          command -v Xvfb
check "libopenal.so.1 present"  test -e /usr/lib/x86_64-linux-gnu/libopenal.so.1
check "venv torch importable"   python3 -c 'import torch, numpy, gymnasium'
check "project dirs exist"      test -d "$PROJECT_DIR/rollouts_shared"
echo ""

echo "=========================================="
if [ "$ok" = true ]; then
    echo "  Install OK."
else
    echo "  Install finished WITH WARNINGS (see [FAIL] above)."
fi
echo "=========================================="
echo ""
echo "Still needed (push from your LOCAL machine):"
echo ""
echo "  1. Game files  -> ~/ascension/game/"
echo "       gcloud compute scp --recurse \\"
echo "         \"C:\\Program Files (x86)\\Steam\\steamapps\\common\\SlayTheSpire\\*\" \\"
echo "         ascension-vm:~/ascension/game/ --zone=us-west1-a"
echo "     (must include desktop-1.0.jar, ModTheSpire.jar, BaseMod.jar,"
echo "      CommunicationMod.jar, SuperFastMode.jar)"
echo ""
echo "  2. Code + model + seeds:"
echo "       .\\vm\\sync.ps1 push          (Windows)"
echo "       ./vm/sync.sh push            (Linux/Mac)"
echo ""
echo "Then start training:"
echo "       cd ~/ascension && ./vm/run_training.sh --workers 8 --hours 12"
echo ""
