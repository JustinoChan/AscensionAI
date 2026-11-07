#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
cp policy_live.npz policy_varL.npz && echo "froze varL combat net -> policy_varL.npz"
echo "--- EVAL_SEEDS def in train_sim_mp.py ---"
grep -n "EVAL_SEEDS" train_sim_mp.py | head
echo "--- npz keys ---"
python - <<'PY'
import numpy as np
for name in ("policy_varE.npz","policy_varL.npz"):
    w=np.load(name)
    print(name, "->", sorted(w.files))
PY
