#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f sim_logs/sim_training_varMsmoke.csv
timeout 220 python train_search.py 12000 4 4000 varMsmoke 2>&1 | tail -n 30
echo "--- npz has zmean/zstd? ---"
python - <<'PY'
import numpy as np
w = np.load("policy_live.npz")
print("zmean in npz:", "zmean" in w.files, " zstd in npz:", "zstd" in w.files)
if "zmean" in w.files:
    print("zmean=%.4f zstd=%.4f" % (float(w["zmean"]), float(w["zstd"])))
PY
pkill -f '[s]im_worker_search' 2>/dev/null
rm -rf sim_models_varMsmoke sim_logs/sim_training_varMsmoke.csv
