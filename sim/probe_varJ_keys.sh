#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python - <<'EOF'
import torch, glob, os
# newest varJ checkpoint
cks = sorted(glob.glob("sim_models_varJ/ppo_sim_*.pt"))
print("varJ checkpoints:", [os.path.basename(c) for c in cks][-3:])
ck = torch.load(cks[-1], map_location="cpu")
print("ckpt keys:", list(ck.keys()))
print("model state_dict keys:")
for k, v in ck["model"].items():
    print("  ", k, tuple(v.shape))
EOF
