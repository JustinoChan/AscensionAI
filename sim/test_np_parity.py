"""NpNet forward must match the torch net on the same obs (greedy action +
value within tolerance)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from train_sim_mp import ActorCritic  # separate-trunk net (OBS=1235)
from np_policy import NpNet

ck = "sim_models_varJ/ppo_sim_latest.pt"
net = ActorCritic()
net.load_state_dict(torch.load(ck, map_location="cpu")["model"])
net.eval()

import subprocess
subprocess.check_call([sys.executable, "export_live.py", ck, "parity.npz"])
npn = NpNet("parity.npz")

rng = np.random.default_rng(0)
mism = 0
verr = 0.0
for _ in range(200):
    obs = rng.random(1235).astype(np.float32)
    mask = (rng.random(128) > 0.5).astype(np.int8)
    mask[0] = 1
    with torch.no_grad():
        lg, v = net(torch.as_tensor(obs)[None])
        lg = lg[0].numpy()
        v = float(v[0])
    lg = np.where(mask > 0, lg, -1e9)
    ta = int(np.argmax(lg))
    na = npn.greedy(obs, mask)
    mism += (ta != na)
    verr = max(verr, abs(v - npn.value(obs)))
print("greedy mismatches: %d/200 | max value err: %.5f" % (mism, verr))
assert mism == 0 and verr < 1e-3, "NumPy/torch parity failed"
print("PASS")
