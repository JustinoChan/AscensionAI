#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
python - <<'PY'
import numpy as np, os
from np_policy import NpNet
w = dict(np.load("policy_varL.npz"))
w["zmean"] = np.float32(7.5); w["zstd"] = np.float32(3.0)
np.savez("tmp_vd.npz", **w)
net = NpNet("tmp_vd.npz")
obs = np.zeros(1235, np.float32)
raw = net.value(obs)
den = net.value_denorm(obs)
ok = abs(den - (raw*3.0 + 7.5)) < 1e-4 and abs(net.zmean-7.5)<1e-4 and abs(net.zstd-3.0)<1e-4
print("raw=%.4f denorm=%.4f expect=%.4f -> %s" % (raw, den, raw*3.0+7.5, "PASS" if ok else "FAIL"))
os.remove("tmp_vd.npz")
assert ok
PY
