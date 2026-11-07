"""Dump a torch checkpoint's model state_dict to an npz NpNet can read.
Atomic (temp .npz + rename) so workers never read a partial file. The temp
name ends in .npz so np.savez does not append a second suffix.
argv: [1] checkpoint .pt   [2] out .npz (default policy_live.npz)"""
import os
import sys
import numpy as np
import torch

ckpt = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "policy_live.npz"
sd = torch.load(ckpt, map_location="cpu")["model"]
tmp = out + ".tmp.npz"
np.savez(tmp, **{k: v.numpy() for k, v in sd.items()})
os.replace(tmp, out)
print("exported %d tensors -> %s" % (len(sd), out))
