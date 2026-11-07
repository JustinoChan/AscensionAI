import sys
import numpy as np
import torch
ckpt, out = sys.argv[1], sys.argv[2]
sd = torch.load(ckpt, map_location="cpu")["model"]
np.savez(out, **{k: v.numpy() for k, v in sd.items()})
print("exported %s -> %s (%d tensors)" % (ckpt, out, len(sd)))
