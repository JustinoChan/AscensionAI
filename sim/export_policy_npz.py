"""Export an ActorCritic checkpoint to .npz for the torch-free harvest.
argv: [1] checkpoint .pt   [2] out .npz
Checkpoint format (save_ckpt in train_sim_mp.py): dict with key "model" =
state_dict of body.{0,2,4}.{weight,bias} (GELU MLP 1200-512-256-256),
pi.{weight,bias} (256-128), v.{weight,bias} (unused here)."""
import sys
import numpy as np
import torch

ck = torch.load(sys.argv[1], map_location="cpu")
sd = ck["model"]
np.savez(sys.argv[2], **{k: v.numpy() for k, v in sd.items()})
print("exported %d tensors -> %s" % (len(sd), sys.argv[2]))
