"""Checkpoint forensics: did the value head ever move from init?
Prints v-head weight norms across checkpoints + a fresh-init reference.
argv: checkpoint paths."""
import sys
import torch
import torch.nn as nn

ref = nn.Linear(256, 1)
print("fresh-init reference: v.w norm %.4f  v.b %.4f" % (
    ref.weight.norm(), ref.bias.item()))
for p in sys.argv[1:]:
    try:
        ck = torch.load(p, map_location="cpu")
        sd = ck["model"]
        vw, vb = sd["v.weight"], sd["v.bias"]
        pw = sd["pi.weight"]
        b0 = sd["body.0.weight"]
        print("%-42s v.w norm %8.4f  max %7.4f | v.b %8.4f | "
              "pi.w norm %8.4f | body0 norm %8.4f" % (
                  p, vw.norm(), vw.abs().max(), vb.item(), pw.norm(), b0.norm()))
    except Exception as e:
        print(p, "ERROR", e)
