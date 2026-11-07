import glob, collections
import numpy as np
parts = sorted(glob.glob("teacher_part_*.npz"))
obs, act, mask, room = [], [], [], []
for p in parts:
    d = np.load(p)
    obs.append(d["obs"]); act.append(d["act"]); mask.append(d["mask"]); room.append(d["room"])
obs = np.concatenate(obs); act = np.concatenate(act)
mask = np.concatenate(mask); room = np.concatenate(room)
np.savez("teacher_data.npz", obs=obs, act=act, mask=mask, room=room)
print("total %d samples; tiers: %s" % (len(room), dict(collections.Counter(room.tolist()))))
