"""Count harvested data: python3 harvest_status.py [dir]"""
import glob, sys, numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "sim_logs/harvest/h1"
fs = sorted(glob.glob(d + "/seed_*.npz"))
nf = nd = nm = w = 0
for f in fs:
    z = np.load(f)
    nf += len(z["f_won"]); nd += len(z["c_chosen"]); nm += len(z["m_action"]); w += int(z["run"][2])
print("seeds %d  fights %d  combat decisions %d  macro decisions %d  wins %d" % (len(fs), nf, nd, nm, w))
