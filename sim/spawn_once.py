"""Spawn one debug worker exactly like the trainer does (Popen, stdin/stdout
pipes), read the handshake, then close. Its stderr (DBG line) goes to the
inherited stderr so it shows in the terminal."""
import subprocess
import struct
import pickle
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = sys.argv[1] if len(sys.argv) > 1 else "sim_worker.py"
p = subprocess.Popen([sys.executable, WORKER, "1"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE)
h = p.stdout.read(4)
if len(h) < 4:
    print("worker died at startup (lottery) -- rerun")
    sys.exit(1)
n = struct.unpack(">I", h)[0]
buf = b""
while len(buf) < n:
    c = p.stdout.read(n - len(buf))
    if not c:
        break
    buf += c
pk = pickle.loads(buf)
import numpy as np
c = np.rint(np.asarray(pk[0][19:239]) * 7.0)
nz = np.nonzero(c)[0]
print("PARENT sees normalized deck slots:", [(int(i), int(c[i])) for i in nz])
p.kill()
