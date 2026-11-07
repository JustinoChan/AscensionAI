import subprocess, struct, pickle, sys, os
N = 8
def recv(p):
    h = p.stdout.read(4)
    if len(h) < 4: return None
    n = struct.unpack(">I", h)[0]; b = b""
    while len(b) < n:
        c = p.stdout.read(n-len(b))
        if not c: return None
        b += c
    return pickle.loads(b)
ps = []
ok = 0
for i in range(N):
    p = subprocess.Popen([sys.executable, "sim_worker.py", str(i*100003)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=os.getcwd())
    o = recv(p)          # handshake: wait until THIS worker is ready
    if o is not None: ok += 1
    else: print("worker %d FAILED rc=%s" % (i, p.poll()))
    ps.append(p)
print("RESULT serial-handshake: %d/%d workers OK" % (ok, N))
for p in ps: p.kill()
