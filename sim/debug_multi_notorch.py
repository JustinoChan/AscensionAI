import subprocess, struct, pickle, sys, os   # NO torch
N = 8
ps = []
for i in range(N):
    p = subprocess.Popen([sys.executable, "sim_worker.py", str(i*100003)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=os.getcwd())
    ps.append(p)
def recv(p):
    h = p.stdout.read(4)
    if len(h) < 4: return None
    n = struct.unpack(">I", h)[0]; b = b""
    while len(b) < n:
        c = p.stdout.read(n-len(b))
        if not c: return None
        b += c
    return pickle.loads(b)
ok = 0
for i, p in enumerate(ps):
    o = recv(p)
    if o is None:
        print("worker %d FAILED rc=%s" % (i, p.poll()))
    else:
        ok += 1
print("RESULT: %d/%d workers OK (no torch in parent)" % (ok, N))
for p in ps: p.kill()
