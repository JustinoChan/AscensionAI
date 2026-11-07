import subprocess, struct, pickle, sys, os, time
N = 8
ps = []
for i in range(N):
    p = subprocess.Popen([sys.executable, "sim_worker.py", str(i*100003)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=os.getcwd())
    ps.append(p)
    time.sleep(0.4)   # stagger startup
def recv(p):
    h = p.stdout.read(4)
    if len(h) < 4: return None
    n = struct.unpack(">I", h)[0]; b = b""
    while len(b) < n:
        c = p.stdout.read(n-len(b))
        if not c: return None
        b += c
    return pickle.loads(b)
ok = sum(1 for p in ps if recv(p) is not None)
print("RESULT staggered: %d/%d workers OK" % (ok, N))
for p in ps: p.kill()
