import subprocess, struct, pickle, sys, os, time

p = subprocess.Popen([sys.executable, "sim_worker.py", "0"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(3)
h = p.stdout.read(4)
print("header len", len(h), repr(h[:8]))
if len(h) == 4:
    n = struct.unpack(">I", h)[0]
    body = b""
    while len(body) < n:
        c = p.stdout.read(n - len(body))
        if not c:
            break
        body += c
    print("body", len(body), "of", n)
    try:
        obj = pickle.loads(body)
        print("OK obs", obj[0].shape, "reward", float(obj[1]), "mask_sum", int(obj[3].sum()))
    except Exception as e:
        print("unpickle err", e)
else:
    print("worker exited rc", p.poll())
    print("STDERR:\n", p.stderr.read().decode()[:2000])
p.kill()
