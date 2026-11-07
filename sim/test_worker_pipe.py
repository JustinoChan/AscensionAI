"""Drive sim_worker.py subprocesses over the real pipe protocol with stderr
captured to worker_stderr.log, to see HOW workers die in worker mode:
Python traceback (logic bug) vs silence (native segfault).

argv: [1] worker filename (default sim_worker.py) -- works with both the
6-field (varB) and 8-field (varC) reply tuples.
"""
import subprocess
import struct
import pickle
import sys
import random
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = sys.argv[1] if len(sys.argv) > 1 else "sim_worker.py"


def _send(p, obj):
    b = pickle.dumps(obj, protocol=4)
    p.stdin.write(struct.pack(">I", len(b)))
    p.stdin.write(b)
    p.stdin.flush()


def _recv(p):
    h = p.stdout.read(4)
    if len(h) < 4:
        return None
    n = struct.unpack(">I", h)[0]
    buf = b""
    while len(buf) < n:
        c = p.stdout.read(n - len(buf))
        if not c:
            return None
        buf += c
    return pickle.loads(buf)


rng = random.Random(11)
deaths = 0
games = 0
steps = 0
err_path = os.path.join(HERE, "worker_stderr.log")
ferr = open(err_path, "wb")
attempt = 0
while games < 60 and deaths < 12:
    attempt += 1
    p = subprocess.Popen(
        [sys.executable, WORKER, str(1000 + attempt), "0.99"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=ferr, cwd=HERE)
    pk = _recv(p)
    if pk is None:
        deaths += 1
        print("startup death #%d (attempt %d)" % (deaths, attempt), flush=True)
        continue
    mask = pk[3]
    while True:
        legal = [i for i in range(128) if mask[i]]
        try:
            _send(p, ("step", rng.choice(legal)))
            pk = _recv(p)
        except Exception:
            pk = None
        if pk is None:
            deaths += 1
            print("mid-run death #%d after %d steps %d games (attempt %d)"
                  % (deaths, steps, games, attempt), flush=True)
            break
        steps += 1
        mask = pk[3]
        if pk[2]:
            games += 1
            if games % 10 == 0:
                ds = pk[6] if len(pk) > 6 else "n/a"
                ph = pk[7] if len(pk) > 7 else 0.0
                print("games=%d steps=%d deck=%s phi=%.2f"
                      % (games, steps, ds, ph), flush=True)
            if games >= 60:
                break
    try:
        p.kill()
    except Exception:
        pass
ferr.flush()
ferr.close()
print("RESULT: games=%d steps=%d deaths=%d" % (games, steps, deaths))
print("---- worker_stderr.log tail ----")
data = open(err_path, "rb").read()[-3000:]
sys.stdout.buffer.write(data)
print()
