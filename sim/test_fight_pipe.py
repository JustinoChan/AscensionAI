"""Drive sim_worker_fight.py over the real pipe protocol with random actions:
stability, episode shape, outcome mix, and reset_seed determinism."""
import subprocess
import struct
import pickle
import sys
import random
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = ["bank_train.jsonl", "fight_config.json"]


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
deaths = fights = steps = wins = losses = caps = 0
eplen = []
ferr = open(os.path.join(HERE, "fight_worker_stderr.log"), "wb")
attempt = 0
while fights < 60 and deaths < 12:
    attempt += 1
    p = subprocess.Popen(
        [sys.executable, "sim_worker_fight.py", str(2000 + attempt)] + ARGS,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=ferr, cwd=HERE)
    pk = _recv(p)
    if pk is None:
        deaths += 1
        print("startup death #%d (attempt %d)" % (deaths, attempt), flush=True)
        continue
    mask = pk[3]
    cur = 0
    while True:
        legal = [i for i in range(128) if mask[i]]
        try:
            _send(p, ("step", rng.choice(legal)))
            pk = _recv(p)
        except Exception:
            pk = None
        if pk is None:
            deaths += 1
            print("mid-run death #%d after %d steps %d fights"
                  % (deaths, steps, fights), flush=True)
            break
        steps += 1
        cur += 1
        mask = pk[3]
        if pk[2]:
            fights += 1
            eplen.append(cur)
            cur = 0
            oc = pk[5]
            wins += oc == 1
            losses += oc == -1
            caps += oc == 0
            if fights >= 60:
                break
    try:
        p.kill()
    except Exception:
        pass
ferr.close()
avg = sum(eplen) / max(1, len(eplen))
print("RESULT: fights=%d steps=%d deaths=%d | win=%d loss=%d cap=%d | avg_ep=%.0f"
      % (fights, steps, deaths, wins, losses, caps, avg))

# eval-path determinism: same entry index -> byte-identical first obs
p = subprocess.Popen(
    [sys.executable, "sim_worker_fight.py", "1"] + ARGS,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, cwd=HERE)
_recv(p)
_send(p, ("reset_seed", 5))
a = _recv(p)
_send(p, ("reset_seed", 5))
b = _recv(p)
same = (a[0] == b[0]).all() and (a[3] == b[3]).all()
print("reset_seed determinism:", "PASS" if same else "FAIL")
_send(p, ("close",))
