import torch  # import torch FIRST (like the trainer)
import subprocess, struct, pickle, sys, os, time
p = subprocess.Popen([sys.executable, "sim_worker.py", "0"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(3)
h = p.stdout.read(4)
print("torch imported in parent; header len", len(h))
if len(h) < 4:
    print("rc", p.poll(), "STDERR:", p.stderr.read().decode()[:1500])
else:
    print("WORKER OK despite torch in parent")
p.kill()
